"""Vectorised PyTorch simulator of exactly the circuit defined in circuit.py.

PennyLane (src/circuit.py) is the reference definition of the model. Running
tens of thousands of training updates through PennyLane's generic simulator
is more than 10x too slow for the required number of seeds, so the experiments execute
the *same* circuit with this specialised simulator. It was checked against
PennyLane on real MNIST inputs (expectation values, gradients, parameter-shift
values and finite-shot distributions all agree); the notebook repeats the
expectation-value check.

Conventions: wire 0 is the most significant bit of the basis index (PennyLane
convention). Weights have shape (L, n, 2): [..., 0] -> RY angle, [..., 1] -> RZ.
"""
from __future__ import annotations

import numpy as np
import torch

CDT = torch.complex128
RDT = torch.float64


class Engine:
    def __init__(self, n_qubits: int, cost: str = "local"):
        self.n = n = n_qubits
        self.dim = 2 ** n
        idx = torch.arange(self.dim)
        # bits[z, q] = value of qubit q in basis state z (q=0 most significant)
        self.bits = torch.stack([(idx >> (n - 1 - q)) & 1 for q in range(n)], dim=1).to(RDT)
        self.spin = 1.0 - 2.0 * self.bits                      # +1 for |0>, -1 for |1>
        if n == 1:
            pairs = []
        elif n == 2:
            pairs = [(0, 1)]
        else:
            pairs = [(q, (q + 1) % n) for q in range(n)]
        cz = torch.ones(self.dim, dtype=RDT)
        for a, b in pairs:
            cz = cz * (1.0 - 2.0 * self.bits[:, a] * self.bits[:, b])
        self.cz_sign = cz
        self.cost = cost
        if cost == "local":
            self.obs = self.spin[:, 0].clone()
        elif cost == "global":
            self.obs = self.spin.prod(dim=1)
        else:
            raise ValueError(cost)

    # ------------------------------------------------------------------ pure
    def embed(self, x):
        """AngleEmbedding(rotation='Y') of x (N, n) applied to |0...0>. Real product state."""
        x = torch.as_tensor(x, dtype=RDT)
        c, s = torch.cos(x / 2), torch.sin(x / 2)
        psi = torch.stack([c[:, 0], s[:, 0]], dim=1)
        for q in range(1, self.n):
            v = torch.stack([c[:, q], s[:, q]], dim=1)
            psi = (psi[:, :, None] * v[:, None, :]).reshape(x.shape[0], -1)
        return psi.to(CDT)

    def apply_y(self, psi, q):
        N = psi.shape[0]
        p = psi.reshape(-1, 2 ** q, 2, 2 ** (self.n - q - 1))
        a, b = p[:, :, 0, :], p[:, :, 1, :]
        return torch.stack([-1j * b, 1j * a], dim=2).reshape(N, -1)

    def diag(self, wz):
        """Diagonal of (prod_q RZ_q(wz[q])) followed by the CZ ring. wz: (n,)."""
        phase = -0.5 * (self.spin @ wz)                         # (dim,)
        return torch.polar(self.cz_sign.abs(), phase) * self.cz_sign

    @staticmethod
    def kron_ry(thetas):
        """Kronecker product of RY(theta_q) matrices (real), qubit order as given."""
        K = torch.ones(1, 1, dtype=RDT)
        for t in thetas:
            c, s = torch.cos(t / 2), torch.sin(t / 2)
            K = torch.kron(K, torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
        return K

    def layer(self, psi, w, return_mid=False):
        """One trainable layer. All RY gates of the layer equal K_A (x) K_B
        (first / second half of the qubits), applied as two matrix products."""
        n, N = self.n, psi.shape[0]
        nA = n // 2
        KA = self.kron_ry(w[:nA, 0]).to(CDT)
        KB = self.kron_ry(w[nA:, 0]).to(CDT)
        p = psi.reshape(N, 2 ** nA, 2 ** (n - nA))
        p = torch.matmul(torch.matmul(KA, p), KB.transpose(0, 1))
        psi = p.reshape(N, -1)
        mid = psi
        psi = psi * self.diag(w[:, 1])
        return (psi, mid) if return_mid else psi

    def state(self, x, weights):
        psi = self.embed(x)
        for l in range(weights.shape[0]):
            psi = self.layer(psi, weights[l])
        return psi

    def expval(self, x, weights):
        psi = self.state(x, weights)
        return (psi.real ** 2 + psi.imag ** 2) @ self.obs

    def expval_chunked(self, x, weights, chunk=2048):
        x = torch.as_tensor(x, dtype=RDT)
        return torch.cat([self.expval(x[i:i + chunk], weights) for i in range(0, x.shape[0], chunk)])

    # ------------------------------------------------- parameter-shift values
    def shift_values(self, x, weights, trainable_layers):
        """Exact values f(theta +- pi/2 e_k) for every parameter k in `trainable_layers`.

        Uses R(theta +- pi/2) = (I -+ iG) R(theta)/sqrt(2) (G = Y or Z), so the
        two shifted final states are (Psi -+ i chi_k)/sqrt(2), where chi_k is the
        final state with the generator G inserted after gate k. Then
            f_+-  = (<Psi|O|Psi> + <chi|O|chi>)/2  +-  Im<Psi|O|chi>.
        Returns f0 (N,), fplus, fminus with shape (N, len(layers), n, 2).
        No approximation is involved; values equal PennyLane's shifted tapes.
        """
        with torch.no_grad():
            weights = torch.as_tensor(weights, dtype=RDT)
            L = weights.shape[0]
            N = x.shape[0]
            psi = self.embed(x)
            mids, posts = [], []
            for l in range(L):
                psi, mid = self.layer(psi, weights[l], return_mid=True)
                mids.append(mid)
                posts.append(psi)
            Psi = psi
            obsPsi = Psi * self.obs
            f0 = (Psi.real ** 2 + Psi.imag ** 2) @ self.obs
            layers = list(trainable_layers)
            fp = torch.zeros(N, len(layers), self.n, 2, dtype=RDT)
            fm = torch.zeros_like(fp)
            for j, l in enumerate(layers):
                d = self.diag(weights[l, :, 1])
                chis = []
                for q in range(self.n):              # Y inserted after RY_q
                    chis.append(self.apply_y(mids[l], q) * d)
                for q in range(self.n):              # Z inserted after RZ_q (commutes with diag)
                    chis.append(posts[l] * self.spin[:, q])
                chi = torch.cat(chis, dim=0)          # (2n*N, dim)
                for l2 in range(l + 1, L):
                    chi = self.layer(chi, weights[l2])
                chi = chi.reshape(2 * self.n, N, self.dim)
                B = (chi.real ** 2 + chi.imag ** 2) @ self.obs           # (2n, N)
                im = (obsPsi.conj()[None] * chi).sum(-1).imag            # (2n, N)
                base = 0.5 * (f0[None] + B)
                plus, minus = base + im, base - im
                fp[:, j, :, 0], fp[:, j, :, 1] = plus[: self.n].T, plus[self.n:].T
                fm[:, j, :, 0], fm[:, j, :, 1] = minus[: self.n].T, minus[self.n:].T
            return f0, fp, fm

def sample_expval(f, shots, rng):
    """Finite-shot estimate of a +-1-valued observable with exact mean f.

    Measuring S shots of an observable with eigenvalues +-1 yields k outcomes
    +1 with k ~ Binomial(S, (1+f)/2); the estimator is 2k/S - 1. This is the
    exact sampling distribution of a shot-based device for Z_0 or Z_0...Z_n-1.
    """
    f = np.asarray(f, dtype=np.float64)
    p = np.clip((1.0 + f) / 2.0, 0.0, 1.0)
    k = rng.binomial(shots, p)
    return 2.0 * k / shots - 1.0
