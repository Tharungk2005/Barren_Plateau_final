"""The variational quantum circuit (PennyLane).

Architecture (per sample x with n features, one per qubit):
    AngleEmbedding(x, rotation="Y")                     # data encoding
    for each layer l:                                    # trainable block
        RY(w[l, q, 0]) RZ(w[l, q, 1])  on every qubit q
        CZ ring: CZ(q, q+1 mod n) for q = 0..n-1
    readout  f(x) = <Z_0>              (local cost)
          or f(x) = <Z_0 Z_1 ... Z_n-1> (global cost)
    classical head  p(y=1|x) = sigmoid(a * f(x) + b)

Weights have shape (n_layers, n_qubits, 2).

Why a CZ ring (rather than CNOT): CZ is diagonal in the computational basis,
so it commutes with every Z-type readout. A new layer whose rotation angles
are zero therefore leaves f(x) exactly unchanged when it is appended at the
end of the circuit, which is what makes "near-identity" layer insertion in
layer-wise learning well defined. CZ and CNOT differ only by Hadamards on the
target, so the entangling power of the ring is the same. Skolik et al. (2021)
also used CZ entanglers.
"""
from __future__ import annotations

import pennylane as qml


def entangler(n_qubits: int):
    if n_qubits == 1:
        return
    pairs = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    if n_qubits == 2:
        pairs = pairs[:1]
    for a, b in pairs:
        qml.CZ(wires=[a, b])


def observable(n_qubits: int, cost: str):
    if cost == "local":
        return qml.PauliZ(0)
    if cost == "global":
        return qml.prod(*[qml.PauliZ(q) for q in range(n_qubits)])
    raise ValueError(cost)


def make_qnode(n_qubits: int, n_layers: int, cost: str = "local",
               device: str = "default.qubit", diff_method: str = "backprop",
               shots=None, interface: str = "torch"):
    """Return a PennyLane QNode f(x, weights) -> expectation value (broadcast over a batch)."""
    dev = qml.device(device, wires=n_qubits)

    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")     # data encoding
        for l in range(n_layers):                                       # trainable layers
            for q in range(n_qubits):
                qml.RY(weights[..., l, q, 0], wires=q)
                qml.RZ(weights[..., l, q, 1], wires=q)
            entangler(n_qubits)
        return qml.expval(observable(n_qubits, cost))                   # readout

    qnode = qml.QNode(circuit, dev, interface=interface, diff_method=diff_method)
    if shots is not None:
        qnode = qml.set_shots(qnode, shots=shots)
    return qnode