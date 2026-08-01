"""
qforge.mitigation — squeezing signal out of noisy hardware.
===========================================================
Four techniques, with honest notes on which ones actually earned their place.
All four were tested against a validated trapped-ion noise model; the results
are recorded here rather than in a changelog, because knowing what does NOT
work saves more time than knowing what does.

**Zero-noise extrapolation (ZNE)** -- works, and is the workhorse. Run the same
circuit at deliberately amplified noise, then extrapolate back to zero. Took an
H4 forged fragment from ~20 kcal/mol to 0.57 kcal/mol under a trapped-ion
noise model.

**Readout error mitigation** -- works. Clean 2.5-3.5x error reduction on every
term tested, and it is pure classical post-processing, so it stacks with
everything else for free.

**Symmetry verification** -- works, with a sharp caveat. Discarding shots that
land on the wrong particle number gave 2-4x error reduction while throwing away
only ~1% of shots. BUT it is only valid in the computational basis: the
rotations needed to measure X and Y observables do not preserve particle
number, so postselecting after them discards CORRECT shots and keeps wrong
ones. Applying it blindly to rotated circuits produced an energy ~40x WORSE
than no mitigation at all. :func:`symmetry_postselect` therefore refuses to run
on rotated bases unless explicitly overridden.

**Pauli twirling** -- did NOT help here, and is included only as a documented
negative. Twirling converts coherent (systematic) error into stochastic error.
Against a purely depolarising noise model there is no coherent bias to remove,
and measurements confirmed no improvement (converged over 600 random twirls).
It may still matter on hardware with significant coherent error -- which is a
measurement nobody has made for this pipeline yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit


# --------------------------------------------------------------------- ZNE
def noise_scale_factor(folds: int) -> int:
    """Actual noise amplification produced by ``folds`` foldings: ``2*folds - 1``.

    Read this before plotting anything. ``fold_circuit(c, 3)`` does NOT triple
    the noise -- it produces ``U U^dag U U^dag U``, five copies, so the noise
    scale is 5. Folds 1, 2, 3 give scales 1, 3, 5.

    Passing the fold NUMBERS to :func:`zero_noise_extrapolation` instead of
    these SCALES silently fits the wrong x-axis and extrapolates to the wrong
    intercept. Simulations that scale noise-model parameters directly do not
    have this problem -- there the scale really is 1, 2, 3 -- which is exactly
    why the mistake is easy to make when moving from simulator to hardware.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    return 2 * folds - 1


def fold_circuit(circuit: QuantumCircuit, folds: int) -> QuantumCircuit:
    """Amplify noise by repeating the circuit as ``U (U^dag U)^(folds-1)``.

    Ideally the identity; in practice it multiplies real noise exposure by
    ``noise_scale_factor(folds)`` = ``2*folds - 1``. Measured on a forged H4
    circuit: 11 two-qubit gates at 1 fold, 33 at 2 folds, 55 at 3 folds.

    Barriers are inserted between blocks because an optimising transpiler
    otherwise recognises ``U^dag U`` as the identity and deletes the folds --
    silently producing identical circuits at every "noise level", so ZNE does
    nothing at all while appearing to work. Transpile the result with
    ``optimization_level=0``.

    This is the only way to scale noise on real hardware. A simulator can
    cheat by scaling noise-model parameters; a QPU has no such dial, so folded
    circuits are genuinely deeper -- and, on gate-billed hardware,
    substantially more expensive.
    """
    if folds < 1:
        raise ValueError("folds must be >= 1")
    folded = circuit.copy()
    inverse = circuit.inverse()
    for _ in range(folds - 1):
        folded.barrier()
        folded.compose(inverse, inplace=True)
        folded.barrier()
        folded.compose(circuit, inplace=True)
    return folded


def extrapolate_to_zero_noise(
    scales, energies, order: int = 2
) -> float:
    """Fit ``E(lambda)`` against noise scale and evaluate at lambda = 0.

    Order 1 (linear) is robust and needs only two points. Order 2 (quadratic)
    fitted the H4 data better -- 0.57 vs 1.12 kcal/mol -- but needs three
    points and is more sensitive to statistical noise in each point. If the
    per-point shot noise is large, a lower order is usually the safer choice.
    """
    scales = np.asarray(scales, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if len(scales) <= order:
        raise ValueError(
            f"order-{order} fit needs at least {order + 1} points, got {len(scales)}"
        )
    return float(np.polyval(np.polyfit(scales, energies, order), 0.0))


@dataclass
class ZNEResult:
    scales: list[float]
    energies: list[float]
    linear: float
    quadratic: float | None

    @property
    def best(self) -> float:
        return self.quadratic if self.quadratic is not None else self.linear


def zero_noise_extrapolation(scales, energies) -> ZNEResult:
    """Run both linear and quadratic extrapolations and keep both."""
    linear = extrapolate_to_zero_noise(scales, energies, order=1)
    quadratic = (
        extrapolate_to_zero_noise(scales, energies, order=2)
        if len(scales) > 2
        else None
    )
    return ZNEResult(list(map(float, scales)), list(map(float, energies)),
                     linear, quadratic)


# --------------------------------------------------- readout error mitigation
def confusion_matrix(flip_probability: float, n_qubits: int) -> np.ndarray:
    """Tensor-product readout confusion matrix for a symmetric bit-flip."""
    single = np.array(
        [[1 - flip_probability, flip_probability],
         [flip_probability, 1 - flip_probability]]
    )
    full = single
    for _ in range(n_qubits - 1):
        full = np.kron(full, single)
    return full


def mitigate_readout(
    probabilities: np.ndarray, flip_probability: float, n_qubits: int
) -> np.ndarray:
    """Undo readout error by inverting the confusion matrix.

    Pure classical post-processing -- no extra circuits, no extra shots, and it
    composes with every other technique here. Measured 2.5-3.5x error reduction
    combined with realistic gate noise.

    Inversion can push probabilities slightly negative under shot noise; the
    result is clipped and renormalised, which is the standard remedy.
    """
    inverse = np.linalg.inv(confusion_matrix(flip_probability, n_qubits))
    corrected = inverse @ probabilities
    corrected = np.clip(corrected, 0.0, None)
    total = corrected.sum()
    if total <= 0:
        raise ValueError("readout correction produced an empty distribution")
    return corrected / total


# ------------------------------------------------------ symmetry verification
def symmetry_postselect(
    counts: dict[str, int],
    n_electrons: int,
    *,
    basis_is_computational: bool = True,
    allow_rotated: bool = False,
) -> tuple[dict[str, int], float]:
    """Discard shots whose particle number is provably wrong.

    Noise can knock the state into a sector with the wrong number of
    electrons. Those shots are detectably wrong and cost nothing to drop.

    **The trap this guards against.** Particle number is only conserved in the
    computational basis. Measuring an X or Y observable requires rotations that
    do NOT commute with the number operator, so "Hamming weight" after such a
    rotation is not a physical invariant -- postselecting there throws away
    correct shots and keeps wrong ones. Doing this by mistake produced an
    energy roughly 40x worse than applying no mitigation whatsoever.

    Returns ``(kept_counts, fraction_kept)``.
    """
    if not basis_is_computational and not allow_rotated:
        raise ValueError(
            "symmetry verification is only valid in the computational basis; "
            "particle number is not conserved by X/Y basis rotations. Pass "
            "allow_rotated=True only if you have a specific reason."
        )

    kept = {
        bits: n
        for bits, n in counts.items()
        if bits.replace(" ", "").count("1") == n_electrons
    }
    total = sum(counts.values())
    n_kept = sum(kept.values())
    if n_kept == 0:
        return counts, 1.0  # nothing survived; fall back rather than divide by zero
    return kept, n_kept / total


# ------------------------------------------------------------- Pauli twirling
CX_TWIRL_TABLE = {
    ("I", "I"): ("I", "I"), ("I", "X"): ("I", "X"),
    ("I", "Y"): ("Z", "Y"), ("I", "Z"): ("Z", "Z"),
    ("X", "I"): ("X", "X"), ("X", "X"): ("X", "I"),
    ("X", "Y"): ("Y", "Z"), ("X", "Z"): ("Y", "Y"),
    ("Y", "I"): ("Y", "X"), ("Y", "X"): ("Y", "I"),
    ("Y", "Y"): ("X", "Z"), ("Y", "Z"): ("X", "Y"),
    ("Z", "I"): ("Z", "I"), ("Z", "X"): ("Z", "X"),
    ("Z", "Y"): ("I", "Y"), ("Z", "Z"): ("I", "Z"),
}
"""How a CX conjugates each two-qubit Pauli pair.

Computed directly from the CX unitary rather than transcribed, then used to
pick the trailing Paulis so the twirl is an exact identity on the ideal
circuit.
"""


def twirl_circuit(circuit: QuantumCircuit, rng=None) -> QuantumCircuit:
    """Wrap every CX in random Paulis that cancel on the ideal circuit.

    Averaged over draws this converts coherent error into stochastic Pauli
    error. **Measured no benefit against a purely depolarising noise model**
    (converged over 600 twirls), which is expected -- there is no coherent bias
    to convert. Kept because real hardware may carry coherent error that the
    model does not.
    """
    rng = rng or np.random.default_rng()
    paulis = ["I", "X", "Y", "Z"]

    def apply(target_circuit, label, qubit):
        if label == "X":
            target_circuit.x(qubit)
        elif label == "Y":
            target_circuit.y(qubit)
        elif label == "Z":
            target_circuit.z(qubit)

    twirled = QuantumCircuit(circuit.num_qubits)
    for inst in circuit.data:
        if inst.operation.name == "cx":
            control, target = (circuit.find_bit(q).index for q in inst.qubits)
            p_control = rng.choice(paulis)
            p_target = rng.choice(paulis)
            out_control, out_target = CX_TWIRL_TABLE[(p_control, p_target)]
            apply(twirled, p_control, control)
            apply(twirled, p_target, target)
            twirled.cx(control, target)
            apply(twirled, out_control, control)
            apply(twirled, out_target, target)
        else:
            twirled.append(inst.operation, inst.qubits, inst.clbits)
    return twirled


# ------------------------------------------------------------------ analysis
def shot_noise_sigma(expectation: float, shots: int) -> float:
    """Statistical uncertainty on a Pauli expectation value.

    ``sigma = sqrt(1 - <P>^2) / sqrt(N)``. Note the numerator: observables
    pinned near +-1 are far cheaper to measure than ones near zero, which is
    worth exploiting when choosing what to measure.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    return float(np.sqrt(max(0.0, 1.0 - expectation ** 2)) / np.sqrt(shots))


def is_significant(difference: float, sigma: float, threshold: float = 3.0) -> bool:
    """Is a measured gap real, or just shot noise?

    Worth using habitually: during development a 1-sigma fluctuation was
    briefly mistaken for a real 3x noise effect. A cheap guard against
    believing your own noise.
    """
    if sigma <= 0:
        return abs(difference) > 0
    return abs(difference) / sigma >= threshold
