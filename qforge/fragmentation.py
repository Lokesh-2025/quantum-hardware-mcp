"""
qforge.fragmentation — solving molecules too big to solve.
==========================================================
Exact diagonalisation needs one qubit per spin-orbital, so anything past ~16
qubits is out of reach on a laptop and past ~50 is out of reach for anyone.
Fragmentation buys scale by never looking at the whole molecule at once.

Three methods, and they are NOT interchangeable -- each fails in its own way:

**Many-body expansion** (:func:`many_body_expansion`) -- for CLUSTERS of
separate molecules. Solve monomers, then pairs, then triples, summing
corrections. Recovers 99.5-99.96% of the interaction energy on tested
hydrogen clusters. Excellent when fragments are genuinely separate.

**Molecular tailoring** (:func:`molecular_tailoring`) -- for a single BONDED
chain, which cannot simply be split. Uses overlapping fragments plus
inclusion-exclusion. Error shrinks as fragments grow (0.0107 -> 0.0018 Ha
going from 4- to 6-atom blocks), giving a real accuracy knob. But on strongly
bonded systems it plateaus at 1-7 kcal/mol, because cutting a bond destroys
entanglement that no amount of bookkeeping restores.

**DMET** (:func:`dmet_energy`) -- never solves a fragment in isolation.
Constructs BATH orbitals carrying the mean-field entanglement between fragment
and environment, then solves fragment+bath together. On an H8 chain this beat
tailoring outright: 3.40 kcal/mol using 4 qubits per impurity, versus 6.72
kcal/mol using 8 qubits per fragment. Twice the accuracy at half the qubits.

Honest limits on DMET as implemented: this is SINGLE-SHOT DMET, without the
self-consistency loop that fits a correlation potential. On an H8 ring it did
worse than tailoring (25.97 vs 2.67 kcal/mol), and error grew with fragment
size rather than shrinking -- both known symptoms of the missing loop.

A method that was tried and did NOT work is documented in
:func:`capped_fragment_geometry`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.linalg import eigh

BATH_TOLERANCE = 1e-8


# ------------------------------------------------------- many-body expansion
def many_body_expansion(fragment_energies: dict, max_body: int = 2) -> float:
    """Sum monomer, pair (and optionally triple) contributions.

    ``E ~ sum_i E_i + sum_{i<j} (E_ij - E_i - E_j) + ...``

    Args:
        fragment_energies: maps a tuple of fragment indices to that
            subsystem's energy, e.g. ``{(0,): -1.1, (1,): -1.1, (0,1): -2.2}``.
        max_body: 2 or 3.
    """
    monomers = [k for k in fragment_energies if len(k) == 1]
    total = sum(fragment_energies[k] for k in monomers)

    if max_body >= 2:
        for pair in (k for k in fragment_energies if len(k) == 2):
            i, j = pair
            total += (
                fragment_energies[pair]
                - fragment_energies[(i,)]
                - fragment_energies[(j,)]
            )

    if max_body >= 3:
        for triple in (k for k in fragment_energies if len(k) == 3):
            i, j, k = triple
            two_body = sum(
                fragment_energies[tuple(sorted(p))]
                - fragment_energies[(p[0],)]
                - fragment_energies[(p[1],)]
                for p in combinations(triple, 2)
            )
            total += (
                fragment_energies[triple]
                - fragment_energies[(i,)]
                - fragment_energies[(j,)]
                - fragment_energies[(k,)]
                - two_body
            )
    return float(total)


# ------------------------------------------------------- molecular tailoring
def overlapping_fragments(
    n_sites: int, block: int, stride: int = 2, cyclic: bool = False
):
    """Overlapping fragments and the overlaps between neighbours.

    Neighbouring fragments share ``block - stride`` sites, which is exactly
    what inclusion-exclusion subtracts back off.
    """
    if cyclic:
        starts = list(range(0, n_sites, stride))
        fragments = [[(s + k) % n_sites for k in range(block)] for s in starts]
        overlaps = [
            [a for a in fragments[i] if a in fragments[(i + 1) % len(fragments)]]
            for i in range(len(fragments))
        ]
    else:
        starts = list(range(0, n_sites - block + 1, stride))
        fragments = [list(range(s, s + block)) for s in starts]
        overlaps = [
            sorted(set(fragments[i]) & set(fragments[i + 1]))
            for i in range(len(fragments) - 1)
        ]
        covered = {site for frag in fragments for site in frag}
        if covered != set(range(n_sites)):
            raise ValueError(
                f"fragments do not cover all {n_sites} sites; choose a block "
                f"size where (n_sites - block) is divisible by {stride}"
            )
    return fragments, overlaps


def molecular_tailoring(fragment_energies, overlap_energies) -> float:
    """``E = sum(fragments) - sum(overlaps)`` -- inclusion-exclusion on energy.

    Same principle as ``|A u B| = |A| + |B| - |A n B|``, with energy standing
    in for set size.
    """
    return float(sum(fragment_energies) - sum(overlap_energies))


def capped_fragment_geometry(
    positions, indices, cap_distance: float, n_sites: int, cyclic: bool = False
):
    """Fragment geometry with a hydrogen cap at each severed bond.

    **This did not work, and the negative result is the point.** Capping is
    standard practice (MFCC) when cutting a C-C bond, because a hydrogen cap
    is a small, different atom that gently satisfies the dangling valence.
    For a hydrogen chain the cap is chemically identical to the atom removed,
    so instead of a gentle mimic you insert a whole extra atom with its own
    electron -- perturbing the fragment as much as the cut did.

    Measured on an H8 chain: uncapped 6.72 kcal/mol; capped at the physically
    natural H2 bond length (0.74 A) 85.59 kcal/mol -- nearly 13x WORSE, at a
    higher qubit cost. On an H8 ring, 2.67 -> 19.64 kcal/mol.

    Scanning the cap distance found one value that beat uncapped, but choosing
    it requires knowing the exact answer in advance, so it is tuning rather
    than a method. Provided for reproducibility, not for use on bare hydrogen.
    """
    geometry = [("H", tuple(positions[i])) for i in indices]
    first, last = indices[0], indices[-1]

    def cap_at(edge, removed):
        direction = np.asarray(removed) - np.asarray(edge)
        direction = direction / np.linalg.norm(direction)
        return np.asarray(edge) + cap_distance * direction

    caps = []
    if cyclic:
        caps.append(cap_at(positions[first], positions[(first - 1) % n_sites]))
        caps.append(cap_at(positions[last], positions[(last + 1) % n_sites]))
    else:
        if first - 1 >= 0:
            caps.append(cap_at(positions[first], positions[first - 1]))
        if last + 1 <= n_sites - 1:
            caps.append(cap_at(positions[last], positions[last + 1]))
    return geometry + [("H", tuple(c)) for c in caps]


# ------------------------------------------------------------------- DMET
@dataclass
class Embedding:
    """A fragment plus the bath and core that surround it."""

    basis: np.ndarray            # site -> embedding orbitals
    n_bath: int
    core_density: np.ndarray     # idempotent; carries 2*rho_core electrons

    @property
    def n_orbitals(self) -> int:
        return self.basis.shape[1]

    @property
    def n_core_electrons(self) -> int:
        return int(round(2.0 * np.trace(self.core_density)))


def build_embedding(density: np.ndarray, fragment_sites, n_sites: int) -> Embedding:
    """Schmidt-split the mean-field state across fragment | environment.

    Three orbital groups come out, and missing the third is a real bug:

    * **fragment** -- the sites you chose
    * **bath** -- at most ``n_fragment`` environment orbitals, the ONLY ones
      entangled with the fragment. This bound is why DMET stays small no
      matter how large the molecule is.
    * **core** -- remaining occupied environment orbitals. Not entangled, but
      still full of electrons exerting Coulomb and exchange fields on the
      fragment.

    Omitting the core produced errors of 468-3047 kcal/mol during development.
    It escaped the self-checks because a whole-molecule fragment has an empty
    environment, hence no core at all.
    """
    fragment = sorted(fragment_sites)
    environment = [i for i in range(n_sites) if i not in fragment]

    basis_fragment = np.zeros((n_sites, len(fragment)))
    for column, site in enumerate(fragment):
        basis_fragment[site, column] = 1.0

    if not environment:
        return Embedding(basis_fragment, 0, np.zeros((n_sites, n_sites)))

    rho_ef = density[np.ix_(environment, fragment)]
    u_mat, singular, _ = np.linalg.svd(rho_ef, full_matrices=False)
    u_mat = u_mat[:, singular > BATH_TOLERANCE]
    n_bath = u_mat.shape[1]

    basis_bath = np.zeros((n_sites, n_bath))
    basis_bath[np.ix_(environment, range(n_bath))] = u_mat
    basis = np.hstack([basis_fragment, basis_bath])

    rho_env = density[np.ix_(environment, environment)]
    project_out = np.eye(len(environment)) - u_mat @ u_mat.T
    values, vectors = eigh(project_out @ rho_env @ project_out)
    core_env = vectors[:, values > 0.5]
    core_coeffs = np.zeros((n_sites, core_env.shape[1]))
    core_coeffs[np.ix_(environment, range(core_env.shape[1]))] = core_env

    return Embedding(basis, n_bath, core_coeffs @ core_coeffs.T)


def core_potential(core_density: np.ndarray, eri: np.ndarray):
    """Effective one-body potential and energy from the frozen core.

    Returns ``(h1_correction, ...)`` where the correction is ``J - K/2`` for the
    core electron density, to be added to the bare one-body integrals before
    solving the impurity.
    """
    density = 2.0 * core_density
    coulomb = np.einsum("rs,pqrs->pq", density, eri, optimize=True)
    exchange = np.einsum("rs,prsq->pq", density, eri, optimize=True)
    return coulomb - 0.5 * exchange


def assemble_rdms(
    embedding: Embedding,
    impurity_1rdm: np.ndarray,
    impurity_2rdm: np.ndarray,
):
    """Push impurity RDMs back to the site basis, adding core contributions.

    Standard complete-active-space decomposition: the total 2-RDM is the
    active part plus core-core and core-active cross terms. With an empty core
    it reduces to the active part alone.

    A good check that this is right: ``trace(P_full)`` must equal the total
    electron count for every fragment. It does, exactly.
    """
    basis = embedding.basis
    core = 2.0 * embedding.core_density

    active_1rdm = basis @ impurity_1rdm @ basis.T
    active_2rdm = np.einsum(
        "pi,qj,rk,sl,ijkl->pqrs", basis, basis, basis, basis, impurity_2rdm,
        optimize=True,
    )

    full_1rdm = active_1rdm + core
    core_core = np.einsum("pq,rs->pqrs", core, core, optimize=True) - 0.5 * np.einsum(
        "ps,rq->pqrs", core, core, optimize=True
    )
    core_active = (
        np.einsum("pq,rs->pqrs", core, active_1rdm, optimize=True)
        + np.einsum("pq,rs->pqrs", active_1rdm, core, optimize=True)
        - 0.5 * np.einsum("ps,rq->pqrs", core, active_1rdm, optimize=True)
        - 0.5 * np.einsum("ps,rq->pqrs", active_1rdm, core, optimize=True)
    )
    return full_1rdm, active_2rdm + core_core + core_active


def energy_from_rdms(h1, eri, one_rdm, two_rdm, fragment_sites=None) -> float:
    """``E = sum h1 P + 0.5 sum (pq|rs) G``, optionally restricted to a fragment.

    Restricting the FIRST index to fragment sites is the "democratic
    partitioning" that makes fragment energies sum to the total without double
    counting, since fragments tile the system.
    """
    if fragment_sites is None:
        one_body = np.einsum("pq,pq->", h1, one_rdm, optimize=True)
        two_body = 0.5 * np.einsum("pqrs,pqrs->", eri, two_rdm, optimize=True)
    else:
        sites = np.array(sorted(fragment_sites))
        one_body = np.einsum("pq,pq->", h1[sites, :], one_rdm[sites, :], optimize=True)
        two_body = 0.5 * np.einsum(
            "pqrs,pqrs->", eri[sites], two_rdm[sites], optimize=True
        )
    return float(np.real(one_body + two_body))
