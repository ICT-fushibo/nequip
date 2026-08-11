#!/usr/bin/env python3
"""Generate every Cu and H2O structure used by the NequIP MD benchmark.

CuN contains N Cu atoms. H2ON contains N water molecules (3N atoms).
All generated structures are periodic, and the output manifest contains paths
relative to the repository root so it is portable between machines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.io import write


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmark_structures"

# ase.build.bulk("Cu", "fcc") is a one-atom primitive cell.
CU_CONFIGS = {
    16: (2, 2, 4),
    32: (2, 4, 4),
    64: (4, 4, 4),
    192: (4, 6, 8),
    512: (8, 8, 8),
    1024: (8, 8, 16),
}

# Cubic water boxes at approximately 1 g/cm^3 (0.0334 molecule/A^3).
H2O_COUNTS = (16, 32, 60, 64, 192, 512, 1024)
H2O_BOX_LENGTHS = {
    count: (count / 0.0334) ** (1.0 / 3.0) for count in H2O_COUNTS
}

DEFAULT_SEED = 42
DEFAULT_MINIMUM_DISTANCE = 1.50
DEFAULT_JITTER_FRACTION = 0.04
DEFAULT_PLACEMENT_ATTEMPTS = 512


def relative_to_repo(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Generated paths must remain inside the repository: {path}"
        ) from error


def build_cu_fcc(n_atoms: int, output_path: Path) -> None:
    atoms = bulk("Cu", "fcc", a=3.61).repeat(CU_CONFIGS[n_atoms])
    if len(atoms) != n_atoms:
        raise RuntimeError(f"Expected {n_atoms} Cu atoms, generated {len(atoms)}")
    write(output_path, atoms, format="cif")
    print(f"Cu{n_atoms}: {len(atoms)} atoms -> {relative_to_repo(output_path)}")


def minimum_image_delta(delta: np.ndarray, box: float) -> np.ndarray:
    return delta - box * np.rint(delta / box)


def periodic_squared_distances(
    points: np.ndarray, reference: np.ndarray, box: float
) -> np.ndarray:
    delta = minimum_image_delta(points - reference, box)
    return np.einsum("ij,ij->i", delta, delta)


def uniform_grid_centers(
    n_molecules: int, box: float
) -> tuple[np.ndarray, float]:
    """Choose well-distributed sites from a periodic cubic grid."""

    n_per_side = int(np.ceil(n_molecules ** (1.0 / 3.0)))
    spacing = box / n_per_side
    axis = (np.arange(n_per_side, dtype=float) + 0.5) * spacing
    candidates = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    if len(candidates) == n_molecules:
        return candidates, spacing

    selected = np.empty(n_molecules, dtype=np.int64)
    selected[0] = 0
    minimum_distance_sq = periodic_squared_distances(
        candidates, candidates[0], box
    )
    minimum_distance_sq[0] = -np.inf
    for index in range(1, n_molecules):
        candidate_index = int(np.argmax(minimum_distance_sq))
        selected[index] = candidate_index
        distance_sq = periodic_squared_distances(
            candidates, candidates[candidate_index], box
        )
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
        minimum_distance_sq[selected[: index + 1]] = -np.inf
    return candidates[selected], spacing


def random_rotation_matrix(rng: np.random.RandomState) -> np.ndarray:
    """Draw a uniformly distributed three-dimensional rotation."""

    u1, u2, u3 = rng.random_sample(3)
    x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def candidate_is_collision_free(
    candidate: np.ndarray,
    placed_positions: np.ndarray,
    box: float,
    minimum_distance: float,
) -> bool:
    if placed_positions.size == 0:
        return True
    delta = candidate[:, None, :] - placed_positions[None, :, :]
    delta = minimum_image_delta(delta, box)
    distance_sq = np.einsum("ijk,ijk->ij", delta, delta)
    return bool(np.all(distance_sq >= minimum_distance**2))


def minimum_intermolecular_distance(
    positions: np.ndarray,
    box: float,
    atoms_per_molecule: int = 3,
    block_size: int = 256,
) -> tuple[float, tuple[int, int]]:
    n_atoms = len(positions)
    molecule_index = np.arange(n_atoms) // atoms_per_molecule
    best_distance_sq = np.inf
    best_pair = (-1, -1)
    for start in range(0, n_atoms, block_size):
        stop = min(start + block_size, n_atoms)
        delta = positions[start:stop, None, :] - positions[None, :, :]
        delta = minimum_image_delta(delta, box)
        distance_sq = np.einsum("ijk,ijk->ij", delta, delta)
        same_molecule = molecule_index[start:stop, None] == molecule_index[None, :]
        distance_sq[same_molecule] = np.inf
        flat_index = int(np.argmin(distance_sq))
        local_index, other_index = np.unravel_index(flat_index, distance_sq.shape)
        value = float(distance_sq[local_index, other_index])
        if value < best_distance_sq:
            best_distance_sq = value
            best_pair = (start + local_index, other_index)
    return float(np.sqrt(best_distance_sq)), best_pair


def build_h2o_box(
    n_molecules: int,
    output_path: Path,
    *,
    minimum_distance: float,
    seed: int,
    jitter_fraction: float,
    placement_attempts: int,
) -> None:
    box = H2O_BOX_LENGTHS[n_molecules]
    water = molecule("H2O")
    centers, spacing = uniform_grid_centers(n_molecules, box)
    rng = np.random.RandomState(seed + n_molecules)
    template_positions = water.get_positions() - water.get_center_of_mass()
    placed_positions = np.empty((3 * n_molecules, 3), dtype=float)
    placed_atoms = 0
    jitter = jitter_fraction * spacing

    for molecule_index, center in enumerate(centers):
        accepted = None
        for attempt in range(placement_attempts):
            offset = (
                np.zeros(3)
                if attempt == 0 or jitter == 0
                else rng.uniform(-jitter, jitter, size=3)
            )
            candidate = (
                template_positions @ random_rotation_matrix(rng).T + center + offset
            ) % box
            if candidate_is_collision_free(
                candidate,
                placed_positions[:placed_atoms],
                box,
                minimum_distance,
            ):
                accepted = candidate
                break
        if accepted is None:
            raise RuntimeError(
                f"Could not place water {molecule_index + 1}/{n_molecules}; "
                "reduce the minimum distance or jitter."
            )
        placed_positions[placed_atoms : placed_atoms + 3] = accepted
        placed_atoms += 3

    atoms = Atoms(
        symbols=water.get_chemical_symbols() * n_molecules,
        positions=placed_positions,
        cell=[box, box, box],
        pbc=True,
    )
    observed_minimum, pair = minimum_intermolecular_distance(
        atoms.get_positions(), box
    )
    if observed_minimum + 1.0e-10 < minimum_distance:
        raise RuntimeError(
            f"H2O{n_molecules}: minimum intermolecular distance "
            f"{observed_minimum:.6f} A for pair {pair}, expected "
            f">= {minimum_distance:.6f} A"
        )
    write(output_path, atoms, format="cif")
    print(
        f"H2O{n_molecules}: {n_molecules} molecules, {len(atoms)} atoms, "
        f"min_inter={observed_minimum:.3f} A -> {relative_to_repo(output_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--water-min-distance", type=float, default=DEFAULT_MINIMUM_DISTANCE
    )
    parser.add_argument(
        "--water-jitter-fraction", type=float, default=DEFAULT_JITTER_FRACTION
    )
    parser.add_argument(
        "--water-placement-attempts",
        type=int,
        default=DEFAULT_PLACEMENT_ATTEMPTS,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    relative_to_repo(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = (args.manifest or output_dir / "systems.tsv").resolve()
    relative_to_repo(manifest)

    records: list[tuple[str, Path]] = []
    for count in CU_CONFIGS:
        output_path = output_dir / f"Cu{count}.cif"
        build_cu_fcc(count, output_path)
        records.append((f"Cu{count}", output_path))
    for count in H2O_COUNTS:
        output_path = output_dir / f"H2O{count}.cif"
        build_h2o_box(
            count,
            output_path,
            minimum_distance=args.water_min_distance,
            seed=args.seed,
            jitter_fraction=args.water_jitter_fraction,
            placement_attempts=args.water_placement_attempts,
        )
        records.append((f"H2O{count}", output_path))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "# label\tstructure path relative to repository root\n"
        + "".join(
            f"{label}\t{relative_to_repo(path)}\n" for label, path in records
        ),
        encoding="utf-8",
    )
    print(f"Manifest: {relative_to_repo(manifest)}")


if __name__ == "__main__":
    main()
