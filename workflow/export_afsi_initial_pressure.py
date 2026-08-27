#!/usr/bin/env python3
"""Generate DESC-backed AFSI birth distributions and pressure moments.

AFSI produces a fusion-product source distribution whose weights have units of
particles/s. Consequently, its direct density and pressure moments are source
rates (m^-3 s^-1 and Pa/s). If ``--accumulation-time-s`` is supplied, this
script also converts the rates to a finite-population estimate by multiplying
them by that duration.

The script never reads pressure from ``analysis_results.h5``. When that file is
available, only its rho/theta/phi coordinates are read so the radial AFSI
moments can be broadcast onto the existing analysis grid.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGGER = logging.getLogger("export_afsi_initial_pressure")
TAG_READY = 1
TAG_WORK = 2
TAG_STOP = 3
TAG_RESULT = 4
ALPHA_MASS_AMU = 4.001506179127
ATOMIC_MASS_KG = 1.66053906892e-27
SPEED_OF_LIGHT_M_S = 299792458.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/scratch/gpfs/rmc2/m5300/results"),
        help="Tree containing sample directories (default: %(default)s).",
    )
    parser.add_argument(
        "--equilibrium-filename",
        default="desc_equilibrium.h5",
        help="DESC equilibrium filename to discover recursively.",
    )
    parser.add_argument(
        "--analysis-filename",
        default="analysis_results.h5",
        help=(
            "Optional analysis file whose coordinate grid is copied. Its pressure "
            "datasets are never read."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default="afsi_initial.h5",
        help="Output filename written beside each equilibrium.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replace-incompatible",
        action="store_true",
        help=(
            "Replace an existing output when its stored AFSI/grid configuration "
            "does not match the current request; otherwise skip it."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--mpi",
        action="store_true",
        help=(
            "Run with an MPI master and dynamic worker queue. Launch using "
            "srun/mpiexec; rank 0 dispatches equilibria to free workers."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Zero-based shard index; also inferred from SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Number of shards; also inferred from SLURM_ARRAY_TASK_COUNT.",
    )

    afsi = parser.add_argument_group("AFSI resolution")
    afsi.add_argument("--nrho-bins", type=int, default=102)
    afsi.add_argument("--nenergy-bins", type=int, default=50)
    afsi.add_argument("--npitch-bins", type=int, default=51)
    afsi.add_argument("--nmc", type=int, default=1000)
    afsi.add_argument("--nthermal-vel", type=int, default=10)
    afsi.add_argument(
        "--accumulation-time-s",
        type=float,
        default=None,
        help=(
            "Optional source accumulation duration. When supplied, pressure in Pa "
            "and density in m^-3 are written in addition to the native rates."
        ),
    )

    ascot = parser.add_argument_group("DESC-to-ASCOT input resolution")
    ascot.add_argument("--field-nr", type=int, default=200)
    ascot.add_argument("--field-nz", type=int, default=200)
    ascot.add_argument("--field-nphi", type=int, default=200)
    ascot.add_argument("--profile-nrho", type=int, default=1024)
    ascot.add_argument("--fraction-tritium", type=float, default=0.5)
    ascot.add_argument("--zeff", type=float, default=1.0)
    ascot.add_argument("--l-radial", type=int, default=10)
    ascot.add_argument("--m-poloidal", type=int, default=10)
    ascot.add_argument(
        "--no-stellarator-symmetry",
        action="store_true",
        help="Disable stellarator symmetry when building the ASCOT field.",
    )
    ascot.add_argument(
        "--show-field-progress",
        action="store_true",
        help="Show the DESC magnetic-field interpolation progress bar.",
    )
    ascot.add_argument(
        "--keep-ascot-input",
        action="store_true",
        help="Keep the generated ASCOT input HDF5 file beside the output.",
    )
    return parser


def configure_logging(verbose: bool, rank: int | None = None) -> None:
    rank_label = "" if rank is None else f" [rank {rank}]"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s{rank_label} %(message)s",
        force=True,
    )
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)


def _resolve_sharding(args: argparse.Namespace) -> tuple[int | None, int | None]:
    shard_index = args.shard_index
    num_shards = args.num_shards
    if shard_index is None and num_shards is None:
        env_index = os.environ.get("SLURM_ARRAY_TASK_ID")
        env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
        if env_index is not None and env_count is not None:
            shard_index = int(env_index)
            num_shards = int(env_count)

    if (shard_index is None) != (num_shards is None):
        raise ValueError("Provide both --shard-index and --num-shards, or neither.")
    if num_shards is not None:
        if num_shards <= 0:
            raise ValueError("--num-shards must be positive.")
        if shard_index is None or not 0 <= shard_index < num_shards:
            raise ValueError("Require 0 <= shard-index < num-shards.")
    return shard_index, num_shards


def discover_equilibria(args: argparse.Namespace) -> list[Path]:
    root = args.results_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {root}")

    equilibria = sorted(root.rglob(args.equilibrium_filename))
    equilibria = [
        path
        for path in equilibria
        if path.is_file() and path.parent.name != "initial_pressure"
    ]
    shard_index, num_shards = _resolve_sharding(args)
    if shard_index is not None and num_shards is not None:
        equilibria = [
            path
            for index, path in enumerate(equilibria)
            if index % num_shards == shard_index
        ]
    if args.limit is not None:
        equilibria = equilibria[: max(args.limit, 0)]
    return equilibria


def _unit_values(value: Any, unit: str | None = None) -> tuple[np.ndarray, str]:
    if unit is not None and hasattr(value, "to"):
        value = value.to(unit)
    units = str(getattr(value, "units", "dimensionless"))
    raw = getattr(value, "value", value)
    return np.asarray(raw), units


def _load_equilibrium(path: Path):
    import desc.io

    family = desc.io.load(path)
    try:
        return family[-1]
    except (IndexError, KeyError, TypeError):
        return family


def _shell_volumes(equilibrium_path: Path, rho_edges: np.ndarray) -> np.ndarray:
    import desc.grid

    equilibrium = _load_equilibrium(equilibrium_path)
    grid = desc.grid.LinearGrid(
        rho=np.asarray(rho_edges),
        M=equilibrium.M_grid,
        N=equilibrium.N_grid,
        NFP=equilibrium.NFP,
        sym=False,
    )
    data = equilibrium.compute("V(r)", grid=grid)
    enclosed = np.asarray(grid.compress(data["V(r)"]), dtype=np.float64)
    volumes = np.diff(enclosed)
    if volumes.shape != (rho_edges.size - 1,):
        raise ValueError(
            f"Unexpected DESC shell-volume shape {volumes.shape}; "
            f"expected {(rho_edges.size - 1,)}."
        )
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("DESC returned non-finite or non-positive shell volumes.")
    return volumes


def calculate_source_moments(dist: Any, equilibrium_path: Path) -> dict[str, Any]:
    """Calculate native spatial moments of an AFSI (E, pitch) distribution."""
    axes = list(dist.abscissae)
    required = {"rho", "theta", "phi", "ekin", "xi"}
    missing = sorted(required.difference(axes))
    if missing:
        raise ValueError(f"AFSI distribution is missing axes: {missing}")

    histogram_raw, _ = _unit_values(dist.histogram())
    histogram = np.asarray(histogram_raw, dtype=np.float64)
    retained = ["rho", "theta", "phi", "ekin", "xi"]
    current_axes = list(axes)
    for axis in reversed(range(len(current_axes))):
        if current_axes[axis] not in retained:
            histogram = histogram.sum(axis=axis)
            current_axes.pop(axis)
    histogram = np.transpose(
        histogram, [current_axes.index(name) for name in retained]
    )

    energy, _ = _unit_values(dist.abscissa("ekin"), "J")
    pitch, _ = _unit_values(dist.abscissa("xi"), "dimensionless")
    energy = np.asarray(energy, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    mass = ALPHA_MASS_AMU * ATOMIC_MASS_KG

    gamma = 1.0 + energy / (mass * SPEED_OF_LIGHT_M_S**2)
    speed = SPEED_OF_LIGHT_M_S * np.sqrt(1.0 - gamma**-2)
    vpara = speed[:, None] * pitch[None, :]
    vperp = speed[:, None] * np.sqrt(np.maximum(0.0, 1.0 - pitch[None, :] ** 2))

    number_rate = histogram.sum(axis=(-2, -1))
    nvpara = (histogram * vpara).sum(axis=(-2, -1))
    nvpara2 = (histogram * vpara**2).sum(axis=(-2, -1))
    nvperp2 = (histogram * vperp**2).sum(axis=(-2, -1))
    mean_vpara = np.divide(
        nvpara,
        number_rate,
        out=np.zeros_like(nvpara),
        where=number_rate > 0,
    )

    rho_edges, _ = _unit_values(dist.abscissa_edges("rho"), "dimensionless")
    theta_edges, _ = _unit_values(dist.abscissa_edges("theta"), "rad")
    phi_edges, _ = _unit_values(dist.abscissa_edges("phi"), "rad")
    shell_volume = _shell_volumes(equilibrium_path, rho_edges)
    theta_fraction = np.diff(theta_edges) / (2.0 * np.pi)
    phi_fraction = np.diff(phi_edges) / (2.0 * np.pi)
    volume = (
        shell_volume[:, None, None]
        * theta_fraction[None, :, None]
        * phi_fraction[None, None, :]
    )
    if volume.shape != number_rate.shape:
        raise ValueError(
            f"Physical-volume shape {volume.shape} does not match AFSI spatial "
            f"shape {number_rate.shape}."
        )

    density_rate = number_rate / volume
    prs_para_rate = mass * (nvpara2 - number_rate * mean_vpara**2) / volume
    prs_perp_rate = 0.5 * mass * nvperp2 / volume
    # Roundoff can make a variance very slightly negative.
    prs_para_rate = np.maximum(prs_para_rate, 0.0)
    pressure_rate = (prs_para_rate + 2.0 * prs_perp_rate) / 3.0

    return {
        "axes": ("rho", "theta", "phi"),
        "rho": np.asarray(dist.abscissa("rho"), dtype=np.float64),
        "theta": np.asarray(dist.abscissa("theta").to("rad").value, dtype=np.float64),
        "phi": np.asarray(dist.abscissa("phi").to("rad").value, dtype=np.float64),
        "volume": volume,
        "number_rate": number_rate,
        "density_rate": density_rate,
        "mean_vpara": mean_vpara,
        "prs_para_rate": prs_para_rate,
        "prs_perp_rate": prs_perp_rate,
        "pressure_rate": pressure_rate,
    }


def _radial_average(values: np.ndarray, volume: np.ndarray) -> np.ndarray:
    numerator = np.sum(values * volume, axis=(1, 2))
    denominator = np.sum(volume, axis=(1, 2))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def _read_analysis_grid(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        names = ("profiles/rho", "profiles/theta", "profiles/phi")
        missing = [name for name in names if name not in h5f]
        if missing:
            raise KeyError(f"Missing coordinate datasets in {path}: {missing}")
        return tuple(np.asarray(h5f[name][...]) for name in names)  # type: ignore[return-value]


def _profile_on_analysis_grid(
    radial_coordinate: np.ndarray,
    native_values: np.ndarray,
    native_volume: np.ndarray,
    target_rho: np.ndarray,
    target_theta: np.ndarray,
    target_phi: np.ndarray,
) -> np.ndarray:
    radial_values = _radial_average(native_values, native_volume)
    interpolated = np.interp(
        target_rho,
        radial_coordinate,
        radial_values,
        left=radial_values[0],
        right=0.0,
    )
    return np.broadcast_to(
        interpolated[:, None, None],
        (target_rho.size, target_theta.size, target_phi.size),
    ).copy()


def _write_dataset(
    group: h5py.Group,
    name: str,
    data: Any,
    *,
    units: str,
    description: str,
    dimensions: tuple[str, ...] | None = None,
) -> h5py.Dataset:
    array = np.asarray(data)
    kwargs: dict[str, Any] = {}
    if array.ndim > 0 and array.size > 1:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    dataset = group.create_dataset(name, data=array, **kwargs)
    dataset.attrs["units"] = units
    dataset.attrs["description"] = description
    if dimensions is not None:
        dataset.attrs["dimensions"] = json.dumps(dimensions)
    return dataset


def write_output(
    output_path: Path,
    *,
    equilibrium_path: Path,
    analysis_path: Path | None,
    ascot_input_path: Path | None,
    dist: Any,
    moments: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        with h5py.File(tmp_path, "w") as h5f:
            h5f.attrs["description"] = "DESC-backed AFSI alpha birth distribution"
            h5f.attrs["desc_equilibrium_path"] = str(equilibrium_path)
            h5f.attrs["analysis_grid_path"] = "" if analysis_path is None else str(analysis_path)
            h5f.attrs["ascot_input_path"] = "" if ascot_input_path is None else str(ascot_input_path)
            h5f.attrs["reaction"] = "DT_He4n"
            h5f.attrs["afsi_weight_units"] = "particles/s"
            h5f.attrs["configuration"] = json.dumps(vars(args), default=str, sort_keys=True)

            distribution = h5f.create_group("afsi_distribution")
            distribution.attrs["axis_order"] = json.dumps(list(dist.abscissae))
            values, units = _unit_values(dist.distribution())
            _write_dataset(
                distribution,
                "distribution_function",
                values,
                units=units,
                description="AFSI differential alpha birth source distribution",
                dimensions=tuple(dist.abscissae),
            )
            values, units = _unit_values(dist.histogram())
            _write_dataset(
                distribution,
                "histogram",
                values,
                units="particles/s",
                description="AFSI alpha birth rate in each phase-space bin",
                dimensions=tuple(dist.abscissae),
            )
            values, units = _unit_values(dist.phasespacevolume())
            _write_dataset(
                distribution,
                "phase_space_volume",
                values,
                units=units,
                description="Coordinate-space volume of each histogram bin",
                dimensions=tuple(dist.abscissae),
            )
            coordinates = distribution.create_group("coordinates")
            for axis in dist.abscissae:
                centers, center_units = _unit_values(dist.abscissa(axis))
                edges, edge_units = _unit_values(dist.abscissa_edges(axis))
                _write_dataset(
                    coordinates,
                    axis,
                    centers,
                    units=center_units,
                    description=f"Centers of the {axis} bins",
                )
                _write_dataset(
                    coordinates,
                    f"{axis}_edges",
                    edges,
                    units=edge_units,
                    description=f"Edges of the {axis} bins",
                )

            native = h5f.create_group("moments_native")
            native.attrs["dimensions"] = json.dumps(moments["axes"])
            native.attrs["weight_interpretation"] = (
                "AFSI weights are particles/s, so direct moments are source rates."
            )
            for coordinate in moments["axes"]:
                _write_dataset(
                    native,
                    coordinate,
                    moments[coordinate],
                    units="dimensionless" if coordinate == "rho" else "rad",
                    description=f"Native AFSI {coordinate} coordinate",
                )
            native_specs = {
                "volume": ("m**3", "Physical DESC volume of each spatial bin"),
                "number_rate": ("particles/s", "Alpha birth rate per spatial bin"),
                "density_rate": ("1/(m**3*s)", "Alpha density source rate"),
                "mean_vpara": ("m/s", "Mean parallel birth velocity"),
                "prs_para_rate": ("Pa/s", "Parallel pressure source rate"),
                "prs_perp_rate": ("Pa/s", "Perpendicular pressure source rate"),
                "pressure_rate": ("Pa/s", "Scalar pressure source rate"),
            }
            for name, (units, description) in native_specs.items():
                _write_dataset(
                    native,
                    name,
                    moments[name],
                    units=units,
                    description=description,
                    dimensions=moments["axes"],
                )

            if args.accumulation_time_s is not None:
                native.attrs["accumulation_time_s"] = args.accumulation_time_s
                for source_name, output_name, units, description in (
                    ("density_rate", "density", "1/m**3", "Accumulated alpha density"),
                    ("prs_para_rate", "prs_para", "Pa", "Accumulated parallel pressure"),
                    ("prs_perp_rate", "prs_perp", "Pa", "Accumulated perpendicular pressure"),
                    ("pressure_rate", "pressure", "Pa", "Accumulated scalar pressure"),
                ):
                    _write_dataset(
                        native,
                        output_name,
                        moments[source_name] * args.accumulation_time_s,
                        units=units,
                        description=description,
                        dimensions=moments["axes"],
                    )

            if analysis_path is not None:
                target_rho, target_theta, target_phi = _read_analysis_grid(analysis_path)
                profiles = h5f.create_group("profiles")
                profiles.attrs["coordinate_source"] = str(analysis_path)
                profiles.attrs["pressure_data_source"] = "AFSI; no pressure read from analysis file"
                for name, values, units in (
                    ("rho", target_rho, "dimensionless"),
                    ("theta", target_theta, "rad"),
                    ("phi", target_phi, "rad"),
                ):
                    _write_dataset(
                        profiles,
                        name,
                        values,
                        units=units,
                        description=f"Analysis-grid {name} coordinate",
                    )
                for source_name, description in (
                    ("density_rate", "Alpha density source rate on the analysis grid"),
                    ("prs_para_rate", "Parallel pressure source rate on the analysis grid"),
                    ("prs_perp_rate", "Perpendicular pressure source rate on the analysis grid"),
                    ("pressure_rate", "Scalar pressure source rate on the analysis grid"),
                ):
                    profile = _profile_on_analysis_grid(
                        moments["rho"],
                        moments[source_name],
                        moments["volume"],
                        target_rho,
                        target_theta,
                        target_phi,
                    )
                    units = "1/(m**3*s)" if source_name == "density_rate" else "Pa/s"
                    _write_dataset(
                        profiles,
                        source_name,
                        profile,
                        units=units,
                        description=description,
                        dimensions=("rho", "theta", "phi"),
                    )
                    if args.accumulation_time_s is not None:
                        output_name = source_name.removesuffix("_rate")
                        output_units = "1/m**3" if output_name == "density" else "Pa"
                        _write_dataset(
                            profiles,
                            output_name,
                            profile * args.accumulation_time_s,
                            units=output_units,
                            description=(
                                f"{description.removesuffix(' source rate')} accumulated "
                                f"for {args.accumulation_time_s:g} s"
                            ),
                            dimensions=("rho", "theta", "phi"),
                        )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


_OUTPUT_CONFIGURATION_KEYS = (
    "nrho_bins",
    "nenergy_bins",
    "npitch_bins",
    "nmc",
    "nthermal_vel",
    "field_nr",
    "field_nz",
    "field_nphi",
    "profile_nrho",
    "fraction_tritium",
    "zeff",
    "l_radial",
    "m_poloidal",
    "no_stellarator_symmetry",
    "accumulation_time_s",
)


def _output_configuration_matches(
    output_path: Path, args: argparse.Namespace
) -> bool:
    try:
        with h5py.File(output_path, "r") as h5f:
            stored = json.loads(str(h5f.attrs["configuration"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    requested = vars(args)
    return all(stored.get(key) == requested.get(key) for key in _OUTPUT_CONFIGURATION_KEYS)


def process_equilibrium(equilibrium_path: Path, args: argparse.Namespace) -> dict[str, str]:
    folder = equilibrium_path.parent
    output_path = folder / args.output_filename
    if output_path.exists() and not args.overwrite:
        compatible = _output_configuration_matches(output_path, args)
        if not args.replace_incompatible or compatible:
            message = "compatible output exists" if compatible else "output exists"
            return {"status": "skipped", "path": str(output_path), "message": message}

    analysis_candidate = folder / args.analysis_filename
    analysis_path = analysis_candidate if analysis_candidate.is_file() else None
    temporary_directory: Path | None = None
    kept_ascot_path: Path | None = None
    try:
        if args.keep_ascot_input:
            ascot_directory = folder
            suffix = "_afsi_ascot"
            kept_ascot_path = folder / f"{equilibrium_path.stem}{suffix}.h5"
            if kept_ascot_path.exists():
                kept_ascot_path.unlink()
        else:
            temporary_directory = Path(tempfile.mkdtemp(prefix=".afsi_", dir=folder))
            ascot_directory = temporary_directory
            suffix = f"_afsi_{os.getpid()}"

        # Import here so --dry-run and discovery work without loading libascot/JAX.
        from alpha_analysis import RunItem

        run = RunItem(
            str(equilibrium_path),
            path=str(ascot_directory),
            create=True,
            suffix=suffix,
            nR=args.field_nr,
            nZ=args.field_nz,
            nPhi=args.field_nphi,
            nrho=args.profile_nrho,
            fraction_T=args.fraction_tritium,
            Zeff=args.zeff,
            L_radial=args.l_radial,
            M_poloidal=args.m_poloidal,
            use_stell_sym=not args.no_stellarator_symmetry,
            waitingbar=args.show_field_progress,
            # AFSI initializes only the magnetic field and plasma. Building the
            # comparatively expensive LCFS wall is unnecessary for this task.
            include_wall=False,
        )
        run.run_afsi(
            mode="magnetic",
            descfn=str(equilibrium_path),
            nR=args.nrho_bins + 1,
            nenergy=args.nenergy_bins + 1,
            npitch=args.npitch_bins,
            nmc=args.nmc,
            nthermal_vel=args.nthermal_vel,
        )
        moments = calculate_source_moments(run.afsi_dist, equilibrium_path)
        write_output(
            output_path,
            equilibrium_path=equilibrium_path,
            analysis_path=analysis_path,
            ascot_input_path=kept_ascot_path,
            dist=run.afsi_dist,
            moments=moments,
            args=args,
        )
        return {"status": "wrote", "path": str(output_path), "message": ""}
    except Exception as exc:
        return {
            "status": "failed",
            "path": str(output_path),
            "message": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if temporary_directory is not None:
            shutil.rmtree(temporary_directory, ignore_errors=True)


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "nrho_bins",
        "nenergy_bins",
        "npitch_bins",
        "nmc",
        "nthermal_vel",
        "field_nr",
        "field_nz",
        "field_nphi",
        "profile_nrho",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.nrho_bins < 2 or args.nenergy_bins < 2:
        raise ValueError("--nrho-bins and --nenergy-bins must be at least 2.")
    if not 0.0 <= args.fraction_tritium <= 1.0:
        raise ValueError("--fraction-tritium must be between 0 and 1.")
    if args.zeff < 1.0:
        raise ValueError("--zeff must be at least 1.")
    if args.accumulation_time_s is not None and args.accumulation_time_s <= 0:
        raise ValueError("--accumulation-time-s must be positive.")


def _discover_validated(args: argparse.Namespace) -> list[Path]:
    validate_args(args)
    return discover_equilibria(args)


def _log_result(result: dict[str, str], equilibrium_path: Path, progress: str = "") -> None:
    prefix = f"{progress} " if progress else ""
    if result["status"] == "wrote":
        LOGGER.info("%sWrote %s", prefix, result["path"])
    elif result["status"] == "skipped":
        LOGGER.info("%sSkipped %s (%s)", prefix, result["path"], result["message"])
    else:
        LOGGER.error(
            "%sFailed %s: %s\n%s",
            prefix,
            equilibrium_path,
            result["message"],
            result.get("traceback", ""),
        )


def run_serial(args: argparse.Namespace) -> int:
    try:
        equilibria = _discover_validated(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    LOGGER.info("Found %d DESC equilibria to process.", len(equilibria))
    if args.dry_run:
        for path in equilibria:
            LOGGER.info("Would process %s", path)
        return 0

    wrote = skipped = failed = 0
    for index, equilibrium_path in enumerate(equilibria, start=1):
        LOGGER.info("[%d/%d] Processing %s", index, len(equilibria), equilibrium_path)
        result = process_equilibrium(equilibrium_path, args)
        _log_result(result, equilibrium_path)
        if result["status"] == "wrote":
            wrote += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1

    LOGGER.info("Done: wrote=%d skipped=%d failed=%d", wrote, skipped, failed)
    return 1 if failed else 0


def run_mpi_worker(comm: Any, args: argparse.Namespace) -> None:
    from mpi4py import MPI

    comm.send(None, dest=0, tag=TAG_READY)
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == TAG_STOP:
            return
        equilibrium_path = Path(task)
        LOGGER.info("Processing %s", equilibrium_path)
        result = process_equilibrium(equilibrium_path, args)
        comm.send((str(equilibrium_path), result), dest=0, tag=TAG_RESULT)


def run_mpi_master(comm: Any, args: argparse.Namespace, size: int) -> int:
    from mpi4py import MPI

    try:
        equilibria = _discover_validated(args)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        for worker in range(1, size):
            comm.send(None, dest=worker, tag=TAG_STOP)
        return 2

    LOGGER.info(
        "MPI dynamic queue: %d equilibria, %d worker ranks.",
        len(equilibria),
        size - 1,
    )
    if args.dry_run:
        for path in equilibria:
            LOGGER.info("Would process %s", path)
        for worker in range(1, size):
            comm.send(None, dest=worker, tag=TAG_STOP)
        return 0

    task_iter = iter(equilibria)
    active_workers = 0
    stopped_workers = 0
    for _ in range(1, size):
        status = MPI.Status()
        comm.recv(source=MPI.ANY_SOURCE, tag=TAG_READY, status=status)
        worker = status.Get_source()
        try:
            path = next(task_iter)
        except StopIteration:
            comm.send(None, dest=worker, tag=TAG_STOP)
            stopped_workers += 1
        else:
            comm.send(str(path), dest=worker, tag=TAG_WORK)
            active_workers += 1

    wrote = skipped = failed = completed = 0
    total = len(equilibria)
    while active_workers:
        status = MPI.Status()
        path_text, result = comm.recv(
            source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status
        )
        worker = status.Get_source()
        active_workers -= 1
        completed += 1
        _log_result(result, Path(path_text), progress=f"[{completed}/{total}]")
        if result["status"] == "wrote":
            wrote += 1
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1

        try:
            path = next(task_iter)
        except StopIteration:
            comm.send(None, dest=worker, tag=TAG_STOP)
            stopped_workers += 1
        else:
            comm.send(str(path), dest=worker, tag=TAG_WORK)
            active_workers += 1

    if stopped_workers != size - 1:
        raise RuntimeError("Not all MPI workers received a stop message.")
    LOGGER.info("Done: wrote=%d skipped=%d failed=%d", wrote, skipped, failed)
    return 1 if failed else 0


def run_mpi(args: argparse.Namespace) -> int:
    try:
        from mpi4py import MPI
    except ModuleNotFoundError:
        LOGGER.error("--mpi requires mpi4py in the active Python environment.")
        return 2

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    configure_logging(args.verbose, rank=rank)
    if size == 1:
        LOGGER.warning("MPI mode has one rank; falling back to serial execution.")
        return run_serial(args)
    if rank == 0:
        return run_mpi_master(comm, args, size)
    run_mpi_worker(comm, args)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.mpi:
        return run_mpi(args)
    return run_serial(args)


if __name__ == "__main__":
    raise SystemExit(main())
