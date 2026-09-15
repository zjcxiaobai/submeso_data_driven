#!/usr/bin/env python3
"""Prepare per-sample physical diagnostics for top model_analysis test samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
FEATURE_NAMES = ["U", "V", "W", "zeta", "strain", "T", "rho", "kpphbl"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-path", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/X.npy")
    parser.add_argument("--time-index", type=Path, default=PROJECT_DIR / "analysis/model_analysis/data/time_index.npy")
    parser.add_argument("--sample-index", type=int, default=None, help="Optional single sample. If omitted, use top-N test samples by R2.")
    parser.add_argument("--top-n", type=int, default=90, help="Number of best test samples to process when --sample-index is omitted.")
    parser.add_argument("--predictions", type=Path, default=PROJECT_DIR / "analysis/model_analysis/outputs/test_predictions_denorm.npz")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dx-m", type=float, default=50_000.0)
    parser.add_argument("--dy-m", type=float, default=50_000.0)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--domain-height-m", type=float, default=1.0e6)
    parser.add_argument("--beta-reference", choices=("center", "south", "north"), default="center")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ddx_periodic(field: np.ndarray, dx_m: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx_m)


def ddy_nonperiodic(field: np.ndarray, dy_m: float) -> np.ndarray:
    return np.gradient(field, dy_m, axis=0, edge_order=2)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > 1.0e-20,
    )


def beta_plane_f(height: int, f0: float, beta: float, domain_height_m: float, reference: str) -> np.ndarray:
    y = np.linspace(0.0, domain_height_m, height, dtype=np.float64)
    if reference == "south":
        y_ref = float(y.min())
    elif reference == "north":
        y_ref = float(y.max())
    else:
        y_ref = float(y.mean())
    return f0 + beta * (y[:, None] - y_ref)


def add(fields: dict[str, np.ndarray], definitions: dict[str, tuple[str, str]], name: str, value: np.ndarray, units: str, definition: str) -> None:
    fields[name] = np.asarray(value, dtype=np.float32)
    definitions[name] = (units, definition)


def calculate_fields(source: dict[str, np.ndarray], dx_m: float, dy_m: float, f0: float, beta: float, domain_height_m: float, beta_reference: str) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    u = np.asarray(source["U"], dtype=np.float64)
    v = np.asarray(source["V"], dtype=np.float64)
    w = np.asarray(source["W"], dtype=np.float64)
    temperature = np.asarray(source["T"], dtype=np.float64)
    input_vorticity = np.asarray(source["zeta"], dtype=np.float64)
    input_strain = np.asarray(source["strain"], dtype=np.float64)
    density = np.asarray(source["rho"], dtype=np.float64)
    kpphbl = np.asarray(source["kpphbl"], dtype=np.float64)
    shape = temperature.shape
    for name, field in {"U": u, "V": v, "W": w, "zeta": input_vorticity, "strain": input_strain, "rho": density, "KPPhbl": kpphbl}.items():
        if field.shape != shape:
            raise ValueError(f"{name} has shape {field.shape}, expected {shape}")

    dtdx = ddx_periodic(temperature, dx_m)
    dtdy = ddy_nonperiodic(temperature, dy_m)
    dudx = ddx_periodic(u, dx_m)
    dudy = ddy_nonperiodic(u, dy_m)
    dvdx = ddx_periodic(v, dx_m)
    dvdy = ddy_nonperiodic(v, dy_m)
    dwdx = ddx_periodic(w, dx_m)
    dwdy = ddy_nonperiodic(w, dy_m)
    drhodx = ddx_periodic(density, dx_m)
    drhody = ddy_nonperiodic(density, dy_m)
    grad_t = np.hypot(dtdx, dtdy)
    grad_rho = np.hypot(drhodx, drhody)
    divergence = dudx + dvdy
    vorticity_derived = dvdx - dudy
    normal_strain = dudx - dvdy
    shear_strain = dvdx + dudy
    strain_derived = np.hypot(normal_strain, shear_strain)
    okubo_weiss = strain_derived**2 - vorticity_derived**2

    frontogenesis_numerator = dtdx**2 * dudx + dtdy**2 * dvdy + dtdx * dtdy * (dudy + dvdx)
    gradt_dot_strain_tensor_gradt = frontogenesis_numerator
    frontogenesis_ft = safe_divide(-frontogenesis_numerator, grad_t)
    frontogenesis_rate = safe_divide(-frontogenesis_numerator, grad_t**2)
    strain_efficiency = safe_divide(2.0 * frontogenesis_ft, strain_derived * grad_t)
    qx = -(dudx * dtdx + dvdx * dtdy)
    qy = -(dudy * dtdx + dvdy * dtdy)
    q_convergence = -(ddx_periodic(qx, dx_m) + ddy_nonperiodic(qy, dy_m))

    temperature_anomaly = temperature - np.nanmean(temperature)
    u_anomaly = u - np.nanmean(u)
    v_anomaly = v - np.nanmean(v)
    w_anomaly = w - np.nanmean(w)
    density_anomaly = density - np.nanmean(density)
    horizontal_temperature_advection = -(u * dtdx + v * dtdy)
    cross_front_velocity = safe_divide(u * dtdx + v * dtdy, grad_t)
    along_front_velocity = safe_divide(-u * dtdy + v * dtdx, grad_t)
    kinetic_energy = 0.5 * (u**2 + v**2)
    eddy_kinetic_energy = 0.5 * (u_anomaly**2 + v_anomaly**2)
    enstrophy = 0.5 * vorticity_derived**2
    laplacian_temperature = ddx_periodic(dtdx, dx_m) + ddy_nonperiodic(dtdy, dy_m)
    laplacian_vorticity = ddx_periodic(ddx_periodic(vorticity_derived, dx_m), dx_m) + ddy_nonperiodic(ddy_nonperiodic(vorticity_derived, dy_m), dy_m)
    speed_derived = np.hypot(u, v)
    grad_speed = np.hypot(ddx_periodic(speed_derived, dx_m), ddy_nonperiodic(speed_derived, dy_m))
    f = np.broadcast_to(beta_plane_f(shape[0], f0, beta, domain_height_m, beta_reference), shape)
    rossby_number = safe_divide(input_vorticity, f)

    fields: dict[str, np.ndarray] = {}
    definitions: dict[str, tuple[str, str]] = {}
    add(fields, definitions, "U", u, "m s-1", "Model input zonal velocity")
    add(fields, definitions, "V", v, "m s-1", "Model input meridional velocity")
    add(fields, definitions, "W", w, "m s-1", "Model input vertical velocity")
    add(fields, definitions, "T", temperature, "degC", "Model input potential temperature")
    add(fields, definitions, "rho", density, "kg m-3", "Model input density")
    add(fields, definitions, "density", density, "kg m-3", "Alias of rho")
    add(fields, definitions, "rho_anomaly", density_anomaly, "kg m-3", "rho minus spatial mean")
    add(fields, definitions, "vorticity", input_vorticity, "s-1", "Model input relative vorticity/zeta")
    add(fields, definitions, "zeta", input_vorticity, "s-1", "Alias of input relative vorticity")
    add(fields, definitions, "strain", input_strain, "s-1", "Model input horizontal strain magnitude")
    add(fields, definitions, "KPPhbl", kpphbl, "m", "Model input KPP boundary-layer depth")
    add(fields, definitions, "dTdx", dtdx, "degC m-1", "Periodic centered zonal derivative of T")
    add(fields, definitions, "dTdy", dtdy, "degC m-1", "Nonperiodic meridional derivative of T")
    add(fields, definitions, "gradT", grad_t, "degC m-1", "sqrt(dTdx^2 + dTdy^2)")
    add(fields, definitions, "drhodx", drhodx, "kg m-4", "Periodic centered zonal derivative of rho")
    add(fields, definitions, "drhody", drhody, "kg m-4", "Nonperiodic meridional derivative of rho")
    add(fields, definitions, "grad_rho", grad_rho, "kg m-4", "sqrt(drhodx^2 + drhody^2)")
    add(fields, definitions, "dUdx", dudx, "s-1", "Periodic centered zonal derivative of U")
    add(fields, definitions, "dUdy", dudy, "s-1", "Nonperiodic meridional derivative of U")
    add(fields, definitions, "dVdx", dvdx, "s-1", "Periodic centered zonal derivative of V")
    add(fields, definitions, "dVdy", dvdy, "s-1", "Nonperiodic meridional derivative of V")
    add(fields, definitions, "dWdx", dwdx, "s-1", "Periodic centered zonal derivative of W")
    add(fields, definitions, "dWdy", dwdy, "s-1", "Nonperiodic meridional derivative of W")
    add(fields, definitions, "divergence", divergence, "s-1", "dUdx + dVdy")
    add(fields, definitions, "vorticity_derived", vorticity_derived, "s-1", "dVdx - dUdy")
    add(fields, definitions, "normal_strain", normal_strain, "s-1", "dUdx - dVdy")
    add(fields, definitions, "shear_strain", shear_strain, "s-1", "dVdx + dUdy")
    add(fields, definitions, "strain_derived", strain_derived, "s-1", "sqrt(normal_strain^2 + shear_strain^2)")
    add(fields, definitions, "Okubo_Weiss", okubo_weiss, "s-2", "strain_derived^2 - vorticity_derived^2")
    add(fields, definitions, "frontogenesis_FT", frontogenesis_ft, "degC m-1 s-1", "-(Tx^2 Ux + Ty^2 Vy + Tx Ty (Uy + Vx))/|gradT|")
    add(fields, definitions, "Fx", frontogenesis_ft, "degC m-1 s-1", "Alias of frontogenesis_FT")
    add(fields, definitions, "frontogenesis_rate", frontogenesis_rate, "s-1", "frontogenesis_FT/|gradT|")
    add(fields, definitions, "strain_efficiency", strain_efficiency, "1", "2*frontogenesis_FT/(strain_derived*|gradT|); positive when strain orientation sharpens the temperature front")
    add(fields, definitions, "Qx", qx, "degC m-1 s-1", "-(Ux Tx + Vx Ty)")
    add(fields, definitions, "Qy", qy, "degC m-1 s-1", "-(Uy Tx + Vy Ty)")
    add(fields, definitions, "Q_convergence", q_convergence, "degC m-2 s-1", "-(dQx/dx + dQy/dy)")
    add(fields, definitions, "T_anomaly", temperature_anomaly, "degC", "T minus its spatial mean")
    add(fields, definitions, "U_anomaly", u_anomaly, "m s-1", "U minus spatial mean")
    add(fields, definitions, "V_anomaly", v_anomaly, "m s-1", "V minus spatial mean")
    add(fields, definitions, "W_anomaly", w_anomaly, "m s-1", "W minus spatial mean")
    add(fields, definitions, "W_times_Tprime", w * temperature_anomaly, "degC m s-1", "W times spatial T anomaly")
    add(fields, definitions, "Wprime_times_Tprime", w_anomaly * temperature_anomaly, "degC m s-1", "Spatial W anomaly times spatial T anomaly")
    add(fields, definitions, "W_times_gradT", w * grad_t, "degC s-1", "W times horizontal temperature-gradient magnitude")
    add(fields, definitions, "strain_times_gradT", input_strain * grad_t, "degC m-1 s-1", "Input strain times |gradT|")
    add(fields, definitions, "S_gradT", input_strain * grad_t, "degC m-1 s-1", "Alias of input strain times |gradT|")
    add(fields, definitions, "gradT_dot_S_gradT", gradt_dot_strain_tensor_gradt, "degC2 m-2 s-1", "gradT dot symmetric horizontal velocity-gradient tensor dot gradT")
    add(fields, definitions, "strain_times_grad_rho", input_strain * grad_rho, "kg m-4 s-1", "Input strain times |grad rho|")
    add(fields, definitions, "S_grad_rho", input_strain * grad_rho, "kg m-4 s-1", "Alias of input strain times |grad rho|")
    add(fields, definitions, "abs_Ro_times_gradT", np.abs(rossby_number) * grad_t, "degC m-1", "|Ro| times |gradT|")
    add(fields, definitions, "okubo_weiss_input_times_gradT", (input_strain**2 - input_vorticity**2) * grad_t, "degC m-1 s-2", "(input strain^2 - input vorticity^2) times |gradT|")
    add(fields, definitions, "hbl_strain_times_gradT", kpphbl * input_strain * grad_t, "degC s-1", "KPPHBL times input strain times |gradT|")
    add(fields, definitions, "vorticity_times_gradT", input_vorticity * grad_t, "degC m-1 s-1", "Input vorticity times |gradT|")
    add(fields, definitions, "horizontal_T_advection", horizontal_temperature_advection, "degC s-1", "-(U dTdx + V dTdy)")
    add(fields, definitions, "cross_front_velocity", cross_front_velocity, "m s-1", "Velocity projected along gradT")
    add(fields, definitions, "along_front_velocity", along_front_velocity, "m s-1", "Velocity projected along temperature contours")
    add(fields, definitions, "kinetic_energy", kinetic_energy, "m2 s-2", "0.5*(U^2 + V^2)")
    add(fields, definitions, "eddy_kinetic_energy", eddy_kinetic_energy, "m2 s-2", "0.5*(Uprime^2 + Vprime^2)")
    add(fields, definitions, "enstrophy", enstrophy, "s-2", "0.5*vorticity_derived^2")
    add(fields, definitions, "grad_speed", grad_speed, "s-1", "Horizontal gradient magnitude of derived speed")
    add(fields, definitions, "laplacian_T", laplacian_temperature, "degC m-2", "Horizontal Laplacian of T")
    add(fields, definitions, "laplacian_vorticity", laplacian_vorticity, "m-2 s-1", "Horizontal Laplacian of derived vorticity")
    add(fields, definitions, "coriolis_f", f, "s-1", "f0 + beta*(y-y_reference)")
    add(fields, definitions, "Rossby_number", rossby_number, "1", "Input vorticity divided by beta-plane f")
    add(fields, definitions, "Rossby_number_derived", safe_divide(vorticity_derived, f), "1", "Derived vorticity divided by beta-plane f")
    add(fields, definitions, "normalized_divergence", safe_divide(divergence, np.abs(f)), "1", "Divergence divided by |f|")
    add(fields, definitions, "strain_vorticity_ratio", safe_divide(strain_derived, np.abs(vorticity_derived)), "1", "Derived strain divided by |derived vorticity|")
    return fields, definitions



def select_sample_indices(args: argparse.Namespace, x_length: int) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if args.sample_index is not None:
        sample_indices = np.asarray([int(args.sample_index)], dtype=np.int64)
        return sample_indices, None, None
    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing prediction file for top-N selection: {args.predictions}")
    with np.load(args.predictions) as data:
        required = {"sample_index", "r2"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{args.predictions} is missing arrays: {sorted(missing)}")
        sample = np.asarray(data["sample_index"], dtype=np.int64)
        r2 = np.asarray(data["r2"], dtype=np.float64)
        pred_time = np.asarray(data["time_index"], dtype=np.int64) if "time_index" in data.files and data["time_index"].size else None
    valid = np.isfinite(r2) & (sample >= 0) & (sample < x_length)
    if not np.any(valid):
        raise ValueError("No finite valid R2 values found for top-N selection.")
    order = np.argsort(r2[valid])[::-1]
    sample_valid = sample[valid]
    r2_valid = r2[valid]
    time_valid = pred_time[valid] if pred_time is not None else None
    keep = order[: int(args.top_n)]
    return sample_valid[keep].astype(np.int64), r2_valid[keep].astype(np.float64), None if time_valid is None else time_valid[keep].astype(np.int64)

def main() -> None:
    args = parse_args()
    x_path = args.x_path.expanduser().resolve()
    x = np.load(x_path, mmap_mode="r")
    if x.ndim != 4 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected X shape (N,{len(FEATURE_NAMES)},20,40), got {x.shape}")

    sample_indices, selected_r2, selected_pred_time = select_sample_indices(args, x.shape[0])
    time_all = np.load(args.time_index, mmap_mode="r") if args.time_index.exists() else None
    data_time = np.asarray(time_all[sample_indices], dtype=np.int64) if time_all is not None else None

    accumulated: dict[str, list[np.ndarray]] = {}
    definitions: dict[str, tuple[str, str]] | None = None
    for n, sample_index in enumerate(sample_indices, start=1):
        source = {name: np.asarray(x[int(sample_index), i], dtype=np.float64) for i, name in enumerate(FEATURE_NAMES)}
        fields, current_definitions = calculate_fields(
            source,
            args.dx_m,
            args.dy_m,
            args.f0,
            args.beta,
            args.domain_height_m,
            args.beta_reference,
        )
        if definitions is None:
            definitions = current_definitions
            accumulated = {name: [] for name in fields}
        for name, field in fields.items():
            accumulated[name].append(field)
        if n == 1 or n % 10 == 0 or n == sample_indices.size:
            print(f"Prepared physics fields for {n}/{sample_indices.size} samples", flush=True)

    field_stacks = {name: np.stack(values, axis=0).astype(np.float32) for name, values in accumulated.items()}
    assert definitions is not None

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif sample_indices.size == 1:
        output_dir = HERE / f"physics_variables_sample_{int(sample_indices[0])}"
    else:
        output_dir = HERE / f"physics_variables_top{sample_indices.size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{name}.npy" for name in field_stacks]
    existing.extend([output_dir / "physics_variables.npz", output_dir / "physics_variable_manifest.csv", output_dir / "selected_samples.csv", output_dir / "metadata.json"])
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"{output_dir} already contains outputs; pass --overwrite")

    for name, field in field_stacks.items():
        np.save(output_dir / f"{name}.npy", field)
    np.savez_compressed(
        output_dir / "physics_variables.npz",
        sample_index=sample_indices.astype(np.int64),
        time_index=np.asarray([], dtype=np.int64) if data_time is None else data_time.astype(np.int64),
        selected_r2=np.asarray([], dtype=np.float64) if selected_r2 is None else selected_r2.astype(np.float64),
        **field_stacks,
    )
    with (output_dir / "physics_variable_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variable", "units", "definition", "filename", "shape", "dtype"])
        writer.writeheader()
        for name, field in field_stacks.items():
            units, definition = definitions[name]
            writer.writerow({"variable": name, "units": units, "definition": definition, "filename": f"{name}.npy", "shape": "x".join(str(value) for value in field.shape), "dtype": str(field.dtype)})
    with (output_dir / "selected_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "sample_index", "time_index", "r2"])
        writer.writeheader()
        for rank, sample_index in enumerate(sample_indices, start=1):
            writer.writerow({
                "rank": rank,
                "sample_index": int(sample_index),
                "time_index": "" if data_time is None else int(data_time[rank - 1]),
                "r2": "" if selected_r2 is None else float(selected_r2[rank - 1]),
            })
    metadata = {
        "selection": "single_sample" if sample_indices.size == 1 else f"top_{sample_indices.size}_test_samples_by_r2",
        "sample_index": sample_indices.astype(int).tolist(),
        "time_index": None if data_time is None else data_time.astype(int).tolist(),
        "selected_r2": None if selected_r2 is None else selected_r2.astype(float).tolist(),
        "mean_selected_r2": None if selected_r2 is None else float(np.nanmean(selected_r2)),
        "x_path": str(x_path),
        "features": FEATURE_NAMES,
        "shape_nyx": list(next(iter(field_stacks.values())).shape),
        "dx_m": args.dx_m,
        "dy_m": args.dy_m,
        "periodic_x": True,
        "periodic_y": False,
        "variable_count": len(field_stacks),
        "notes": ["Physics diagnostics are saved per selected model_analysis sample, shape (N, 20, 40).", "rho is the density channel used by the new model.", "speed is not saved as a diagnostic; grad_speed is retained and derived from U and V for each sample."],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(field_stacks)} per-sample aligned physics variables to {output_dir}")


if __name__ == "__main__":
    main()
