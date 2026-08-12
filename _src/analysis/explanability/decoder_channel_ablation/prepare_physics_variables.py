#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "decoder_channel_ablation_best_snapshot.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare current-frame physical and dynamical fields aligned with the saved "
            "decoder feature maps. No feature-map correlations are calculated."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dx-m", type=float, default=50_000.0)
    parser.add_argument("--dy-m", type=float, default=50_000.0)
    parser.add_argument("--f0", type=float, default=-1.363e-4)
    parser.add_argument("--beta", type=float, default=1.313e-11)
    parser.add_argument("--domain-height-m", type=float, default=1.0e6)
    parser.add_argument(
        "--beta-reference",
        choices=["center", "south", "north"],
        default="center",
    )
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


def beta_plane_f(
    height: int,
    f0: float,
    beta: float,
    domain_height_m: float,
    reference: str,
) -> np.ndarray:
    y = np.linspace(0.0, domain_height_m, height, dtype=np.float64)
    if reference == "south":
        y_ref = float(y.min())
    elif reference == "north":
        y_ref = float(y.max())
    else:
        y_ref = float(y.mean())
    return f0 + beta * (y[:, None] - y_ref)


def add(
    fields: dict[str, np.ndarray],
    definitions: dict[str, tuple[str, str]],
    name: str,
    value: np.ndarray,
    units: str,
    definition: str,
) -> None:
    fields[name] = np.asarray(value, dtype=np.float32)
    definitions[name] = (units, definition)


def calculate_fields(
    source: np.lib.npyio.NpzFile,
    dx_m: float,
    dy_m: float,
    f0: float,
    beta: float,
    domain_height_m: float,
    beta_reference: str,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, str]]]:
    u = np.asarray(source["U"], dtype=np.float64)
    v = np.asarray(source["V"], dtype=np.float64)
    w = np.asarray(source["W"], dtype=np.float64)
    temperature = np.asarray(source["T"], dtype=np.float64)
    input_vorticity = np.asarray(source["vorticity"], dtype=np.float64)
    input_strain = np.asarray(source["strain"], dtype=np.float64)
    input_speed = np.asarray(source["speed"], dtype=np.float64)
    kpphbl = np.asarray(source["KPPhbl"], dtype=np.float64)
    shape = temperature.shape
    for name, field in {
        "U": u,
        "V": v,
        "W": w,
        "vorticity": input_vorticity,
        "strain": input_strain,
        "speed": input_speed,
        "KPPhbl": kpphbl,
    }.items():
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
    grad_t = np.hypot(dtdx, dtdy)
    divergence = dudx + dvdy
    vorticity_derived = dvdx - dudy
    normal_strain = dudx - dvdy
    shear_strain = dvdx + dudy
    strain_derived = np.hypot(normal_strain, shear_strain)
    okubo_weiss = strain_derived**2 - vorticity_derived**2

    frontogenesis_numerator = (
        dtdx**2 * dudx
        + dtdy**2 * dvdy
        + dtdx * dtdy * (dudy + dvdx)
    )
    frontogenesis_ft = safe_divide(-frontogenesis_numerator, grad_t)
    frontogenesis_rate = safe_divide(-frontogenesis_numerator, grad_t**2)
    qx = -(dudx * dtdx + dvdx * dtdy)
    qy = -(dudy * dtdx + dvdy * dtdy)
    q_convergence = -(ddx_periodic(qx, dx_m) + ddy_nonperiodic(qy, dy_m))

    temperature_anomaly = temperature - np.nanmean(temperature)
    u_anomaly = u - np.nanmean(u)
    v_anomaly = v - np.nanmean(v)
    w_anomaly = w - np.nanmean(w)
    horizontal_temperature_advection = -(u * dtdx + v * dtdy)
    cross_front_velocity = safe_divide(u * dtdx + v * dtdy, grad_t)
    along_front_velocity = safe_divide(-u * dtdy + v * dtdx, grad_t)
    kinetic_energy = 0.5 * (u**2 + v**2)
    eddy_kinetic_energy = 0.5 * (u_anomaly**2 + v_anomaly**2)
    enstrophy = 0.5 * vorticity_derived**2
    laplacian_temperature = (
        ddx_periodic(dtdx, dx_m) + ddy_nonperiodic(dtdy, dy_m)
    )
    laplacian_vorticity = (
        ddx_periodic(ddx_periodic(vorticity_derived, dx_m), dx_m)
        + ddy_nonperiodic(ddy_nonperiodic(vorticity_derived, dy_m), dy_m)
    )
    speed_derived = np.hypot(u, v)
    grad_speed = np.hypot(
        ddx_periodic(speed_derived, dx_m),
        ddy_nonperiodic(speed_derived, dy_m),
    )
    f = np.broadcast_to(
        beta_plane_f(
            shape[0], f0, beta, domain_height_m, beta_reference
        ),
        shape,
    )

    fields: dict[str, np.ndarray] = {}
    definitions: dict[str, tuple[str, str]] = {}
    add(fields, definitions, "U", u, "m s-1", "Model input zonal velocity")
    add(fields, definitions, "V", v, "m s-1", "Model input meridional velocity")
    add(fields, definitions, "W", w, "m s-1", "Model input vertical velocity")
    add(fields, definitions, "T", temperature, "degC", "Model input potential temperature")
    add(fields, definitions, "vorticity", input_vorticity, "s-1", "Model input relative vorticity")
    add(fields, definitions, "strain", input_strain, "s-1", "Model input horizontal strain magnitude")
    add(fields, definitions, "speed", input_speed, "m s-1", "Model input horizontal speed")
    add(fields, definitions, "KPPhbl", kpphbl, "m", "Model input KPP boundary-layer depth")
    add(fields, definitions, "dTdx", dtdx, "degC m-1", "Periodic centered zonal derivative of T")
    add(fields, definitions, "dTdy", dtdy, "degC m-1", "Nonperiodic meridional derivative of T")
    add(fields, definitions, "gradT", grad_t, "degC m-1", "sqrt(dTdx^2 + dTdy^2)")
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
    add(fields, definitions, "Fx", frontogenesis_ft, "degC m-1 s-1", "Alias of frontogenesis_FT retained for requested Fx naming")
    add(fields, definitions, "frontogenesis_rate", frontogenesis_rate, "s-1", "frontogenesis_FT/|gradT|")
    add(fields, definitions, "Qx", qx, "degC m-1 s-1", "-(Ux Tx + Vx Ty)")
    add(fields, definitions, "Qy", qy, "degC m-1 s-1", "-(Uy Tx + Vy Ty)")
    add(fields, definitions, "Q_convergence", q_convergence, "degC m-2 s-1", "-(dQx/dx + dQy/dy)")
    add(fields, definitions, "T_anomaly", temperature_anomaly, "degC", "T minus its spatial mean")
    add(fields, definitions, "U_anomaly", u_anomaly, "m s-1", "U minus its spatial mean")
    add(fields, definitions, "V_anomaly", v_anomaly, "m s-1", "V minus its spatial mean")
    add(fields, definitions, "W_anomaly", w_anomaly, "m s-1", "W minus its spatial mean")
    add(fields, definitions, "W_times_Tprime", w * temperature_anomaly, "degC m s-1", "W times spatial T anomaly")
    add(fields, definitions, "Wprime_times_Tprime", w_anomaly * temperature_anomaly, "degC m s-1", "Spatial W anomaly times spatial T anomaly")
    add(fields, definitions, "W_times_gradT", w * grad_t, "degC s-1", "W times horizontal temperature-gradient magnitude")
    add(fields, definitions, "strain_times_gradT", input_strain * grad_t, "degC m-1 s-1", "Input strain times |gradT|")
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
    add(fields, definitions, "Rossby_number", safe_divide(input_vorticity, f), "1", "Input vorticity divided by beta-plane f")
    add(fields, definitions, "Rossby_number_derived", safe_divide(vorticity_derived, f), "1", "Derived vorticity divided by beta-plane f")
    add(fields, definitions, "normalized_divergence", safe_divide(divergence, np.abs(f)), "1", "Divergence divided by |f|")
    add(fields, definitions, "strain_vorticity_ratio", safe_divide(strain_derived, np.abs(vorticity_derived)), "1", "Derived strain divided by |derived vorticity|")
    return fields, definitions


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    with np.load(input_path, allow_pickle=True) as source:
        sample_index = int(source["sample_index"])
        fields, definitions = calculate_fields(
            source,
            args.dx_m,
            args.dy_m,
            args.f0,
            args.beta,
            args.domain_height_m,
            args.beta_reference,
        )
        source_config = str(source["config"]) if "config" in source.files else ""
        source_checkpoint = str(source["checkpoint"]) if "checkpoint" in source.files else ""

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else HERE / f"physics_variables_sample_{sample_index}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / f"{name}.npy" for name in fields]
    existing.extend(
        [
            output_dir / "physics_variables.npz",
            output_dir / "physics_variable_manifest.csv",
            output_dir / "metadata.json",
        ]
    )
    if not args.overwrite and any(path.exists() for path in existing):
        raise FileExistsError(f"{output_dir} already contains outputs; pass --overwrite")

    for name, field in fields.items():
        np.save(output_dir / f"{name}.npy", field)
    np.savez_compressed(
        output_dir / "physics_variables.npz",
        sample_index=np.asarray(sample_index, dtype=np.int64),
        **fields,
    )
    with (output_dir / "physics_variable_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variable", "units", "definition", "filename", "shape", "dtype"],
        )
        writer.writeheader()
        for name, field in fields.items():
            units, definition = definitions[name]
            writer.writerow(
                {
                    "variable": name,
                    "units": units,
                    "definition": definition,
                    "filename": f"{name}.npy",
                    "shape": "x".join(str(value) for value in field.shape),
                    "dtype": str(field.dtype),
                }
            )
    metadata = {
        "sample_index": sample_index,
        "source": str(input_path),
        "source_config": source_config,
        "source_checkpoint": source_checkpoint,
        "shape_yx": list(next(iter(fields.values())).shape),
        "dx_m": args.dx_m,
        "dy_m": args.dy_m,
        "periodic_x": True,
        "periodic_y": False,
        "f0_s-1": args.f0,
        "beta_s-1_m-1": args.beta,
        "domain_height_m": args.domain_height_m,
        "beta_reference": args.beta_reference,
        "variable_count": len(fields),
        "correlations_calculated": False,
        "notes": [
            "All fields are aligned with the current-frame 20x40 decoder feature maps.",
            "Zonal derivatives use periodic centered differences.",
            "Meridional derivatives use nonperiodic second-order NumPy gradients.",
            "Fx is an explicit alias of the previously used frontogenesis_FT diagnostic.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(fields)} aligned physics variables to {output_dir}")


if __name__ == "__main__":
    main()
