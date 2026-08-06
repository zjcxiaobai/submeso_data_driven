#!/usr/bin/env python3
"""Calculate temperature-density correlations for the reviewer response.

This script follows the density calculation used in ``src/utils/buo.py``:
practical salinity + potential temperature -> absolute salinity +
conservative temperature -> in-situ density with TEOS-10/GSW.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import gsw
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency: gsw. Run this in the project science environment.") from exc

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit("Missing dependency: xarray. Run this in the project science environment.") from exc


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DIR = PROJECT_DIR / "data/mld"
HERE = Path(__file__).resolve().parent

DEFAULT_CROPS = {
    "none": None,
    "upper_40x20": (slice(30, 50), slice(10, 50)),
    "lower_40x20": (slice(10, 30), slice(10, 50)),
    "lower_left": (slice(10, 30), slice(10, 30)),
    "lower_right": (slice(10, 30), slice(30, 50)),
    "upper_left": (slice(30, 50), slice(10, 30)),
    "upper_right": (slice(30, 50), slice(30, 50)),
}


@dataclass(frozen=True)
class CorrelationResult:
    name: str
    n: int
    pearson: float
    abs_pearson: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute correlations among potential temperature, salinity, and TEOS-10 density. "
            "Defaults point to the mixed-layer averaged NPZ files in data/mld."
        )
    )
    parser.add_argument(
        "--t-path",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "T.npz",
        help="Potential-temperature file. Supports NetCDF, .npz with data, or .npy.",
    )
    parser.add_argument(
        "--s-path",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "S.npz",
        help="Practical-salinity file. Supports NetCDF, .npz with data, or .npy.",
    )
    parser.add_argument(
        "--eta-path",
        type=Path,
        default=None,
        help="Optional sea-surface-height file used only when depth coordinates are available.",
    )
    parser.add_argument("--t-var", default=None, help="Variable name in the T NetCDF file.")
    parser.add_argument("--s-var", default=None, help="Variable name in the S NetCDF file.")
    parser.add_argument("--eta-var", default=None, help="Variable name in the Eta NetCDF file.")
    parser.add_argument(
        "--x-path",
        type=Path,
        default=None,
        help="Optional X.npy to read temperature from model channel instead of --t-path.",
    )
    parser.add_argument(
        "--t-channel",
        type=int,
        default=5,
        help="Temperature channel in X.npy. The final eight-feature model uses channel 5.",
    )
    parser.add_argument(
        "--pressure-dbar",
        type=float,
        default=0.0,
        help=(
            "Fallback pressure for 2-D or mixed-layer-averaged fields. "
            "Used when no vertical coordinate is found."
        ),
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=-55.0,
        help="Latitude used for pressure and SA conversion, matching src/utils/buo.py.",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=153.0,
        help="Longitude used for SA conversion, matching src/utils/buo.py.",
    )
    parser.add_argument(
        "--crop",
        choices=sorted(DEFAULT_CROPS),
        default="none",
        help="Optional crop matching the small-domain paper tests.",
    )
    parser.add_argument(
        "--include-gradients",
        action="store_true",
        help="Also correlate horizontal |grad T| and |grad rho|.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on the number of time samples for a quick test.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=HERE / "T_S_rho_correlation",
        help="Output prefix for CSV and JSON summaries.",
    )
    return parser.parse_args()


def first_data_var(ds: xr.Dataset, preferred: str | None) -> xr.DataArray:
    if preferred is not None:
        if preferred not in ds:
            raise KeyError(f"{preferred!r} not found. Available variables: {list(ds.data_vars)}")
        return ds[preferred]
    if len(ds.data_vars) != 1:
        for name in ("T", "S", "Eta", "THETA", "Salt", "rho"):
            if name in ds:
                return ds[name]
        raise ValueError(f"Cannot infer variable from {list(ds.data_vars)}; pass --*-var.")
    return ds[next(iter(ds.data_vars))]


def load_array(path: Path, var_name: str | None = None) -> xr.DataArray:
    path = path.expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
        return xr.DataArray(arr, dims=tuple(f"dim_{i}" for i in range(arr.ndim)), name=path.stem)
    if suffix == ".npz":
        with np.load(path) as data:
            key = var_name or ("data" if "data" in data.files else data.files[0])
            arr = np.asarray(data[key])
            coords = {}
            dims = tuple(f"dim_{i}" for i in range(arr.ndim))
            if "time_index" in data.files and arr.ndim >= 1:
                coords[dims[0]] = data["time_index"]
        return xr.DataArray(arr, dims=dims, coords=coords, name=path.stem)
    ds = xr.open_dataset(path, decode_times=False)
    return first_data_var(ds, var_name).load()


def temperature_from_x(path: Path, channel: int) -> xr.DataArray:
    x = np.load(path.expanduser().resolve(), mmap_mode="r")
    if x.ndim != 4:
        raise ValueError(f"Expected X.npy shape (time, channel, y, x), got {x.shape}")
    if not 0 <= channel < x.shape[1]:
        raise ValueError(f"Temperature channel {channel} is outside X channel count {x.shape[1]}")
    return xr.DataArray(np.asarray(x[:, channel], dtype=np.float32), dims=("time", "y", "x"), name="T")


def rename_spatial_dims(da: xr.DataArray) -> xr.DataArray:
    """Normalize common dimension names while preserving coordinates."""
    rename = {}
    dims = list(da.dims)
    if len(dims) >= 2:
        y_dim, x_dim = dims[-2], dims[-1]
        if y_dim not in {"y", "j"}:
            rename[y_dim] = "y"
        if x_dim not in {"x", "i"}:
            rename[x_dim] = "x"
    return da.rename(rename) if rename else da


def apply_crop(da: xr.DataArray, crop_name: str) -> xr.DataArray:
    crop = DEFAULT_CROPS[crop_name]
    if crop is None:
        return da
    y_slice, x_slice = crop
    da = rename_spatial_dims(da)
    y_dim, x_dim = da.dims[-2], da.dims[-1]
    return da.isel({y_dim: y_slice, x_dim: x_slice})


def maybe_limit_samples(da: xr.DataArray, sample_limit: int | None) -> xr.DataArray:
    if sample_limit is None:
        return da
    if sample_limit <= 0:
        raise ValueError("--sample-limit must be positive")
    return da.isel({da.dims[0]: slice(0, sample_limit)}) if da.ndim >= 3 else da


def find_vertical_coord(da: xr.DataArray) -> xr.DataArray | None:
    for name in ("RC", "Z", "depth", "z", "lev", "k"):
        if name in da.coords and da.coords[name].ndim == 1 and name in da.dims:
            values = da.coords[name]
            if np.issubdtype(values.dtype, np.number):
                return values
    return None


def pressure_field(
    temperature: xr.DataArray,
    eta: xr.DataArray | None,
    *,
    fallback_dbar: float,
    lat: float,
) -> xr.DataArray:
    z_coord = find_vertical_coord(temperature)
    if z_coord is None:
        return xr.zeros_like(temperature) + fallback_dbar

    z = z_coord
    if np.nanmean(np.asarray(z.values, dtype=np.float64)) > 0:
        z = -z
    z_rel = xr.zeros_like(temperature) + z
    if eta is not None:
        eta_aligned = eta
        if eta_aligned.ndim == temperature.ndim - 1:
            z_rel = z_rel - eta_aligned

    return xr.apply_ufunc(
        gsw.p_from_z,
        z_rel,
        xr.zeros_like(temperature) + lat,
        dask="parallelized",
        output_dtypes=[temperature.dtype],
    )


def calculate_density(
    temperature: xr.DataArray,
    salinity: xr.DataArray,
    eta: xr.DataArray | None,
    *,
    pressure_dbar: float,
    lat: float,
    lon: float,
) -> xr.DataArray:
    temperature, salinity = xr.align(temperature, salinity, join="inner")
    if eta is not None:
        eta = eta.squeeze(drop=True)
    p_dbar = pressure_field(temperature, eta, fallback_dbar=pressure_dbar, lat=lat)
    lat_da = xr.zeros_like(temperature) + lat
    lon_da = xr.zeros_like(temperature) + np.mod(lon, 360.0)

    sa = xr.apply_ufunc(
        gsw.SA_from_SP,
        salinity,
        p_dbar,
        lon_da,
        lat_da,
        dask="parallelized",
        output_dtypes=[temperature.dtype],
    )
    ct = xr.apply_ufunc(
        gsw.CT_from_pt,
        sa,
        temperature,
        dask="parallelized",
        output_dtypes=[temperature.dtype],
    )
    rho = xr.apply_ufunc(
        gsw.rho,
        sa,
        ct,
        p_dbar,
        dask="parallelized",
        output_dtypes=[temperature.dtype],
    )
    rho.name = "rho"
    rho.attrs.update(units="kg m-3", long_name="in-situ density from TEOS-10")
    return rho


def finite_pair(a: xr.DataArray | np.ndarray, b: xr.DataArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(aa) & np.isfinite(bb)
    if ok.sum() < 3:
        raise ValueError("Fewer than three finite paired samples.")
    return aa[ok], bb[ok]


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom == 0:
        return float("nan")
    return float(np.sum(a * b) / denom)


def corr_result(name: str, a: xr.DataArray | np.ndarray, b: xr.DataArray | np.ndarray) -> CorrelationResult:
    aa, bb = finite_pair(a, b)
    r = pearson(aa, bb)
    return CorrelationResult(name=name, n=int(aa.size), pearson=r, abs_pearson=abs(r))


def spatial_anomaly(da: xr.DataArray) -> xr.DataArray:
    spatial_dims = da.dims[-2:]
    return da - da.mean(dim=spatial_dims, skipna=True)


def gradient_magnitude(da: xr.DataArray) -> xr.DataArray:
    da = rename_spatial_dims(da)
    y_dim, x_dim = da.dims[-2], da.dims[-1]
    values = np.asarray(da, dtype=np.float64)
    grad_y, grad_x = np.gradient(values, axis=(-2, -1))
    mag = np.hypot(grad_x, grad_y)
    return xr.DataArray(mag, dims=da.dims, coords=da.coords, name=f"grad_{da.name or 'field'}")


def write_csv(path: Path, rows: Iterable[CorrelationResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "n", "pearson", "abs_pearson"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.name,
                    "n": row.n,
                    "pearson": f"{row.pearson:.12g}",
                    "abs_pearson": f"{row.abs_pearson:.12g}",
                }
            )


def main() -> None:
    args = parse_args()

    if args.x_path is not None:
        temperature = temperature_from_x(args.x_path, args.t_channel)
    else:
        temperature = load_array(args.t_path, args.t_var)
    salinity = load_array(args.s_path, args.s_var)
    eta = load_array(args.eta_path, args.eta_var) if args.eta_path and args.eta_path.exists() else None

    temperature = maybe_limit_samples(apply_crop(temperature.squeeze(drop=True), args.crop), args.sample_limit)
    salinity = maybe_limit_samples(apply_crop(salinity.squeeze(drop=True), args.crop), args.sample_limit)
    if eta is not None:
        eta = maybe_limit_samples(apply_crop(eta.squeeze(drop=True), args.crop), args.sample_limit)

    rho = calculate_density(
        temperature,
        salinity,
        eta,
        pressure_dbar=float(args.pressure_dbar),
        lat=float(args.lat),
        lon=float(args.lon),
    )
    temperature, salinity, rho = xr.align(temperature, salinity, rho, join="inner")

    t_anom = spatial_anomaly(temperature)
    s_anom = spatial_anomaly(salinity)
    rho_anom = spatial_anomaly(rho)

    rows = [
        corr_result("T_vs_S_raw", temperature, salinity),
        corr_result("T_vs_rho_raw", temperature, rho),
        corr_result("minus_T_vs_rho_raw", -temperature, rho),
        corr_result("S_vs_rho_raw", salinity, rho),
        corr_result("T_anomaly_vs_S_anomaly", t_anom, s_anom),
        corr_result("T_anomaly_vs_rho_anomaly", t_anom, rho_anom),
        corr_result("minus_T_anomaly_vs_rho_anomaly", -t_anom, rho_anom),
        corr_result("S_anomaly_vs_rho_anomaly", s_anom, rho_anom),
    ]

    if args.include_gradients:
        rows.extend(
            [
                corr_result("grad_abs_T_vs_grad_abs_S", gradient_magnitude(temperature), gradient_magnitude(salinity)),
                corr_result("grad_abs_T_vs_grad_abs_rho", gradient_magnitude(temperature), gradient_magnitude(rho)),
                corr_result("grad_abs_S_vs_grad_abs_rho", gradient_magnitude(salinity), gradient_magnitude(rho)),
                corr_result(
                    "grad_abs_T_anomaly_vs_grad_abs_S_anomaly",
                    gradient_magnitude(t_anom),
                    gradient_magnitude(s_anom),
                ),
                corr_result(
                    "grad_abs_T_anomaly_vs_grad_abs_rho_anomaly",
                    gradient_magnitude(t_anom),
                    gradient_magnitude(rho_anom),
                ),
                corr_result(
                    "grad_abs_S_anomaly_vs_grad_abs_rho_anomaly",
                    gradient_magnitude(s_anom),
                    gradient_magnitude(rho_anom),
                ),
            ]
        )

    output_prefix = args.output_prefix.expanduser().resolve()
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    write_csv(csv_path, rows)

    summary = {
        "temperature_path": str(args.x_path or args.t_path),
        "salinity_path": str(args.s_path),
        "eta_path": str(args.eta_path) if args.eta_path and args.eta_path.exists() else None,
        "temperature_channel": args.t_channel if args.x_path is not None else None,
        "crop": args.crop,
        "pressure_dbar_fallback": float(args.pressure_dbar),
        "lat": float(args.lat),
        "lon": float(args.lon),
        "temperature_shape": list(temperature.shape),
        "salinity_shape": list(salinity.shape),
        "density_shape": list(rho.shape),
        "results": [row.__dict__ for row in rows],
    }
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print("Temperature-salinity-density correlation results")
    for row in rows:
        print(f"{row.name}: r={row.pearson:+.6f}, |r|={row.abs_pearson:.6f}, n={row.n}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
