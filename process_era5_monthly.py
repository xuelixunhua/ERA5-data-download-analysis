# -*- coding: utf-8 -*-
"""
Clean ERA5 monthly NetCDF data into analysis-ready CSV files.

This script keeps the public example small:
1. read ERA5 NetCDF files, including CDS files that are actually ZIP archives;
2. convert core variables into business-friendly units;
3. optionally aggregate grid cells by masks such as province_masks.npz.
"""

from __future__ import annotations

import argparse
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


ERA5_VARIABLES = ("t2m", "u10", "v10", "u100", "v100", "ssrd", "tp", "si10")


@contextmanager
def open_dataset_from_bytes(name: str, data: bytes) -> Iterator[Any]:
    import netCDF4 as nc4

    ds = nc4.Dataset(name, memory=data)
    try:
        yield ds
    finally:
        ds.close()


@contextmanager
def open_dataset_from_path(path: Path) -> Iterator[Any]:
    import netCDF4 as nc4

    ds = nc4.Dataset(str(path))
    try:
        yield ds
    finally:
        ds.close()


def iter_datasets(path: Path) -> Iterator[tuple[str, Any]]:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".nc"):
                    continue
                data = archive.read(name)
                with open_dataset_from_bytes(name, data) as ds:
                    yield name, ds
    else:
        with open_dataset_from_path(path) as ds:
            yield path.name, ds


def decode_time_value(time_var: Any, raw_value: Any) -> pd.Timestamp:
    import netCDF4 as nc4

    units = getattr(time_var, "units", None)
    calendar = getattr(time_var, "calendar", "standard")
    if units:
        decoded = nc4.num2date(raw_value, units=units, calendar=calendar)
        return pd.Timestamp(decoded.isoformat())
    return pd.Timestamp(int(raw_value), unit="s")


def as_array(value: Any) -> np.ndarray:
    return np.asarray(np.ma.filled(value, np.nan), dtype="float64")


def read_time_data(path: Path) -> dict[pd.Timestamp, dict[str, Any]]:
    time_data: dict[pd.Timestamp, dict[str, Any]] = {}

    for _, ds in iter_datasets(path):
        time_name = "valid_time" if "valid_time" in ds.variables else "time"
        time_var = ds.variables[time_name]
        lats = as_array(ds.variables["latitude"][:])
        lons = as_array(ds.variables["longitude"][:])

        for variable in ERA5_VARIABLES:
            if variable not in ds.variables:
                continue
            arr = ds.variables[variable][:]
            for index, raw_time in enumerate(time_var[:]):
                ts = decode_time_value(time_var, raw_time)
                time_data.setdefault(ts, {"latitude": lats, "longitude": lons})
                if getattr(arr, "ndim", 0) >= 3:
                    time_data[ts][variable] = as_array(arr[index])
                else:
                    time_data[ts][variable] = as_array(arr)

    return time_data


def calculate_metrics(data: dict[str, Any], days_in_month: int) -> dict[str, np.ndarray]:
    metrics: dict[str, np.ndarray] = {}

    if "t2m" in data:
        # ERA5 temperature is Kelvin. Celsius is easier to read and compare.
        metrics["temperature_c"] = data["t2m"] - 273.15

    if "si10" in data:
        metrics["windspeed_10m_ms"] = data["si10"]
    elif "u10" in data and "v10" in data:
        metrics["windspeed_10m_ms"] = np.sqrt(data["u10"] ** 2 + data["v10"] ** 2)

    if "u100" in data and "v100" in data:
        metrics["windspeed_100m_ms"] = np.sqrt(data["u100"] ** 2 + data["v100"] ** 2)

    if "ssrd" in data:
        # Current monthly ERA5 script treats SSRD as daily energy and converts it to W/m2.
        metrics["solar_radiation_wm2"] = data["ssrd"] / 86400

    if "tp" in data:
        # Current monthly ERA5 script converts daily precipitation in meters to monthly mm.
        metrics["precipitation_mm"] = data["tp"] * 1000 * days_in_month

    return metrics


def to_grid_frame(time_data: dict[pd.Timestamp, dict[str, Any]], sample_step: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for ts, data in sorted(time_data.items()):
        days_in_month = pd.Timestamp(ts.year, ts.month, 1).days_in_month
        metrics = calculate_metrics(data, days_in_month)
        lats = data["latitude"]
        lons = data["longitude"]

        for lat_index in range(0, len(lats), sample_step):
            for lon_index in range(0, len(lons), sample_step):
                row = {
                    "year": ts.year,
                    "month": ts.month,
                    "latitude": float(lats[lat_index]),
                    "longitude": float(lons[lon_index]),
                }
                for name, values in metrics.items():
                    row[name] = float(values[lat_index, lon_index])
                rows.append(row)

    return pd.DataFrame(rows)


def to_region_frame(time_data: dict[pd.Timestamp, dict[str, Any]], masks_path: Path) -> pd.DataFrame:
    masks = np.load(masks_path, allow_pickle=True)["masks"].item()
    rows: list[dict[str, Any]] = []

    for ts, data in sorted(time_data.items()):
        days_in_month = pd.Timestamp(ts.year, ts.month, 1).days_in_month
        metrics = calculate_metrics(data, days_in_month)

        for region, raw_mask in masks.items():
            mask = np.asarray(raw_mask).astype(bool)
            row = {"year": ts.year, "month": ts.month, "region": str(region)}
            for name, values in metrics.items():
                if values.shape != mask.shape:
                    continue
                selected = values[mask]
                row[name] = float(np.nanmean(selected))
            rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ERA5 monthly NetCDF data.")
    parser.add_argument("input", type=Path, help="ERA5 NetCDF file, or CDS ZIP-style .nc file")
    parser.add_argument("--output", type=Path, default=Path("data/era5_cleaned.csv"))
    parser.add_argument("--province-masks", type=Path, help="Optional province_masks.npz file")
    parser.add_argument(
        "--grid-sample-step",
        type=int,
        default=1,
        help="When no mask is supplied, keep every Nth grid cell to reduce CSV size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_sample_step < 1:
        raise SystemExit("--grid-sample-step must be >= 1")

    time_data = read_time_data(args.input)
    if not time_data:
        raise SystemExit("No ERA5 variables were found in the input file.")

    if args.province_masks:
        output = to_region_frame(time_data, args.province_masks)
    else:
        output = to_grid_frame(time_data, args.grid_sample_step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
