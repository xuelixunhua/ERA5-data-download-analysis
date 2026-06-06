# -*- coding: utf-8 -*-
"""Inspect variables and dimensions in a NetCDF file.

The new CDS download may return a zip-like NetCDF package. This helper tries to
handle both ordinary NetCDF files and zip files containing .nc members.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a NetCDF file.")
    parser.add_argument("path", type=Path, help="Path to .nc file downloaded from CDS.")
    return parser.parse_args()


def print_dataset_summary(name: str, dataset) -> None:
    print(f"\n{name}")
    print("Dimensions:")
    for dim_name, dim in dataset.dimensions.items():
        print(f"  {dim_name}: {len(dim)}")

    print("Variables:")
    for var_name, var in dataset.variables.items():
        dims = ", ".join(var.dimensions)
        units = getattr(var, "units", "")
        long_name = getattr(var, "long_name", "")
        suffix = f" | {units}" if units else ""
        label = f" | {long_name}" if long_name else ""
        print(f"  {var_name}: ({dims}){suffix}{label}")


def inspect_path(path: Path) -> None:
    try:
        import netCDF4 as nc4
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install netCDF4") from exc

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".nc")]
            if not members:
                raise SystemExit("Zip file contains no .nc members.")
            for member in members:
                data = archive.read(member)
                dataset = nc4.Dataset(member, memory=data)
                try:
                    print_dataset_summary(member, dataset)
                finally:
                    dataset.close()
    else:
        dataset = nc4.Dataset(path)
        try:
            print_dataset_summary(str(path), dataset)
        finally:
            dataset.close()


def main() -> None:
    args = parse_args()
    inspect_path(args.path)


if __name__ == "__main__":
    main()
