# -*- coding: utf-8 -*-
"""Download ERA5 monthly averaged single-level data from CDS.

Example:
    python download_era5_monthly.py --years 2024 --months 1 2 3
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


DEFAULT_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "total_precipitation",
]

# ERA5 API uses [north, west, south, east].
DEFAULT_CHINA_BBOX = [55, 70, 15, 135]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5 monthly averaged data from Copernicus CDS."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Years to download, for example: --years 2024 2025",
    )
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        default=list(range(1, 13)),
        help="Months to download, default is 1-12.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="ERA5 variable names. Defaults cover temperature, wind, radiation, precipitation.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        default=DEFAULT_CHINA_BBOX,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        help="Download area in CDS order: north west south east.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/era5_china_monthly.nc"),
        help="Output NetCDF path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def download_era5(
    years: list[int],
    months: list[int],
    variables: list[str],
    area: list[float],
    output: Path,
    overwrite: bool = False,
) -> Path:
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install cdsapi") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        print(f"File already exists, skip download: {output}")
        print("Use --overwrite if you want to download again.")
        return output

    request = {
        "product_type": "monthly_averaged_reanalysis",
        "variable": variables,
        "year": [str(year) for year in sorted(set(years))],
        "month": [f"{month:02d}" for month in sorted(set(months))],
        "time": "00:00",
        "area": area,
        "format": "netcdf",
    }

    print("Download request")
    print(f"  dataset: reanalysis-era5-single-levels-monthly-means")
    print(f"  years:   {request['year']}")
    print(f"  months:  {request['month']}")
    print(f"  vars:    {variables}")
    print(f"  area:    {area}")
    print(f"  output:  {output}")
    print(f"  start:   {datetime.now():%Y-%m-%d %H:%M:%S}")

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        request,
        str(output),
    )

    size_mb = output.stat().st_size / 1024 / 1024
    print(f"Done: {output} ({size_mb:.1f} MB)")
    return output


def main() -> None:
    args = parse_args()
    download_era5(
        years=args.years,
        months=args.months,
        variables=args.variables,
        area=args.area,
        output=args.output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
