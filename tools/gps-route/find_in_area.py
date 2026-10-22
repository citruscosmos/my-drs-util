#!/usr/bin/env python3
"""List recordings whose *_maneuvers.csv has a point inside a lat/lon area.

Searches <root> recursively for *_maneuvers.csv (as written by plot_gps_route.py
--tag-maneuvers) and prints the recording name (the `*` part) of every file with
at least one row inside the area. The area is either a bounding box
(--lat-min/--lat-max/--lon-min/--lon-max) or a circle (--center-lat/--center-lon/--radius-m).

Note: a maneuvers.csv only has rows where a maneuver was detected, so a recording
that passed through the area without any maneuver there is not found.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

SUFFIX = "_maneuvers.csv"
WGS84_R = 6378137.0


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * WGS84_R * math.asin(math.sqrt(a))


def make_predicate(args):
    if args.radius_m is not None:
        return lambda lat, lon: haversine_m(args.center_lat, args.center_lon, lat, lon) <= args.radius_m
    return lambda lat, lon: (args.lat_min <= lat <= args.lat_max
                             and args.lon_min <= lon <= args.lon_max)


def scan_file(path: Path, inside):
    """Return rows (as dicts) of `path` that fall inside the area."""
    hits = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
            except (KeyError, ValueError):
                continue
            if inside(lat, lon):
                hits.append(row)
    return hits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path,
                    help="directory to search recursively, e.g. /mnt/DataPool/SharedData/raw/<group>/gps-route")
    box = ap.add_argument_group("bounding box")
    box.add_argument("--lat-min", type=float)
    box.add_argument("--lat-max", type=float)
    box.add_argument("--lon-min", type=float)
    box.add_argument("--lon-max", type=float)
    circ = ap.add_argument_group("circle")
    circ.add_argument("--center-lat", type=float)
    circ.add_argument("--center-lon", type=float)
    circ.add_argument("--radius-m", type=float)
    ap.add_argument("--details", action="store_true",
                    help="also print count and maneuver types of the in-area rows")
    ap.add_argument("--csv", type=Path, help="also write the result list to this CSV")
    args = ap.parse_args(argv)

    box_given = [v is not None for v in (args.lat_min, args.lat_max, args.lon_min, args.lon_max)]
    circ_given = [v is not None for v in (args.center_lat, args.center_lon, args.radius_m)]
    if all(box_given) == all(circ_given) or (any(box_given) and not all(box_given)) \
            or (any(circ_given) and not all(circ_given)):
        ap.error("specify either all of --lat-min/--lat-max/--lon-min/--lon-max, "
                 "or all of --center-lat/--center-lon/--radius-m")
    if all(box_given) and (args.lat_min > args.lat_max or args.lon_min > args.lon_max):
        ap.error("min must be <= max")

    inside = make_predicate(args)
    files = sorted(args.root.rglob(f"*{SUFFIX}"))
    if not files:
        print(f"no *{SUFFIX} under {args.root}", file=sys.stderr)
        return 1

    results = []
    for p in files:
        hits = scan_file(p, inside)
        if hits:
            types = {}
            for r in hits:
                types[r["maneuver_type"]] = types.get(r["maneuver_type"], 0) + 1
            results.append((p.name[:-len(SUFFIX)], p.parent, len(hits), types))

    for name, parent, n, types in results:
        if args.details:
            print(f"{name}\t{n}\t{','.join(f'{k}:{v}' for k, v in sorted(types.items()))}\t{parent}")
        else:
            print(name)
    print(f"# {len(results)} / {len(files)} recordings have maneuver points in the area",
          file=sys.stderr)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["recording", "n_points", "maneuver_types", "dir"])
            for name, parent, n, types in results:
                w.writerow([name, n, ";".join(f"{k}:{v}" for k, v in sorted(types.items())), parent])
    return 0


if __name__ == "__main__":
    sys.exit(main())
