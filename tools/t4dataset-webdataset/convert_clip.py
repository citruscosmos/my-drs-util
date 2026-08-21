#!/usr/bin/env python3
"""Convert one t4dataset (nuScenes-format) clip folder into a pair of WebDataset shards.

A clip folder looks like:
    <clip_id>/
        annotation/*.json   (nuScenes-style tables)
        data/<CHANNEL>/*.*  (camera jpgs, lidar pcd.bin, ...)
        fastlabel/*.json    (raw 2D fastlabel export, not used here)
        fastlabel_3d/*.json (raw 3D fastlabel export, not used here)

Output (written next to each other in --out-dir):
    sensor-<clip_id>.tar        raw sensor data, keyed by frame index, effectively write-once
    anno-<clip_id>.tar          2D/3D annotations, keyed by the same frame index, re-generated
                                 whenever annotations are revised
    index-<clip_id>.parquet     one row per frame: byte offset + size of every member in both
                                 tars, plus queryable columns (category names, box counts, ego
                                 location) so a frame can be fetched with a single seek()+read()
                                 instead of scanning the tar, and frames can be filtered without
                                 touching the tars at all
    index-<clip_id>.manifest.json  clip-level info: frame count, channel list, and the byte
                                 offset/size of each tar's __meta__.json member

Both tars use the same "00042.<channel>.<ext>" member naming per frame so a consumer
can open both shards and zip-iterate them by matching key. IMPORTANT: this produces
ONE sensor tar and ONE anno tar per clip (not one tar per frame) -- frames 00000..00295
are members inside those two tars.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd


def load_table(annotation_dir: Path, name: str):
    path = annotation_dir / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)


def index_by_token(records):
    return {r["token"]: r for r in records}


def group_by(records, key):
    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return groups


def resolve_attribute_names(attribute_tokens, attribute_by_token):
    return [attribute_by_token[t]["name"] for t in attribute_tokens if t in attribute_by_token]


def order_samples(samples, scene):
    """Walk the sample.next chain starting at scene.first_sample_token.

    Falls back to JSON array order (with a warning) if the chain is broken,
    so a malformed clip doesn't hard-fail the whole batch.
    """
    by_token = index_by_token(samples)
    ordered = []
    token = scene["first_sample_token"]
    seen = set()
    while token:
        if token in seen or token not in by_token:
            break
        seen.add(token)
        sample = by_token[token]
        ordered.append(sample)
        token = sample["next"]
    if len(ordered) != len(samples):
        print(
            f"  WARNING: next-chain walk found {len(ordered)} samples, "
            f"expected {len(samples)}; falling back to JSON array order",
            file=sys.stderr,
        )
        return list(samples)
    return ordered


def add_json_member(tar: tarfile.TarFile, name: str, obj, mtime: int) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(payload))


def add_bytes_member(tar: tarfile.TarFile, name: str, data: bytes, mtime: int) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = mtime
    tar.addfile(info, io.BytesIO(data))


def convert_clip(clip_dir: Path, out_dir: Path, mtime: int | None = None) -> tuple[Path, Path]:
    clip_id = clip_dir.name
    annotation_dir = clip_dir / "annotation"
    if mtime is None:
        mtime = int(time.time())

    scene = load_table(annotation_dir, "scene")[0]
    log = load_table(annotation_dir, "log")[0]
    map_ = load_table(annotation_dir, "map")[0]
    samples = load_table(annotation_dir, "sample")
    sample_data = load_table(annotation_dir, "sample_data")
    ego_pose = load_table(annotation_dir, "ego_pose")
    calibrated_sensor = load_table(annotation_dir, "calibrated_sensor")
    sensor = load_table(annotation_dir, "sensor")
    sample_annotation = load_table(annotation_dir, "sample_annotation")
    object_ann = load_table(annotation_dir, "object_ann")
    surface_ann = load_table(annotation_dir, "surface_ann")
    instance = load_table(annotation_dir, "instance")
    category = load_table(annotation_dir, "category")
    attribute = load_table(annotation_dir, "attribute")
    visibility = load_table(annotation_dir, "visibility")
    try:
        autolabel_model = json.loads((annotation_dir / "autolabel_model.json").read_text())
    except FileNotFoundError:
        autolabel_model = None

    sensor_by_token = index_by_token(sensor)
    category_by_token = index_by_token(category)
    attribute_by_token = index_by_token(attribute)
    instance_by_token = index_by_token(instance)
    ego_pose_by_token = index_by_token(ego_pose)

    calibrated_sensor_by_token = {}
    for cs in calibrated_sensor:
        s = sensor_by_token[cs["sensor_token"]]
        calibrated_sensor_by_token[cs["token"]] = {**cs, "channel": s["channel"], "modality": s["modality"]}

    sample_data_by_sample = group_by(sample_data, "sample_token")
    sample_annotation_by_sample = group_by(sample_annotation, "sample_token")
    object_ann_by_sample_data = group_by(object_ann, "sample_data_token")
    surface_ann_by_sample_data = group_by(surface_ann, "sample_data_token")

    ordered_samples = order_samples(samples, scene)

    out_dir.mkdir(parents=True, exist_ok=True)
    sensor_tar_path = out_dir / f"sensor-{clip_id}.tar"
    anno_tar_path = out_dir / f"anno-{clip_id}.tar"
    index_path = out_dir / f"index-{clip_id}.parquet"
    manifest_path = out_dir / f"index-{clip_id}.manifest.json"

    n_frames = len(ordered_samples)
    pad = max(5, len(str(max(n_frames - 1, 0))))
    channels = sorted(s["channel"] for s in sensor)
    ref_lidar_channel = next((s["channel"] for s in sensor if s["modality"] == "lidar"), channels[0] if channels else None)

    frame_rows = []

    with tarfile.open(sensor_tar_path, "w") as sensor_tar, tarfile.open(anno_tar_path, "w") as anno_tar:
        add_json_member(
            sensor_tar,
            "__meta__.json",
            {"scene": scene, "log": log, "map": map_, "sensors": sensor, "calibrated_sensors": calibrated_sensor},
            mtime,
        )
        add_json_member(
            anno_tar,
            "__meta__.json",
            {
                "categories": category,
                "attributes": attribute,
                "visibilities": visibility,
                "autolabel_model": autolabel_model,
            },
            mtime,
        )

        for i, sample in enumerate(ordered_samples):
            key = str(i).zfill(pad)
            sample_token = sample["token"]

            frame_meta = {"sample_token": sample_token, "scene_token": sample["scene_token"], "timestamp": sample["timestamp"], "channels": {}}
            ann2d = {}
            row = {"clip_id": clip_id, "frame_index": i, "key": key, "sample_token": sample_token, "timestamp": sample["timestamp"]}
            for ch in channels:
                row[f"{ch.lower()}_member"] = None
            ego_geocoordinate = None

            for sd in sample_data_by_sample.get(sample_token, []):
                cs = calibrated_sensor_by_token[sd["calibrated_sensor_token"]]
                channel = cs["channel"]
                channel_key = channel.lower()
                ext = sd["fileformat"]

                src_path = clip_dir / sd["filename"]
                data = src_path.read_bytes()
                member_name = f"{key}.{channel_key}.{ext}"
                add_bytes_member(sensor_tar, member_name, data, mtime)
                row[f"{channel_key}_member"] = member_name

                pose = ego_pose_by_token.get(sd["ego_pose_token"])
                if channel == ref_lidar_channel and pose is not None:
                    ego_geocoordinate = pose.get("geocoordinate")
                frame_meta["channels"][channel] = {
                    "member": member_name,
                    "timestamp": sd["timestamp"],
                    "is_key_frame": sd["is_key_frame"],
                    "width": sd["width"],
                    "height": sd["height"],
                    "calibrated_sensor": {
                        "translation": cs["translation"],
                        "rotation": cs["rotation"],
                        "camera_intrinsic": cs.get("camera_intrinsic", []),
                        "camera_distortion": cs.get("camera_distortion", []),
                    },
                    "ego_pose": pose,
                }

                if cs["modality"] == "camera":
                    boxes = []
                    for oa in object_ann_by_sample_data.get(sd["token"], []):
                        boxes.append(
                            {
                                **oa,
                                "category_name": category_by_token.get(oa["category_token"], {}).get("name"),
                                "attribute_names": resolve_attribute_names(oa["attribute_tokens"], attribute_by_token),
                                "instance_name": instance_by_token.get(oa["instance_token"], {}).get("instance_name"),
                            }
                        )
                    masks = []
                    for sa in surface_ann_by_sample_data.get(sd["token"], []):
                        masks.append(
                            {
                                **sa,
                                "category_name": category_by_token.get(sa["category_token"], {}).get("name"),
                                "attribute_names": resolve_attribute_names(sa["attribute_tokens"], attribute_by_token),
                            }
                        )
                    if boxes or masks:
                        ann2d[channel] = {"boxes": boxes, "masks": masks}

            meta_member = f"{key}.meta.json"
            add_json_member(sensor_tar, meta_member, frame_meta, mtime)

            ann3d = []
            for sa in sample_annotation_by_sample.get(sample_token, []):
                inst = instance_by_token.get(sa["instance_token"], {})
                ann3d.append(
                    {
                        **sa,
                        "category_name": category_by_token.get(inst.get("category_token"), {}).get("name"),
                        "attribute_names": resolve_attribute_names(sa["attribute_tokens"], attribute_by_token),
                        "instance_name": inst.get("instance_name"),
                    }
                )
            ann3d_member = f"{key}.ann3d.json"
            ann2d_member = f"{key}.ann2d.json"
            add_json_member(anno_tar, ann3d_member, ann3d, mtime)
            add_json_member(anno_tar, ann2d_member, ann2d, mtime)

            num_ann2d_boxes = sum(len(v["boxes"]) for v in ann2d.values())
            num_ann2d_masks = sum(len(v["masks"]) for v in ann2d.values())
            ann2d_categories = sorted({b["category_name"] for v in ann2d.values() for b in v["boxes"] if b["category_name"]} | {m["category_name"] for v in ann2d.values() for m in v["masks"] if m["category_name"]})

            row.update(
                {
                    "meta_member": meta_member,
                    "ann3d_member": ann3d_member,
                    "ann2d_member": ann2d_member,
                    "num_ann3d": len(ann3d),
                    "ann3d_categories": sorted({a["category_name"] for a in ann3d if a["category_name"]}),
                    "num_ann2d_boxes": num_ann2d_boxes,
                    "num_ann2d_masks": num_ann2d_masks,
                    "ann2d_categories": ann2d_categories,
                    "ego_lat": ego_geocoordinate[0] if ego_geocoordinate else None,
                    "ego_lon": ego_geocoordinate[1] if ego_geocoordinate else None,
                    "ego_alt": ego_geocoordinate[2] if ego_geocoordinate else None,
                }
            )
            frame_rows.append(row)

    # Reopen the tars read-only to read back the byte offset of every member. tarfile does not
    # reliably expose offset_data on the TarInfo passed into addfile() during writing, but it is
    # always correct once the archive is parsed back -- and parsing headers-only is cheap (no
    # payload bytes are read for members we skip over).
    def collect_offsets(path: Path) -> dict[str, tuple[int, int]]:
        offsets = {}
        with tarfile.open(path, "r") as t:
            for m in t.getmembers():
                offsets[m.name] = (m.offset_data, m.size)
        return offsets

    sensor_offsets = collect_offsets(sensor_tar_path)
    anno_offsets = collect_offsets(anno_tar_path)

    for row in frame_rows:
        meta_off, meta_size = sensor_offsets[row.pop("meta_member")]
        row["meta_offset"], row["meta_size"] = meta_off, meta_size
        for ch in channels:
            member_key = f"{ch.lower()}_member"
            member_name = row.pop(member_key)
            if member_name is not None:
                off, size = sensor_offsets[member_name]
                row[f"{ch.lower()}_offset"], row[f"{ch.lower()}_size"] = off, size
            else:
                row[f"{ch.lower()}_offset"], row[f"{ch.lower()}_size"] = None, None
        ann3d_off, ann3d_size = anno_offsets[row.pop("ann3d_member")]
        row["ann3d_offset"], row["ann3d_size"] = ann3d_off, ann3d_size
        ann2d_off, ann2d_size = anno_offsets[row.pop("ann2d_member")]
        row["ann2d_offset"], row["ann2d_size"] = ann2d_off, ann2d_size

    df = pd.DataFrame(frame_rows)
    df.insert(0, "sensor_shard", sensor_tar_path.name)
    df.insert(1, "anno_shard", anno_tar_path.name)
    df.to_parquet(index_path, engine="pyarrow", index=False)

    sensor_meta_off, sensor_meta_size = sensor_offsets["__meta__.json"]
    anno_meta_off, anno_meta_size = anno_offsets["__meta__.json"]
    manifest = {
        "clip_id": clip_id,
        "n_frames": n_frames,
        "channels": channels,
        "sensor_shard": sensor_tar_path.name,
        "anno_shard": anno_tar_path.name,
        "sensor_meta_offset": sensor_meta_off,
        "sensor_meta_size": sensor_meta_size,
        "anno_meta_offset": anno_meta_off,
        "anno_meta_size": anno_meta_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return sensor_tar_path, anno_tar_path, index_path, manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip_dir", type=Path, help="Path to a single t4dataset clip folder (e.g. .../<name>_0)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Where to write the shard tars (default: alongside clip_dir)")
    args = parser.parse_args()

    clip_dir = args.clip_dir.resolve()
    if not (clip_dir / "annotation").is_dir():
        parser.error(f"{clip_dir} does not look like a t4dataset clip folder (no annotation/ subdir)")

    out_dir = args.out_dir.resolve() if args.out_dir else clip_dir.parent
    sensor_tar, anno_tar, index_path, manifest_path = convert_clip(clip_dir, out_dir)
    print(f"wrote {sensor_tar}")
    print(f"wrote {anno_tar}")
    print(f"wrote {index_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
