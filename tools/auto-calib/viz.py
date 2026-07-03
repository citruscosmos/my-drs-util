#!/usr/bin/env python3
"""Shared logging/visualization helpers for Steps 0-4.

Every step's ``main()`` wraps its body in ``tee_log(out_dir, "stepN_log.txt")``
so the full stdout trace (including prints from deep inside helper functions,
e.g. Step 4's per-iteration convergence lines) is preserved on disk for later
debugging, not just whatever summary line happens to be printed last.

Plot helpers use matplotlib's Agg backend (headless-safe) and silently
subsample point clouds above ``viz_max_points`` so large scans/maps don't
blow up PNG render time. All plot functions return the path they wrote so
callers can log it.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np


class _Tee:
    def __init__(self, path, stream):
        self.file = open(path, "w")
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


@contextlib.contextmanager
def tee_log(out_dir, filename):
    """Duplicate stdout to <out_dir>/<filename> for the duration of the block."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    tee = _Tee(path, sys.stdout)
    old = sys.stdout
    sys.stdout = tee
    try:
        yield path
    finally:
        sys.stdout = old
        tee.close()


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_fig(fig, path, dpi=110):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def subsample_idx(n, cap, rng=None):
    if n <= cap:
        return None  # caller keeps the full array
    rng = rng or np.random.default_rng(0)
    return rng.choice(n, cap, replace=False)


def subsample(arr, cap, rng=None):
    idx = subsample_idx(len(arr), cap, rng)
    return arr if idx is None else arr[idx]


def topdown_scatter(path, points_list, cfg=None, title="", xlabel="x [m]", ylabel="y [m]",
                     equal_aspect=True, extra=None, colorbar_label=None):
    """points_list: list of (xy (N,2), kwargs-dict for ax.scatter).

    If kwargs['c'] is a per-point array (same length as xy), it is subsampled
    together with xy so a downsampled scatter never desyncs point/color counts.
    Pass colorbar_label to add a colorbar for the last array-colored series
    (e.g. a displacement-magnitude heatmap).

    extra: optional callable(ax) for annotations (start/end markers etc.).
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 7))
    cap = int((cfg or {}).get("viz_max_points", 150000))
    rng = np.random.default_rng(0)
    mappable = None
    for xy, kwargs in points_list:
        if len(xy) == 0:
            continue
        xy = np.asarray(xy)
        kwargs = dict(kwargs)
        idx = subsample_idx(len(xy), cap, rng)
        if idx is not None:
            c = kwargs.get("c")
            if isinstance(c, (np.ndarray, list)) and len(c) == len(xy):
                kwargs["c"] = np.asarray(c)[idx]
            xy = xy[idx]
        sc = ax.scatter(xy[:, 0], xy[:, 1], **kwargs)
        if isinstance(kwargs.get("c"), (np.ndarray, list)):
            mappable = sc
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    if extra:
        extra(ax)
    if any(k.get("label") for _, k in points_list):
        ax.legend(markerscale=10, loc="best")
    if colorbar_label is not None and mappable is not None:
        fig.colorbar(mappable, ax=ax, label=colorbar_label, shrink=0.8)
    save_fig(fig, path, dpi=int((cfg or {}).get("viz_dpi", 110)))
    return path


def line_plot(path, series, cfg=None, title="", xlabel="", ylabel="", hlines=None, vlines=None):
    """series: list of (x, y, kwargs-dict for ax.plot)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4))
    for x, y, kwargs in series:
        ax.plot(x, y, **kwargs)
    for y, kwargs in (hlines or []):
        ax.axhline(y, **kwargs)
    for x, kwargs in (vlines or []):
        ax.axvline(x, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if any(k.get("label") for _, _, k in series):
        ax.legend(loc="best")
    save_fig(fig, path, dpi=int((cfg or {}).get("viz_dpi", 110)))
    return path


def hist_plot(path, values, cfg=None, bins=50, title="", xlabel="", ylabel="count",
              vlines=None, log_y=False):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    values = np.asarray(values)
    if len(values):
        ax.hist(values, bins=bins, color="steelblue", edgecolor="none")
    if log_y:
        ax.set_yscale("log")
    for x, kwargs in (vlines or []):
        ax.axvline(x, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    save_fig(fig, path, dpi=int((cfg or {}).get("viz_dpi", 110)))
    return path


def hist_compare_plot(path, before, after, cfg=None, bins=60, title="", xlabel="",
                       labels=("before", "after")):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    before, after = np.asarray(before), np.asarray(after)
    if len(before):
        ax.hist(before, bins=bins, alpha=0.5, label=labels[0], color="tomato")
    if len(after):
        ax.hist(after, bins=bins, alpha=0.5, label=labels[1], color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(fig, path, dpi=int((cfg or {}).get("viz_dpi", 110)))
    return path


def bar_plot(path, labels, values, cfg=None, title="", ylabel="", hline=None, log_y=False):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(l) for l in labels], values, color="steelblue")
    if log_y:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="red", linestyle="--", linewidth=1, label=f"threshold {hline:.2g}")
        ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, path, dpi=int((cfg or {}).get("viz_dpi", 110)))
    return path
