"""

Classical (non-deep-learning) left-ventricle segmentation and cardiac-function
analysis for the ACDC short-axis cine-MRI dataset.

"""
from __future__ import annotations


#
# What this file does:
#
# 1. Read one patient's cine-MRI (a 4D scan: 2D image x slices x time frames)
#    and the metadata that says which group they are (NOR/DCM/HCM/...) and
#    which frames are end-diastole (ED, heart full) and end-systole (ES,
#    heart contracted).                                            [io]
# 2. Ask the user to click once inside the LV blood pool, on ED and on ES.
#    That click is the "seed".                                     [seed]
# 3. From the seed, grow a segmentation of the bright blood pool on that
#    slice, then propagate it up and down through the other slices, and
#    across all time frames.                          [segment_blood, propagate]
# 4. Grow the muscle ring (myocardium) just outside the blood pool. [segment_myo]
# 5. Turn the per-frame blood volumes into clinical numbers:
#    EF (ejection fraction), EDV-index, wall thickness.   [indices, wall_thickness]
# 6. Compare those numbers against the textbook criteria for the patient's
#    known group (this is a CONSISTENCY CHECK, not a classifier). [classify_correlate]
# 7. Compare the segmentation to the expert ground truth (Dice/Jaccard) and
#    check whether the whole ventricle was imaged.        [validate, coverage]
# 8. Save overlays, a volume-vs-time curve, and a summary CSV.       [plots, run_]
#



from dataclasses import dataclass, field
from pathlib import Path
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, closing, disk
from skimage.morphology import skeletonize
import argparse
import csv
import json
import nibabel as nib
import numpy as np
import re


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Dataset Label Convention
# ---------------------------------------------------------------------------
LABEL_BACKGROUND = 0
LABEL_RV = 1          # right-ventricle cavity
LABEL_MYO = 2         # left-ventricle myocardium (the muscle ring)
LABEL_LV = 3          # left-ventricle blood pool (the bright cavity)

# ---------------------------------------------------------------------------
# Paths 
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    dataset_folder: Path = Path(
        r"PATH\TO\database\training"
    )
    output_folder: Path = Path(
        r"PATH\TO\output\folder"
    )

    def __post_init__(self):
        if self.output_folder is None:
            self.output_folder = self.dataset_folder / "lv_segmentation_outputs"


# The 12 subjects to analyze
DEFAULT_PATIENTS = [
    # Cohort actually analysed: 4 DCM / 5 NOR / 3 HCM, balanced across classes.
    # Dual-seed (separate ED and ES clicks) is used per patient.
    "patient001", "patient004", "patient012", "patient019",                 # DCM
    "patient067", "patient069", "patient070", "patient072", "patient075",   # NOR
    "patient021", "patient031", "patient040",                               # HCM
]


# ---------------------------------------------------------------------------
# Segmentation parameters (blood pool / endocardium)
# ---------------------------------------------------------------------------
@dataclass
class SegmentationParams:
    # Circular ROI radius (mm) around the propagated center, restricts the
    # search so bright structures far from the LV cannot be picked up.
    roi_radius_mm: float = 45.0
    # Intensity percentiles for contrast normalization (robust min/max).
    norm_low_pct: float = 1.0
    norm_high_pct: float = 99.0
    # Minimum blob area (pixels) to survive cleanup on a slice.
    min_blob_px: int = 20
    # Morphological closing radius (pixels) to smooth the cavity.
    close_radius_px: int = 2
    # Region-grow lower bound as a fraction of the seed/cavity intensity:
    # pixels brighter than (blood_seed_frac * seed_intensity) and connected to
    # the seed are accepted as blood. Lower = fills more of the cavity; too low
    # leaks through the dark endocardial border into muscle. ~0.6 fills the
    # cavity to the myocardial border without crossing it.
    blood_seed_frac: float = 0.55
    #   blood_seed_frac / fill_cap_frac - how bright a pixel must be to count as
    #   blood (controls under- vs over-filling the cavity).
    # Upper cap on the blood threshold, as a fraction of seed intensity. The
    # Otsu split may raise the threshold to block leaks into bright muscle, but
    # never above this fraction of the seed, or it clips the dim end-systolic
    # cavity rim (severe ES under-segmentation). A per-patient GT threshold
    # sweep (NOR + HCM, ED + ES) found Dice peaks at ~0.55*seed with the
    # leak/merge cliff below it, so 0.55 fills the cavity with margin to spare.
    fill_cap_frac: float = 0.55
    # Minimum normalised intensity at the seed for a slice to be considered to
    # contain a cavity at all (guards apical/basal slices and bad seeds).
    min_seed_intensity: float = 0.30


# ---------------------------------------------------------------------------
# Myocardium / epicardium parameters
# ---------------------------------------------------------------------------
@dataclass
class MyocardiumParams:
    # Outward search ceiling (mm) for the myocardium shell. Even severe HCM
    # rarely exceeds ~18-20 mm, so this bounds the wall search.
    max_wall_mm: float = 20.0
    # A ring (shell at increasing radius) is accepted as still-muscle while at
    # least this fraction of it falls in the muscle-intensity window. When the
    # ring drops below this, we have reached the epicardial border and stop.
    shell_min_frac: float = 0.30
    # Lower fraction of the cavity-mean intensity that still counts as muscle
    # (myocardium is mid-grey: darker than blood, brighter than lung/fat void).
    myo_low_frac: float = 0.15
    myo_high_frac: float = 0.85
    # Smoothing of the epicardial contour (closing radius, pixels).
    epi_close_px: int = 3


# ---------------------------------------------------------------------------
# Temporal/spatial propagation control
# ---------------------------------------------------------------------------
@dataclass
class PropagationParams:
    # This entire section essentially guards in
    # the slice-to-slice propagation that stop the region leaking into other
    # tissue or collapsing to nothing.
    # Stop propagating to neighbouring slices if the area jumps/collapses
    # beyond these multiples of the previous slice's area.
    area_grow_max: float = 2.8
    area_shrink_min: float = 0.08
    # Number of consecutive shrinking slices required before the taper guard
    # locks (after which re-expansion stops propagation). A single noisy dip
    # near the base/outflow must not latch the guard, or a seed placed off the
    # mid-cavity slice (e.g. an ES seed near the base) terminates prematurely
    # and captures only a sliver of the cavity. 2 tolerates a one-slice dip
    # while still catching a genuine multi-slice apex/base taper.
    taper_min_run: int = 2
    # Seed-relative leak cap: reject any slice whose area exceeds this multiple
    # of the seed-slice area. The per-slice area_grow_max only compares to the
    # immediately previous slice, so a gradual multi-slice leak (each step under
    # area_grow_max) slips through; and a single slice that balloons relative to
    # the seed is a leak into adjacent tissue or a truncated base/apex, not a
    # real cross-section. The true LV cross-section never exceeds a few times
    # the seed-slice area.
    seed_area_cap_mult: float = 2.8
    # Adaptive retry ladder: when a slice leaks past the grow cap under the
    # global fill threshold, re-segment it at blood_seed_frac raised in steps of
    # `retry_frac_step` up to `retry_frac_ceiling`, accepting the first setting
    # whose area fits the cap. This recovers a leaked neighbour instead of
    # abandoning propagation.
    # The ceiling bounds the search; if nothing fits by then the slice is a true
    # cavity boundary and propagation stops.
    retry_frac_step: float = 0.05
    retry_frac_ceiling: float = 0.75
    # Absolute minimum slice area (pixels) before we declare "out of ventricle".
    min_slice_px: int = 20
    # For the temporal curve: a per-frame total volume is "implausible" if it
    # deviates from the local (neighbour) median by more than this fraction.
    # Such frames are linearly interpolated and logged (flag-and-interpolate).
    frame_outlier_frac: float = 0.40
    # A frame whose volume collapses below this fraction of the cycle median is
    # treated as a segmentation dropout (lost cavity), not a real end-systole,
    # and is interpolated. Prevents spuriously high EF from a near-zero ESV.
    dropout_frac: float = 0.45
    # A frame whose volume balloons above this multiple of a leak-resistant
    # baseline (25th percentile of positive frame volumes) is treated as
    # segmentation leakage (region grew out of the cavity into adjacent bright
    # anatomy), not a real end-diastole, and is interpolated. Catches contiguous
    # leak runs that the neighbour-median test misses. A real LV roughly doubles
    # from ES to ED, so ~2.2x the low-quartile baseline is a safe ceiling.
    blowup_mult: float = 2.2


# ---------------------------------------------------------------------------
# Clinical thresholds for class correlation (from the project slides)
#   NOR: EF > 50%,                         diastolic wall < 12 mm
#   DCM: EF < 40%, LVEDV index > 100 mL/m2, diastolic wall < 12 mm
#   HCM: EF > 55%,                         diastolic wall > 15 mm
# ---------------------------------------------------------------------------
@dataclass
class ClinicalThresholds:
    nor_ef_min: float = 50.0
    nor_wall_max_mm: float = 12.0
    dcm_ef_max: float = 40.0
    dcm_edv_index_min: float = 100.0   # mL/m^2
    dcm_wall_max_mm: float = 12.0
    hcm_ef_min: float = 55.0
    hcm_wall_min_mm: float = 15.0


@dataclass
class Config:
    paths: Paths = field(default_factory=Paths)
    seg: SegmentationParams = field(default_factory=SegmentationParams)
    myo: MyocardiumParams = field(default_factory=MyocardiumParams)
    prop: PropagationParams = field(default_factory=PropagationParams)
    clinical: ClinicalThresholds = field(default_factory=ClinicalThresholds)
    patients: list = field(default_factory=lambda: list(DEFAULT_PATIENTS))
    # If True, the seed for each patient is chosen interactively (one click on
    # the ED mid-slice). If False, an automatic centre-of-brightness seed is
    # used.
    interactive_seed: bool = True


DEFAULT = Config()


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
#
# normalize_image(): rescales a raw MRI slice so its
# intensities run 0..1, clipping at the 1st/99th percentile so a few very
# bright/dark stray pixels don't squash the contrast. Every other step expects
# images on this common 0..1 scale, so this is called everywhere.

def normalize_image(image: np.ndarray, low_pct: float = 1.0,
                    high_pct: float = 99.0) -> np.ndarray:
    """
    Robust contrast normalisation to [0, 1] using intensity percentiles.

    Clipping at the 1st/99th percentiles (rather than min/max) prevents a few
    bright or dark outlier voxels from compressing the useful dynamic range.
    """
    image = np.asarray(image, dtype=np.float64)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image)
    lo = np.percentile(finite, low_pct)
    hi = np.percentile(finite, high_pct)
    if hi <= lo:
        return np.zeros_like(image)
    out = (image - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
#
# The scoring metrics. dice() and jaccard() measure how much the segmentation
# overlaps the expert ground truth (1.0 = perfect overlap, 0 = none). These are
# quality checks reported in the .csv file (dice_lv_ed, etc.); they don't affect the
# segmentation itself. Only the ED and ES frames have ground truth, so that's
# all that gets scored.

def dice(pred, gt) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(pred, gt).sum() / denom)


def jaccard(pred, gt) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def evaluate(pred_mask, gt_mask):
    """Return {'dice':..., 'jaccard':...} for a predicted vs GT mask."""
    return {"dice": dice(pred_mask, gt_mask), "jaccard": jaccard(pred_mask, gt_mask)}


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------
#
# Checks whether the whole ventricle was actually imaged. Ejection fraction is
# only meaningful if the scan covers the LV from base to apex. It walks the
# slices and, if the LV area is still large at the very first or last slice,
# flags the ventricle as "truncated" at that end.


@dataclass
class CoverageReport:
    full_coverage: bool
    reason: str
    apex_closed: bool
    base_closed: bool
    n_lv_slices: int


def _slice_areas(label_volume, target_label):
    return np.array([
        int((label_volume[:, :, z] == target_label).sum())
        for z in range(label_volume.shape[2])
    ])


def assess_coverage(label_volume, target_label=LABEL_LV,
                    edge_frac=0.15) -> CoverageReport:
    """
    Judge whether the LV is fully imaged from a labeled volume.

    edge_frac: a slice end is considered "closed" if its LV area is below
    edge_frac of the maximum LV slice area.
    """
    areas = _slice_areas(label_volume, target_label)
    n_lv = int((areas > 0).sum())
    if areas.max() == 0:
        return CoverageReport(False, "no LV label present", False, False, 0)

    peak = areas.max()
    base_closed = bool(areas[0] < edge_frac * peak)    # first slice ~ base
    apex_closed = bool(areas[-1] < edge_frac * peak)   # last slice ~ apex

    full = bool(base_closed and apex_closed)
    if full:
        reason = "ventricle tapers to <edge at both ends (fully imaged)"
    else:
        ends = []
        if not base_closed:
            ends.append("base")
        if not apex_closed:
            ends.append("apex")
        reason = f"LV truncated at {' and '.join(ends)} slice(s)"
    return CoverageReport(full, reason, apex_closed, base_closed, n_lv)


# ---------------------------------------------------------------------------
# Indices
# ---------------------------------------------------------------------------
#
# Turns the per-frame volume curve into the headline clinical numbers:
#   LVEDV = biggest volume (end-diastole), LVESV = smallest (end-systole),
#   stroke volume = EDV - ESV, ejection fraction EF = 100*SV/EDV,
#   EDV-index = EDV / body-surface-area.
# It can read EDV/ESV at the known ED/ES frames (from the metadata) rather than
# trusting the curve's raw max/min, and it guards against an accidentally
# swapped pair (EDV must be the larger).


@dataclass
class CardiacIndices:
    lvedv_ml: float
    lvesv_ml: float
    lvsv_ml: float
    lvef_pct: float
    ed_frame_idx: int    # 0-based frame index of max volume
    es_frame_idx: int    # 0-based frame index of min volume
    edv_index_ml_m2: float = float("nan")


def compute_indices(volumes_ml, bsa_m2: float = float("nan"),
                    ed_frame_idx: int = None,
                    es_frame_idx: int = None) -> CardiacIndices:
    """
    Derive global LV indices from a per-frame volume curve.

    If ed_frame_idx / es_frame_idx are given (0-based indices of the known
    end-diastole / end-systole frames, e.g. from Info.cfg), EDV and ESV are
    read at those frames. This anchors EF to the clinically-defined phases
    rather than to noisy curve extrema. When not given, the maximum and
    minimum of the (smoothed) curve are used as a fallback.
    """
    v = np.asarray(volumes_ml, dtype=float)
    if v.size == 0 or np.all(v == 0):
        return CardiacIndices(0, 0, 0, float("nan"), -1, -1)

    n = v.size
    if ed_frame_idx is not None and 0 <= ed_frame_idx < n:
        ed_idx = int(ed_frame_idx)
    else:
        ed_idx = int(np.argmax(v))
    if es_frame_idx is not None and 0 <= es_frame_idx < n:
        es_idx = int(es_frame_idx)
    else:
        es_idx = int(np.argmin(v))

    edv = float(v[ed_idx])
    esv = float(v[es_idx])
    # Guard against a swapped pair (e.g. if the curve disagrees with the
    # labeled phases): EDV must be the larger of the two.
    if esv > edv:
        edv, esv = esv, edv
        ed_idx, es_idx = es_idx, ed_idx
    sv = edv - esv
    ef = 100.0 * sv / edv if edv > 0 else float("nan")
    edv_index = edv / bsa_m2 if bsa_m2 and not np.isnan(bsa_m2) else float("nan")

    return CardiacIndices(
        lvedv_ml=edv, lvesv_ml=esv, lvsv_ml=sv, lvef_pct=ef,
        ed_frame_idx=ed_idx, es_frame_idx=es_idx,
        edv_index_ml_m2=edv_index,
    )


# ---------------------------------------------------------------------------
# Classify_correlate
# ---------------------------------------------------------------------------
#
# The is just a consistency check - not a classifier. Given the patient's known group, 
# it checks whether our measured EF / EDV-index / wall thickness fall in the textbook 
# range for that group (e.g. NOR: EF>50% and wall<12mm). It returns which criteria were
# checked, which were met, and whether all of them agree. Groups without
# defined criteria (e.g. RV) return "no criteria" rather than a forced verdict.

@dataclass
class CorrelationResult:
    group: str
    ef_pct: float
    wall_mm: float
    edv_index: float
    criteria_checked: list      # human-readable criterion strings
    criteria_met: list          # bool per criterion
    agrees: bool                # all checkable criteria met
    note: str = ""


def correlate(group: str, ef_pct: float, wall_mm: float, edv_index: float,
              th: ClinicalThresholds = None) -> CorrelationResult:
    th = th or ClinicalThresholds()
    g = (group or "").upper()
    checks, met = [], []

    def add(desc, ok):
        checks.append(desc)
        met.append(bool(ok))

    if g == "NOR":
        add(f"EF > {th.nor_ef_min}%", ef_pct > th.nor_ef_min)
        if not np.isnan(wall_mm):
            add(f"diastolic wall < {th.nor_wall_max_mm} mm", wall_mm < th.nor_wall_max_mm)
    elif g == "DCM":
        add(f"EF < {th.dcm_ef_max}%", ef_pct < th.dcm_ef_max)
        if not np.isnan(edv_index):
            add(f"EDV index > {th.dcm_edv_index_min} mL/m^2",
                edv_index > th.dcm_edv_index_min)
        if not np.isnan(wall_mm):
            add(f"diastolic wall < {th.dcm_wall_max_mm} mm", wall_mm < th.dcm_wall_max_mm)
    elif g == "HCM":
        add(f"EF > {th.hcm_ef_min}%", ef_pct > th.hcm_ef_min)
        if not np.isnan(wall_mm):
            add(f"diastolic wall > {th.hcm_wall_min_mm} mm", wall_mm > th.hcm_wall_min_mm)
    else:
        return CorrelationResult(
            group=g, ef_pct=ef_pct, wall_mm=wall_mm, edv_index=edv_index,
            criteria_checked=[], criteria_met=[], agrees=False,
            note=f"no slide criteria defined for group '{g}'",
        )

    agrees = all(met) if met else False
    return CorrelationResult(
        group=g, ef_pct=ef_pct, wall_mm=wall_mm, edv_index=edv_index,
        criteria_checked=checks, criteria_met=met, agrees=agrees,
    )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
#
# Decides the starting point ("seed"/click) for segmentation - one click inside the
# LV blood pool. Three ways to get it:
#   - interactive_seed: shows the slice and captures one mouse click (the real,
#     grade-relevant method).
#   - auto_seed: automatically guesses the brightest central blob (debug only).
#   - choose_seed: dispatches to whichever mode is selected.
# A Seed is just (row, col, slice). Everything downstream grows from it.

@dataclass
class Seed:
    row: int
    col: int
    slice: int      # 0-based slice index into the 3D volume


def _mid_slice(volume3d: np.ndarray) -> int:
    return volume3d.shape[2] // 2


def auto_seed(volume3d: np.ndarray, norm_low_pct=1.0, norm_high_pct=99.0) -> Seed:
    """
    Automatic center-of-brightness seed on the mid-cavity slice.

    The LV blood pool is bright and roughly central. We bias toward the image
    center (the LV sits near the middle of a SAX acquisition), take the
    brightest connected blob in that central window, and use its centroid.
    """
    z = _mid_slice(volume3d)
    img = normalize_image(volume3d[:, :, z], norm_low_pct, norm_high_pct)
    rows, cols = img.shape

    # Central window weighting: distance from centre, normalized.
    rr, cc = np.ogrid[:rows, :cols]
    cr, cc0 = rows / 2.0, cols / 2.0
    dist = np.sqrt(((rr - cr) / rows) ** 2 + ((cc - cc0) / cols) ** 2)
    central = np.exp(-(dist ** 2) / (2 * 0.25 ** 2))   # gaussian centre bias

    weighted = img * central
    # Threshold to the brightest ~10% within the weighted map, keep biggest blob.
    thr = np.percentile(weighted[weighted > 0], 90)
    mask = weighted >= thr
    labels, n = ndi.label(mask)
    if n == 0:
        return Seed(row=rows // 2, col=cols // 2, slice=z)
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    cy, cx = ndi.center_of_mass(mask, labels, biggest)
    return Seed(row=int(round(cy)), col=int(round(cx)), slice=z)


def interactive_seed(volume3d: np.ndarray, title: str,
                     norm_low_pct=1.0, norm_high_pct=99.0) -> Seed:
    """
    Show the ED mid-cavity slice and capture a single click inside the LV.

    Forces an interactive matplotlib backend so the click window appears even
    when another module (e.g. plotting) has selected a non-interactive one.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # Ensure an interactive GUI backend is active; otherwise plt.ginput cannot
    # capture a click and the window never appears. plt.switch_backend() truly
    # attempts to load the GUI toolkit and raises if it is unavailable, so we
    # try the common Windows/conda backends in order and keep the first that
    # loads. (matplotlib.use() can silently "succeed" without a working GUI,
    # which is why we use switch_backend here instead.)
    if "agg" in matplotlib.get_backend().lower():
        working = None
        for backend in ("QtAgg", "Qt5Agg", "TkAgg"):
            try:
                plt.switch_backend(backend)
                working = backend
                break
            except Exception:
                continue
        if working is None:
            raise RuntimeError(
                "No interactive matplotlib backend could be loaded, so the "
                "seed-click window cannot be shown. Try:  pip install PyQt5   "
                "then restart the terminal. (Or use --auto-seed to seed "
                "automatically without clicking.)"
            )

    z = _mid_slice(volume3d)
    img = normalize_image(volume3d[:, :, z], norm_low_pct, norm_high_pct)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{title}\nClick once inside the LV blood pool (mid-slice {z})")
    ax.axis("off")
    fig.canvas.draw()
    plt.show(block=False)
    pts = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not pts:
        raise RuntimeError("Seed selection cancelled (no click received).")
    x, y = pts[0]
    return Seed(row=int(round(y)), col=int(round(x)), slice=z)


def choose_seed(volume3d: np.ndarray, title: str, interactive: bool,
                norm_low_pct=1.0, norm_high_pct=99.0) -> Seed:
    """Dispatch to interactive or automatic seeding."""
    if interactive:
        return interactive_seed(volume3d, title, norm_low_pct, norm_high_pct)
    return auto_seed(volume3d, norm_low_pct, norm_high_pct)


# ---------------------------------------------------------------------------
# Segment_blood
# ---------------------------------------------------------------------------
# 
# Segments the LV blood pool on one 2D slice (the core image-processing step).
# Steps: normalize -> restrict to a circular region around the seed (so far-away
# bright things can't be grabbed) -> threshold to keep bright (blood) pixels ->
# clean up (fill holes, drop specks, smooth) -> keep the connected blob that
# contains the seed -> return the mask and its new centre.
# Key detail: the threshold is the higher of a seed-relative level and an Otsu
# split (to stop leaking into bright muscle), but capped at fill_cap_frac*seed
# so it can't sit so high that it clips the dim end-systolic cavity. That cap
# was the fix for severe ES under-segmentation.

def _disk_footprint(radius_px: int):
    return disk(max(1, int(radius_px)))


def _circular_roi(shape, center_rc, dx, dy, radius_mm):
    """Boolean ROI mask: voxels within radius_mm (physical) of center_rc."""
    rows, cols = shape
    rr, cc = np.ogrid[:rows, :cols]
    dr_mm = (rr - center_rc[0]) * dy
    dc_mm = (cc - center_rc[1]) * dx
    dist_mm = np.sqrt(dr_mm ** 2 + dc_mm ** 2)
    return dist_mm <= radius_mm


def _choose_component(labels, n_labels, center_rc):
    """
    Pick the component containing the center pixel; if the center lands on
    background, fall back to the component whose centroid is nearest the center.
    """
    rows, cols = labels.shape
    r = int(np.clip(round(center_rc[0]), 0, rows - 1))
    c = int(np.clip(round(center_rc[1]), 0, cols - 1))
    hit = labels[r, c]
    if hit > 0:
        return int(hit)

    centroids = ndi.center_of_mass(
        np.ones_like(labels), labels, index=np.arange(1, n_labels + 1)
    )
    best, best_d = 1, np.inf
    for idx, (cy, cx) in enumerate(centroids, start=1):
        d = (cy - center_rc[0]) ** 2 + (cx - center_rc[1]) ** 2
        if d < best_d:
            best, best_d = idx, d
    return best


def _keep_largest(mask):
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    return labels == biggest


def segment_blood_slice(slice_img: np.ndarray, center_rc, voxel_size_mm,
                        params: SegmentationParams = None):
    """
    Segment the LV blood pool on one 2D slice.

    Parameters
    ----------
    slice_img : 2D array (raw intensities)
    center_rc : (row, col) propagated centre to anchor the ROI/component pick
    voxel_size_mm : (dx, dy, dz); only dx, dy are used here
    params : SegmentationParams

    Returns
    -------
    mask : 2D bool array of the blood pool
    center_rc : updated (row, col) centroid of the segmented cavity
    """
    if params is None:
        params = SegmentationParams()

    dx, dy = float(voxel_size_mm[0]), float(voxel_size_mm[1])
    img = normalize_image(slice_img, params.norm_low_pct, params.norm_high_pct)

    roi = _circular_roi(img.shape, center_rc, dx, dy, params.roi_radius_mm)
    roi_values = img[roi]
    if roi_values.size == 0 or np.all(roi_values == roi_values.flat[0]):
        return np.zeros(img.shape, dtype=bool), center_rc

    rows, cols = img.shape
    sr = int(np.clip(round(center_rc[0]), 0, rows - 1))
    sc = int(np.clip(round(center_rc[1]), 0, cols - 1))

    # Reference intensity of the blood pool, sampled robustly from a small
    # neighbourhood around the seed/center (median is robust to a stray dark
    # pixel).
    rr0, rr1 = max(0, sr - 2), min(rows, sr + 3)
    cc0, cc1 = max(0, sc - 2), min(cols, sc + 3)
    seed_intensity = float(np.median(img[rr0:rr1, cc0:cc1]))

    # The seed must sit on bright tissue, else there is no cavity on
    # this slice (apex/base beyond the ventricle, or a bad seed).
    if seed_intensity < params.min_seed_intensity:
        return np.zeros(img.shape, dtype=bool), center_rc

    # Two thresholds, and we take the stricter (higher) of them, then cap:
    #   (a) seed-relative lower bound — fills the cavity in normal hearts where
    #       blood is much brighter than muscle.
    #   (b) an Otsu split of the ROI intensities — separates the brightest
    #       (blood) population from the rest. When muscle is bright (HCM), Otsu
    #       sits high and stops the region leaking through the endocardial
    #       border into THICK BRIGHT myocardium.
    lower_seed = params.blood_seed_frac * seed_intensity
    try:
        otsu_split = float(threshold_otsu(roi_values))
    except ValueError:
        otsu_split = lower_seed
    fill_cap = params.fill_cap_frac * seed_intensity
    lower = min(max(lower_seed, otsu_split), fill_cap)
    # Guard: never let the cap fall below the seed-relative floor (which would
    # happen only if fill_cap_frac < blood_seed_frac, a misconfiguration).
    lower = max(lower, lower_seed)


    candidate = (img >= lower) & roi
    candidate = ndi.binary_fill_holes(candidate)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", FutureWarning)
        candidate = remove_small_objects(candidate, min_size=params.min_blob_px + 1)
    candidate = closing(candidate, _disk_footprint(params.close_radius_px))

    labels, n = ndi.label(candidate)
    if n == 0:
        return np.zeros(img.shape, dtype=bool), center_rc

    chosen = _choose_component(labels, n, center_rc)
    mask = labels == chosen
    mask = ndi.binary_fill_holes(mask)
    mask = closing(mask, _disk_footprint(params.close_radius_px))
    mask = _keep_largest(mask)

    if mask.any():
        cy, cx = ndi.center_of_mass(mask)
        center_rc = (float(cy), float(cx))
    return mask, center_rc


# ---------------------------------------------------------------------------
# Segment_myo
# ---------------------------------------------------------------------------
# 
# Segments the myocardium (the muscle ring) on a slice, using the already-found
# blood pool as an anchor. It grows outward from the blood pool one thin ring at
# a time, keeping pixels whose brightness is in the "muscle" range (darker than
# blood, brighter than lung/fat), and stops when a ring stops looking like
# muscle (the outer/epicardial border). Myocardium = the outer disc minus the
# blood pool. Deliberately conservative so it doesn't absorb nearby organs.

def _disk_footprint(radius_px: int):
    return disk(max(1, int(radius_px)))


def segment_myo_slice(slice_img: np.ndarray, blood_mask: np.ndarray,
                      voxel_size_mm, params: MyocardiumParams = None):
    """
    Segment the LV myocardium ring on one 2D slice.

    Parameters
    ----------
    slice_img : 2D raw intensity slice
    blood_mask : 2D bool, the endocardial (blood-pool) mask for this slice
    voxel_size_mm : (dx, dy, dz)
    params : MyocardiumParams

    Returns
    -------
    myo_mask : 2D bool myocardium ring
    epi_mask : 2D bool solid epicardial disc (myo + blood), useful for plots
    """
    if params is None:
        params = MyocardiumParams()

    if not blood_mask.any():
        empty = np.zeros(slice_img.shape, dtype=bool)
        return empty, empty

    dx, dy = float(voxel_size_mm[0]), float(voxel_size_mm[1])
    img = normalize_image(slice_img)

    # Cavity-relative intensity window (adapts per slice/scanner). Muscle is
    # darker than blood but brighter than the lung/fat void around the heart.
    cavity_mean = float(img[blood_mask].mean())
    low = params.myo_low_frac * cavity_mean
    high = params.myo_high_frac * cavity_mean

    # Distance (mm) outward from the blood pool.
    outside = ~blood_mask
    dist_mm = ndi.distance_transform_edt(outside, sampling=(dy, dx))

    # The myocardium is the muscle shell immediately outside the cavity, not
    # everything within max_wall_mm. We grow outward one ring at a time and
    # stop where muscle-intensity voxels run out (the epicardial border).
    # Without this, mid-grey structures far from the heart (chest wall, liver)
    # within the search band get absorbed and the wall is hugely over-measured.
    muscle_intensity = (img >= low) & (img <= high)
    step = max(dx, dy)                      # mm per ring
    shell = np.zeros_like(blood_mask, dtype=bool)
    prev_ring_filled = True
    r = 0.0
    while r < params.max_wall_mm and prev_ring_filled:
        r0, r1 = r, r + step
        ring = (dist_mm > r0) & (dist_mm <= r1)
        ring_muscle = ring & muscle_intensity
        # Fraction of this ring that looks like muscle. If the wall has ended,
        # this drops sharply (we've reached fat/lung/RV-blood beyond the epi).
        ring_total = ring.sum()
        frac = (ring_muscle.sum() / ring_total) if ring_total else 0.0
        if frac >= params.shell_min_frac:
            shell |= ring_muscle
            r = r1
        else:
            prev_ring_filled = False

    # Solidify into an epicardial disc, then subtract blood to get the ring.
    epi = blood_mask | shell
    epi = closing(epi, _disk_footprint(params.epi_close_px))
    epi = ndi.binary_fill_holes(epi)

    # Keep the component that contains the blood pool (drop disconnected fat).
    labels, n = ndi.label(epi)
    if n > 1:
        overlap = [(labels == k)[blood_mask].sum() for k in range(1, n + 1)]
        keep = int(np.argmax(overlap)) + 1
        epi = labels == keep

    myo = epi & ~blood_mask
    return myo, epi


def segment_myo_volume(volume3d: np.ndarray, blood_volume: np.ndarray,
                       voxel_size_mm, params: MyocardiumParams = None):
    """Apply the myocardium segmenter to every slice that has a blood pool."""
    if params is None:
        params = MyocardiumParams()
    myo = np.zeros_like(blood_volume, dtype=bool)
    epi = np.zeros_like(blood_volume, dtype=bool)
    for z in range(volume3d.shape[2]):
        if not blood_volume[:, :, z].any():
            continue
        m, e = segment_myo_slice(
            volume3d[:, :, z], blood_volume[:, :, z], voxel_size_mm, params
        )
        myo[:, :, z] = m
        epi[:, :, z] = e
    return myo, epi


# ---------------------------------------------------------------------------
# Propagate
# ---------------------------------------------------------------------------
#
# Takes the single-slice blood segmenter and extends it in two directions:
#   SPATIAL (segment_blood_volume): start at the seed slice, walk up toward the
#     apex and down toward the base, carrying the cavity centre forward. Stops a
#     direction when the area collapses, balloons, or the cavity tapers away.
#   TEMPORAL (segment_cycle): repeat the whole 3D segmentation on every time
#     frame to build the volume-vs-time curve.
# Several guards live here:
#   - taper_min_run: needs 2 consecutive shrinking slices before it assumes the
#     cavity is tapering (so one noisy dip doesn't end propagation early).
#   - seed_area_cap_mult: rejects a slice that balloons far above the seed slice
#     (a leak into other tissue or a truncated base).
#   - _segment_slice_area_constrained: if a neighbour slice leaks, RE-segments
#     it at a stricter threshold instead of abandoning the whole direction
#     (this fixed the case where the volume collapsed to one slice -> EF 94%).
#   - _interpolate_outliers: on the time curve, replaces frames that ballooned
#     (blowup_mult) or collapsed (dropout_frac) with interpolated values, so a
#     few bad frames don't wreck EF. Those frames are logged as "interpolated".

def _should_stop(area, prev_area, prop: PropagationParams, seed_area=0):
    if area < prop.min_slice_px:
        return True
    if prev_area > 0 and area > prop.area_grow_max * prev_area:
        return True
    if prev_area > 0 and area < prop.area_shrink_min * prev_area:
        return True
    # Seed-relative leak cap. The per-slice grow cap above only compares to the
    # immediately previous slice, so a leak that inflates gradually across
    # slices (each step under area_grow_max) slips through - and a single slice
    # that balloons relative to the trustworthy SEED slice is a leak into
    # adjacent tissue or into a truncated base/apex (where the true cavity is
    # absent or tiny), not a real cross-section. The LV cross-section never
    # exceeds a few times the seed-slice area, so cap against the seed.
    if seed_area > 0 and area > prop.seed_area_cap_mult * seed_area:
        return True
    return False


def _segment_slice_area_constrained(slice_img, center, voxel_size_mm, seg, prop,
                                    prev_area, seed_area):
    """
    Segment one slice, retrying at progressively stricter thresholds if the
    result leaks past the grow cap.

    A slice whose area exceeds area_grow_max * prev_area (or the seed-relative
    cap) has bled into adjacent bright tissue because the global fill threshold
    (tuned for the cavity) is too low for this slice's local intensity. Rather
    than abandon propagation, we re-segment the slice at increasing
    blood_seed_frac until the area falls within the cap, then accept that
    constrained mask. If even the strictest setting still leaks, the slice is a
    true boundary (we have left the cavity) and the caller stops.

    Returns (mask, center, area, leaked_out) where leaked_out is True only if no
    threshold in the retry ladder brought the area within bounds.
    """
    from dataclasses import replace

    m, c = segment_blood_slice(slice_img, center, voxel_size_mm, seg)
    area = int(m.sum())

    def over_cap(a):
        if prev_area > 0 and a > prop.area_grow_max * prev_area:
            return True
        if seed_area > 0 and a > prop.seed_area_cap_mult * seed_area:
            return True
        return False

    if not over_cap(area):
        return m, c, area, False

    # Retry ladder: raise blood_seed_frac toward a ceiling. Each step excludes
    # more of the bright bridge to the neighbour, shrinking the leak toward the
    # true cavity. Accept the first (least-strict) setting that fits the cap, to
    # avoid over-tightening a genuinely large slice. Start above the effective
    # fill threshold actually in use (max of blood_seed_frac and the cap), so we
    # do not waste a rung re-testing the threshold that already leaked.
    base_frac = max(seg.blood_seed_frac, getattr(seg, "fill_cap_frac",
                                                 seg.blood_seed_frac))
    ceiling = getattr(prop, "retry_frac_ceiling", 0.75)
    step = getattr(prop, "retry_frac_step", 0.05)
    frac = base_frac + step
    best = (m, c, area)
    while frac <= ceiling + 1e-9:
        seg_try = replace(seg, blood_seed_frac=frac, fill_cap_frac=frac)
        m2, c2 = segment_blood_slice(slice_img, center, voxel_size_mm, seg_try)
        a2 = int(m2.sum())
        if a2 >= prop.min_slice_px and not over_cap(a2):
            return m2, c2, a2, False     # constrained fit found
        # keep the smallest-so-far in case nothing fits (for diagnostics)
        if a2 < best[2] and a2 >= prop.min_slice_px:
            best = (m2, c2, a2)
        frac += step

    # Nothing in the ladder fit the cap: genuine boundary / unrecoverable leak.
    return best[0], best[1], best[2], True


def _propagate_direction(volume3d, z_range, start_center, start_area,
                         voxel_size_mm, seg, prop, mask3d):
    """
    Segment outward from the seed slice in one direction (toward apex or base).

    Beyond the Ppeak (widest) slice the true LV cross-section tapers, so once we
    are confidently past the peak we forbid re-expansion: a bright apical/basal
    partial-volume ring must not re-inflate the cavity.

    When the seed is NOT at the mid-cavity slice (e.g. an ES seed clicked near the base), 
    propagation first climbs base -> mid (area grows) and only then tapers mid -> 
    apex. A naive "lock as soon as area dips once" rule mis-fires on a single noisy 
    dip near the base/outflow: it latches on an early slice, then breaks at the very next
    slice when the area legitimately climbs toward mid-cavity, capturing only a
    sliver of the true cavity. We therefore require the shrink to
    persist for `taper_min_run` consecutive slices before locking, and a single
    dip that immediately recovers does not lock. A genuine apex/base taper
    shrinks over many consecutive slices, so the over-grow protection is kept.

    Adaptive retry: when a neighbour slice leaks past the grow cap under the
    global fill threshold, we re-segment that slice at stricter thresholds
    rather than abandon propagation. Only if no threshold recovers an in-bounds
    cavity do we treat the slice as a true boundary and stop.
    """
    taper_min_run = getattr(prop, "taper_min_run", 2)
    seed_area = start_area
    center, prev_area = start_center, start_area
    shrink_run, locked = 0, False
    for z in z_range:
        m, center, area, leaked = _segment_slice_area_constrained(
            volume3d[:, :, z], center, voxel_size_mm, seg, prop,
            prev_area, seed_area
        )
        if area < prop.min_slice_px or leaked:
            break
        if area < prev_area:
            shrink_run += 1
        else:
            shrink_run = 0
        if shrink_run >= taper_min_run:
            locked = True
        # Once we have confidently tapered, re-expansion means we have hit a
        # bright partial-volume ring beyond the ventricle: stop.
        if locked and area > prev_area:
            break
        mask3d[:, :, z] = m
        prev_area = area
    return mask3d


def segment_blood_volume(volume3d, seed: Seed, voxel_size_mm,
                         seg: SegmentationParams = None,
                         prop: PropagationParams = None):
    """
    Region-grow the blood pool through a 3D volume from a seed slice.

    Returns a 3D boolean mask.
    """
    seg = seg or SegmentationParams()
    prop = prop or PropagationParams()

    mask3d = np.zeros(volume3d.shape, dtype=bool)
    z0 = int(np.clip(seed.slice, 0, volume3d.shape[2] - 1))

    seed_mask, seed_center = segment_blood_slice(
        volume3d[:, :, z0], (seed.row, seed.col), voxel_size_mm, seg
    )
    if not seed_mask.any():
        return mask3d  # seed slice failed; caller can warn

    mask3d[:, :, z0] = seed_mask

    # Upward (+z) and downward (-z) from the seed slice.
    mask3d = _propagate_direction(
        volume3d, range(z0 + 1, volume3d.shape[2]),
        seed_center, int(seed_mask.sum()), voxel_size_mm, seg, prop, mask3d
    )
    mask3d = _propagate_direction(
        volume3d, range(z0 - 1, -1, -1),
        seed_center, int(seed_mask.sum()), voxel_size_mm, seg, prop, mask3d
    )
    return mask3d


@dataclass
class CycleResult:
    """Per-frame LV blood-pool volumes over the cardiac cycle."""
    volumes_ml: np.ndarray                 # length = n_frames, post-interpolation
    raw_volumes_ml: np.ndarray             # before interpolation
    interpolated_frames: list = field(default_factory=list)  # 0-based indices
    masks: dict = field(default_factory=dict)  # frame_idx(0-based) -> 3D mask


def _interpolate_outliers(volumes, frac, dropout_frac=0.5, blowup_mult=2.2):
    """
    Replace implausible frames with linear interpolation across good
    neighbours. Triggers: (1) local deviation from the neighbour median by
    more than `frac`; (2) a collapse below `dropout_frac` of the cycle median
    (a segmentation dropout); (3) a blow-up above `blowup_mult` times a
    leak-resistant baseline (segmentation leakage into adjacent anatomy).
    Returns (clean_volumes, interpolated_indices).
    """
    v = volumes.astype(float).copy()
    n = len(v)
    good = np.ones(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - 1)
        hi = min(n - 1, i + 1)
        neighbours = [v[j] for j in (lo, hi) if j != i and good[j]]
        if not neighbours:
            continue
        med = np.median(neighbours)
        if med > 0 and abs(v[i] - med) / med > frac:
            good[i] = False

    # Blow-up guard: a frame whose volume balloons far above the rest of the
    # cycle is segmentation leakage (the region grew out of the cavity into the
    # RV/outflow/bright pool), not a real end-diastole. The neighbour-median
    # test above misses this when the leak spans a contiguous run of frames,
    # because each leaked frame's neighbours are themselves leaked. We therefore
    # anchor to a leak-resistant baseline: the 25th percentile of positive
    # volumes. Leakage pushes values UP, so as long as fewer than ~half the
    # frames leak, the low quartile still reflects the true cavity scale and the
    # inflated plateau is caught.
    positive = v[v > 0]
    if positive.size:
        cycle_med = float(np.median(positive))
        baseline = float(np.percentile(positive, 25))
        for i in range(n):
            if v[i] < dropout_frac * cycle_med:
                good[i] = False
            if baseline > 0 and v[i] > blowup_mult * baseline:
                good[i] = False

    interp = []
    idx = np.arange(n)
    if good.any() and not good.all():
        v[~good] = np.interp(idx[~good], idx[good], v[good])
        interp = list(idx[~good])
    return v, interp


def _frame_centroid(mask3d, seed_slice):
    """
    In-plane (row, col) centroid of the cavity on the seed slice of a 3D mask,
    falling back to the whole-volume centroid, or None if the mask is empty.
    """
    if not mask3d.any():
        return None
    z = int(np.clip(seed_slice, 0, mask3d.shape[2] - 1))
    sl = mask3d[:, :, z]
    if sl.any():
        cy, cx = ndi.center_of_mass(sl)
        return (float(cy), float(cx))
    # seed slice empty on this frame: use the slice with the most cavity
    areas = [mask3d[:, :, k].sum() for k in range(mask3d.shape[2])]
    z2 = int(np.argmax(areas))
    cy, cx = ndi.center_of_mass(mask3d[:, :, z2])
    return (float(cy), float(cx))


def segment_cycle(cine4d, seed: Seed, voxel_volume_ml, voxel_size_mm,
                  seg: SegmentationParams = None,
                  prop: PropagationParams = None,
                  store_masks=True, ed_frame_idx: int = None) -> CycleResult:
    """
    Segment the LV blood pool in every frame of the cine and return the
    volume-vs-time curve.

    Temporal seed tracking: rather than segment every frame from the same fixed
    ED seed, we start at the ED frame (where the user's click is most reliable)
    and walk outward in both temporal directions, re-anchoring each frame's
    in-plane seed to the previous frame's cavity centroid. This lets the seed
    follow a cavity that shrinks and shifts toward end-systole — essential for
    HCM, whose small systolic cavity moves off a fixed ED seed point. If a
    frame segments empty, the last good centroid is retained so tracking does
    not derail.
    """
    seg = seg or SegmentationParams()
    prop = prop or PropagationParams()

    n_frames = cine4d.shape[3]
    raw = np.zeros(n_frames, dtype=float)
    masks = {}

    # Segment each frame independently from the fixed seed. (An earlier
    # temporal-tracking variant re-anchored the seed to the previous frame's
    # centroid, but on small/shifting HCM cavities the anchor could drift onto
    # muscle and then track the wrong structure, inverting the curve. The fixed
    # seed is more stable; per-frame outliers are handled by interpolation.)
    for f in range(n_frames):
        m = segment_blood_volume(cine4d[:, :, :, f], seed, voxel_size_mm, seg, prop)
        raw[f] = m.sum() * voxel_volume_ml
        if store_masks:
            masks[f] = m

    clean, interp = _interpolate_outliers(
        raw, prop.frame_outlier_frac, prop.dropout_frac, prop.blowup_mult
    )
    return CycleResult(
        volumes_ml=clean, raw_volumes_ml=raw,
        interpolated_frames=interp, masks=masks,
    )


# ---------------------------------------------------------------------------
# Wall_thickness
# ---------------------------------------------------------------------------
# 
# Measures the LV wall thickness at end-diastole, on the slice where the cavity
# is largest. Three methods are provided; the pipeline uses the "profile" one:
#   - profile_thickness (used): casts rays out from the cavity centre and
#     measures, along each ray, how far the muscle band extends from the inner
#     (endocardial) to outer (epicardial) border, reading the image directly.
#     Because it reads the image (not a possibly-overgrown muscle mask) it can't
#     be inflated by a leaky myo segmentation.
#   - radial_thickness / centreline_thickness: alternative methods kept for
#     comparison / fallback.

@dataclass
class WallThickness:
    mean_mm: float
    max_mm: float
    method: str


def _mid_cavity_slice(blood_volume):
    """Slice index with the largest blood-pool area (the mid-cavity slice)."""
    areas = [blood_volume[:, :, z].sum() for z in range(blood_volume.shape[2])]
    return int(np.argmax(areas))


def centreline_thickness(myo_slice, voxel_size_mm) -> WallThickness:
    """
    Centerline wall thickness on a single myocardium slice.

    Uses the medial axis of the ring: thickness at a medial point is 2x its
    distance to the ring boundary. Reported as mean and max over the medial
    axis. Works for non-circular rings (unlike the radial method).
    """
    dy, dx = float(voxel_size_mm[1]), float(voxel_size_mm[0])
    if not myo_slice.any():
        return WallThickness(float("nan"), float("nan"), "centreline")

    # Distance (mm) from each myocardium voxel to the nearest edge.
    dist_mm = ndi.distance_transform_edt(myo_slice, sampling=(dy, dx))

    # Medial axis of the ring; thickness = 2 x distance at the medial axis.
    medial = skeletonize(myo_slice)
    ridge = dist_mm[medial]
    ridge = ridge[ridge > 0]
    if ridge.size == 0:
        # Fall back to using all myocardium distances.
        ridge = dist_mm[myo_slice]
        ridge = ridge[ridge > 0]
        if ridge.size == 0:
            return WallThickness(float("nan"), float("nan"), "centreline")

    return WallThickness(
        mean_mm=float(2.0 * ridge.mean()),
        max_mm=float(2.0 * ridge.max()),
        method="centreline",
    )


def radial_thickness(blood_slice, myo_slice, voxel_size_mm,
                     n_rays=72, max_wall_mm=25.0) -> WallThickness:
    """
    Radial wall thickness measured as the endocardium-to-epicardium border
    distance along rays cast from the cavity centroid.

    For each ray we find the radius where it leaves the blood pool (the
    endocardial border) and the radius where it leaves the myocardium (the
    epicardial border); the wall thickness on that ray is the physical distance
    between those two borders. This is the standard clinical definition and,
    unlike counting every myocardium pixel a ray crosses, it is not inflated by
    tangential rays or by a thick/blobby mask. Rays with no clear wall, or an
    implausibly large one (> max_wall_mm), are discarded. The representative
    thickness is the median over rays (robust to a few bad rays).
    """
    dy, dx = float(voxel_size_mm[1]), float(voxel_size_mm[0])
    if not myo_slice.any() or not blood_slice.any():
        return WallThickness(float("nan"), float("nan"), "radial")

    epi = blood_slice | myo_slice          # solid epicardial disc
    cy, cx = ndi.center_of_mass(blood_slice)
    rows, cols = myo_slice.shape
    max_r = int(np.hypot(rows, cols))

    thicknesses = []
    for theta in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        sr, sc = np.sin(theta), np.cos(theta)
        endo_r = None       # last radius still inside blood
        epi_r = None        # last radius still inside epicardial disc
        for r in range(max_r):
            rr = int(round(cy + r * sr))
            cc = int(round(cx + r * sc))
            if not (0 <= rr < rows and 0 <= cc < cols):
                break
            if blood_slice[rr, cc]:
                endo_r = r
            if epi[rr, cc]:
                epi_r = r
        if endo_r is None or epi_r is None or epi_r <= endo_r:
            continue
        step_mm = np.hypot(sr * dy, sc * dx)
        wall = (epi_r - endo_r) * step_mm
        if 0 < wall <= max_wall_mm:
            thicknesses.append(wall)

    if not thicknesses:
        return WallThickness(float("nan"), float("nan"), "radial")
    t = np.array(thicknesses)
    return WallThickness(float(np.median(t)), float(t.max()), "radial")


def profile_thickness(slice_img, blood_slice, voxel_size_mm,
                      n_rays=120, max_wall_mm=22.0,
                      norm_low_pct=1.0, norm_high_pct=99.0) -> WallThickness:
    """
    Wall thickness from local intensity profiles along rays — no myocardium
    segmentation required (so nothing can leak into the RV/septum).

    From the blood-pool centroid we cast rays outward. Along each ray the
    intensity goes: bright (blood) -> mid-grey (myocardium) -> dark (lung/fat)
    or bright again (RV blood). Starting at the endocardial border (where the
    ray leaves the blood pool), we walk outward while the intensity stays in
    the muscle band; the wall ends where intensity falls below a muscle floor.
    Wall thickness on that ray is the length of that muscle segment.

    Rays are rejected when the muscle segment exceeds max_wall_mm, or when the
    pixel past the wall is brighter than the muscle band (we reached the RV
    blood pool through the septum, not epicardial fat/lung) — septal rays do
    not give a clean free-wall thickness. The representative thickness is the
    median over accepted rays (robust to a few bad rays).
    """

    dy, dx = float(voxel_size_mm[1]), float(voxel_size_mm[0])
    if not blood_slice.any():
        return WallThickness(float("nan"), float("nan"), "profile")

    img = normalize_image(slice_img, norm_low_pct, norm_high_pct)
    blood_mean = float(img[blood_slice].mean())
    muscle_floor = 0.20 * blood_mean      # below this = lung/fat/dark void
    muscle_ceil = 0.85 * blood_mean       # above this = still blood / RV blood

    cy, cx = ndi.center_of_mass(blood_slice)
    rows, cols = img.shape
    step = 0.5 * min(dx, dy)              # sub-pixel marching, in mm
    max_steps = int((max_wall_mm + 30.0) / step)

    walls = []
    for theta in np.linspace(0, 2 * np.pi, n_rays, endpoint=False):
        sr, sc = np.sin(theta), np.cos(theta)
        # 1) endocardial border: where the ray leaves the blood pool
        endo_mm = None
        for k in range(max_steps):
            d = k * step
            rr = int(round(cy + (d / dy) * sr))
            cc = int(round(cx + (d / dx) * sc))
            if not (0 <= rr < rows and 0 <= cc < cols):
                break
            if not blood_slice[rr, cc]:
                endo_mm = d
                break
        if endo_mm is None:
            continue
        # 2) walk through muscle until intensity leaves the muscle band
        wall_mm = 0.0
        ended = False
        for k in range(1, max_steps):
            d = endo_mm + k * step
            rr = int(round(cy + (d / dy) * sr))
            cc = int(round(cx + (d / dx) * sc))
            if not (0 <= rr < rows and 0 <= cc < cols):
                break
            val = img[rr, cc]
            if muscle_floor < val < muscle_ceil:
                wall_mm = d - endo_mm
                if wall_mm > max_wall_mm:
                    wall_mm = 0.0
                    break
            else:
                if val >= muscle_ceil:
                    wall_mm = 0.0       # RV blood through septum: discard ray
                ended = True
                break
        if ended and 0 < wall_mm <= max_wall_mm:
            walls.append(wall_mm)

    if not walls:
        return WallThickness(float("nan"), float("nan"), "profile")
    w = np.array(walls)
    return WallThickness(float(np.median(w)), float(np.percentile(w, 90)),
                         "profile")


def diastolic_wall_thickness(ed_blood_volume, ed_myo_volume, voxel_size_mm,
                             ed_image_volume=None):
    """
    Representative end-diastolic wall thickness on the mid-cavity slice.

    Uses the local intensity-profile method (profile_thickness), which measures
    endocardium-to-epicardium distance directly from the image along rays and
    needs no myocardium ring — so it cannot leak into the RV/septum the way a
    segmented ring does. ed_image_volume (the raw ED image) is required for the
    profile method; if not supplied we fall back to the radial method on the
    myocardium mask. Returns (WallThickness, mid_slice_index).
    """
    z = _mid_cavity_slice(ed_blood_volume)
    if ed_image_volume is not None:
        wt = profile_thickness(ed_image_volume[:, :, z],
                               ed_blood_volume[:, :, z], voxel_size_mm)
        return wt, z
    return radial_thickness(ed_blood_volume[:, :, z],
                            ed_myo_volume[:, :, z], voxel_size_mm), z


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
# 
# All the figures saved to disk:
#   - plot_volume_curve: LV volume across the cardiac cycle, marking ED (max),
#     ES (min), and any interpolated frames (orange rings).
#   - plot_overlay: the slice image with three contours drawn on it -
#     GREEN = our segmented blood pool, BLUE = our segmented muscle,
#     RED = the expert ground-truth blood pool (for visual comparison).
#   - plot_summary: a scatter of EF vs wall thickness for all patients, coloured
#     by group, with the textbook threshold lines drawn in.

def _plt():
    import matplotlib
    # Only fall back to the non-interactive Agg backend if none is set. Forcing
    # Agg unconditionally would break the seed-click window (which needs a GUI
    # backend). Figures are still saved to disk regardless of backend.
    if matplotlib.get_backend().lower() == "":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_volume_curve(volumes_ml, patient_id, ed_idx=None, es_idx=None,
                      interpolated=None, save_path=None, show=False):
    """LV blood-pool volume across the cardiac cycle."""
    plt = _plt()
    frames = np.arange(1, len(volumes_ml) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frames, volumes_ml, "-o", ms=4, label="LV volume")
    if ed_idx is not None:
        ax.scatter([ed_idx + 1], [volumes_ml[ed_idx]], c="tab:green",
                   zorder=5, label=f"ED (max, {volumes_ml[ed_idx]:.1f} mL)")
    if es_idx is not None:
        ax.scatter([es_idx + 1], [volumes_ml[es_idx]], c="tab:red",
                   zorder=5, label=f"ES (min, {volumes_ml[es_idx]:.1f} mL)")
    if interpolated:
        ax.scatter([i + 1 for i in interpolated],
                   [volumes_ml[i] for i in interpolated],
                   facecolors="none", edgecolors="orange", s=80,
                   label="interpolated")
    ax.set_xlabel("Cardiac frame")
    ax.set_ylabel("LV blood-pool volume (mL)")
    ax.set_title(f"{patient_id}: LV volume over the cardiac cycle")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def plot_overlay(slice_img, blood_mask=None, myo_mask=None, gt_lv=None,
                 title="", save_path=None, show=False):
    """Overlay predicted blood (green) / myo (blue) / GT-LV outline (red)."""
    from skimage.segmentation import find_boundaries
    plt = _plt()

    img = slice_img.astype(float)
    img = (img - img.min()) / (np.ptp(img) + 1e-9)
    rgb = np.dstack([img, img, img])

    if blood_mask is not None and blood_mask.any():
        rgb[find_boundaries(blood_mask, mode="outer")] = [0, 1, 0]
    if myo_mask is not None and myo_mask.any():
        rgb[find_boundaries(myo_mask, mode="outer")] = [0.2, 0.4, 1.0]
    if gt_lv is not None and gt_lv.any():
        rgb[find_boundaries(gt_lv, mode="outer")] = [1, 0, 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


def plot_summary(rows, save_path=None, show=False):
    """
    Scatter of EF vs diastolic wall thickness, coloured by group, with the
    slide threshold lines drawn for orientation.

    rows: list of dicts with keys patient, group, ef_pct, wall_mm.
    """
    plt = _plt()
    colors = {"NOR": "tab:green", "DCM": "tab:red", "HCM": "tab:purple"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        ax.scatter(r["ef_pct"], r["wall_mm"],
                   c=colors.get(r["group"], "gray"), s=60)
        ax.annotate(r["patient"].replace("patient", "p"),
                    (r["ef_pct"], r["wall_mm"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(50, ls="--", c="green", alpha=0.4)
    ax.axvline(55, ls="--", c="purple", alpha=0.4)
    ax.axvline(40, ls="--", c="red", alpha=0.4)
    ax.axhline(12, ls=":", c="gray", alpha=0.5)
    ax.axhline(15, ls=":", c="gray", alpha=0.5)
    ax.set_xlabel("Ejection fraction (%)")
    ax.set_ylabel("Diastolic wall thickness (mm)")
    ax.set_title("EF vs wall thickness by patient group")
    handles = [plt.Line2D([0], [0], marker="o", ls="", c=c, label=g)
               for g, c in colors.items()]
    ax.legend(handles=handles, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
# 
# All file reading. It parses Info.cfg (group, ED/ES frame numbers, height,
# weight -> body-surface-area), loads the 4D cine and the ED/ES still frames
# plus their ground-truth label files (the *_gt.nii files), and exposes voxel
# size / voxel volume (needed to convert pixel counts into millilitres and
# millimetres). load_patient() bundles everything for one patient into one
# object the rest of the pipeline uses.

@dataclass
class PatientInfo:
    """Parsed contents of the Info.cfg file."""
    group: str = ""        # NOR / DCM / HCM / MINF / RV
    ed_frame: int = -1     # end-diastole frame index (1-based, as in filenames)
    es_frame: int = -1     # end-systole frame index
    nb_frame: int = -1     # number of frames in the cine
    height_cm: float = float("nan")
    weight_kg: float = float("nan")

    @property
    def bsa_m2(self) -> float:
        """Body surface area (Mosteller formula) for BSA-indexed volumes."""
        if np.isnan(self.height_cm) or np.isnan(self.weight_kg):
            return float("nan")
        return float(np.sqrt(self.height_cm * self.weight_kg / 3600.0))


def read_info_cfg(info_path: Path) -> PatientInfo:
    """Parse a 'Key: value' style Info.cfg into a PatientInfo."""
    info = PatientInfo()
    text = Path(info_path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        try:
            if key == "group":
                info.group = value
            elif key == "ed":
                info.ed_frame = int(round(float(value)))
            elif key == "es":
                info.es_frame = int(round(float(value)))
            elif key == "nbframe":
                info.nb_frame = int(round(float(value)))
            elif key == "height":
                info.height_cm = float(value)
            elif key == "weight":
                info.weight_kg = float(value)
        except ValueError:
            # Leave the field at its default if a value cannot be parsed.
            continue
    return info


def _load_nii(path: Path):
    """Load a NIfTI file, returning (data as float64-friendly array, nib image)."""
    img = nib.load(str(path))
    return np.asarray(img.dataobj), img


def voxel_size_mm(img) -> np.ndarray:
    """Return (dx, dy, dz) voxel size in millimetres from a NIfTI header."""
    zooms = img.header.get_zooms()
    return np.asarray(zooms[:3], dtype=float)


def voxel_volume_ml(img) -> float:
    """Volume of one voxel in millilitres (1 mL = 1000 mm^3)."""
    dx, dy, dz = voxel_size_mm(img)
    return float(dx * dy * dz) / 1000.0


@dataclass
class PatientData:
    """Everything needed to analyse one patient."""
    patient_id: str
    info: PatientInfo
    folder: Path

    # 4D cine: shape (rows, cols, slices, frames)
    cine: np.ndarray
    cine_img: object                # nib image (for header / affine)

    # ED / ES still frames and their ground truth (may be None if absent)
    ed_image: np.ndarray = None
    ed_gt: np.ndarray = None
    es_image: np.ndarray = None
    es_gt: np.ndarray = None

    @property
    def voxel_size_mm(self) -> np.ndarray:
        return voxel_size_mm(self.cine_img)

    @property
    def voxel_volume_ml(self) -> float:
        return voxel_volume_ml(self.cine_img)

    @property
    def n_frames(self) -> int:
        return self.cine.shape[3]

    @property
    def n_slices(self) -> int:
        return self.cine.shape[2]

    def frame(self, frame_1based: int) -> np.ndarray:
        """Return the 3D volume for a 1-based frame index (as in filenames)."""
        return self.cine[:, :, :, frame_1based - 1]


def _find_frame_files(folder: Path, patient_id: str):
    """Map frame number -> (image_path, gt_path or None) for explicit frames."""
    out = {}
    pattern = re.compile(rf"^{re.escape(patient_id)}_frame(\d+)(_gt)?\.nii(\.gz)?$")
    for p in folder.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        n = int(m.group(1))
        is_gt = m.group(2) is not None
        img_path, gt_path = out.get(n, (None, None))
        if is_gt:
            gt_path = p
        else:
            img_path = p
        out[n] = (img_path, gt_path)
    return out


def load_patient(folder: Path, patient_id: str) -> PatientData:
    """
    Load the 4D cine, Info.cfg, and the ED/ES still frames + ground truth
    for one patient. Frame files may be plain .nii or .nii.gz.
    """
    folder = Path(folder)
    info = read_info_cfg(folder / "Info.cfg")

    # 4D cine (try .nii then .nii.gz)
    cine_path = folder / f"{patient_id}_4d.nii"
    if not cine_path.exists():
        cine_path = folder / f"{patient_id}_4d.nii.gz"
    cine, cine_img = _load_nii(cine_path)
    cine = cine.astype(np.float64)
    if cine.ndim != 4:
        raise ValueError(
            f"{cine_path.name}: expected a 4D volume, got shape {cine.shape}"
        )

    data = PatientData(
        patient_id=patient_id, info=info, folder=folder,
        cine=cine, cine_img=cine_img,
    )

    frames = _find_frame_files(folder, patient_id)

    def _load_pair(frame_no):
        if frame_no not in frames:
            return None, None
        img_path, gt_path = frames[frame_no]
        image = _load_nii(img_path)[0].astype(np.float64) if img_path else None
        gt = _load_nii(gt_path)[0].astype(np.int16) if gt_path else None
        return image, gt

    data.ed_image, data.ed_gt = _load_pair(info.ed_frame)
    data.es_image, data.es_gt = _load_pair(info.es_frame)
    return data


# ---------------------------------------------------------------------------
# Run_
# ---------------------------------------------------------------------------
# 
# This is the entry point and the glue that calls everything above in order.
#   - parse_args(): defines the command-line flags (--patients, --all-patients,
#     --dataset, --output, --reset-seeds, --no-show, etc.).
#   - DEFAULT_DATASET / DEFAULT_PATIENTS: defaults used when no flags are given.
#   - the seed-cache helpers (load/save/seed_to_dict): remember the clicks in
#     seeds.json so you don't re-click on later runs.
#   - show_slice_montage / manual_seed: show the slice grid, ask which slice,
#     and capture the click.
#   - process_patient(): the per-patient pipeline in sequence - load data,
#     get ED & ES seeds, segment blood + muscle, compute volumes/EF/wall,
#     build the volume curve, validate against ground truth, check group
#     consistency, save overlays, and return one row of results.
#   - main(): resolves the patient list and paths, loops over patients (logging
#     any failures), writes summary.csv and summary_scatter.png at the end.
# Outputs written: summary.csv, seeds.json, summary_scatter.png, and per-patient
# overlay/curve/montage PNGs, all in the --output folder.

#!/usr/bin/env python3








DEFAULT_DATASET = Path(
    r"PATH\TO\database\training"
)


DEFAULT_PATIENTS = [
    "patient001", "patient004", "patient012", "patient019",
    "patient067", "patient069", "patient070", "patient072", "patient075",
    "patient021", "patient031", "patient040"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Semi-automatic dataset LV blood-pool and myocardium pipeline."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to the dataset training folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder for outputs (CSV, overlays, seed cache). Defaults to "
             "<dataset>/lv_segmentation_outputs if not given.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=DEFAULT_PATIENTS,
        help="Patient IDs to process.",
    )
    parser.add_argument(
        "--all-patients",
        action="store_true",
        help="Process every patientNNN folder found in the dataset directory "
             "(overrides --patients). Existing cached seeds are reused; only "
             "patients without a cached seed will prompt for a click.",
    )
    parser.add_argument(
        "--auto-seed",
        action="store_true",
        help="Debug only: use automatic seeds. Not recommended for final results.",
    )
    parser.add_argument(
        "--seed-from-gt-debug",
        action="store_true",
        help="Debug only: initialize seeds from ground truth. Do not use for final report.",
    )
    parser.add_argument(
        "--reset-seeds",
        action="store_true",
        help="Ignore saved manual seeds and ask for new clicks.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display result figures. Seed-selection windows still appear.",
    )
    parser.add_argument(
        "--roi-radius-mm",
        type=float,
        default=38.0,
        help="Search radius around the seed. Smaller values reduce off-target leakage.",
    )
    parser.add_argument(
        "--blood-frac",
        type=float,
        default=0.50,
        help="Seed-relative blood threshold fraction.",
    )
    return parser.parse_args()


def load_seed_cache(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_seed_cache(path: Path, cache):
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def seed_to_dict(seed: Seed):
    return {"row": int(seed.row), "col": int(seed.col), "slice": int(seed.slice)}


def seed_from_dict(value):
    return Seed(
        row=int(value["row"]),
        col=int(value["col"]),
        slice=int(value["slice"]),
    )


def show_slice_montage(volume, patient_id, phase_name, output_folder):
    """Save and show a numbered montage so the user can choose a seed slice."""
    import matplotlib.pyplot as plt

    n_slices = volume.shape[2]
    ncols = int(np.ceil(np.sqrt(n_slices)))
    nrows = int(np.ceil(n_slices / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows))
    axes = np.ravel(np.atleast_1d(axes))

    for z in range(n_slices):
        ax = axes[z]
        ax.imshow(normalize_image(volume[:, :, z]), cmap="gray")
        ax.set_title(f"slice {z + 1}", fontsize=9)
        ax.axis("off")

    for ax in axes[n_slices:]:
        ax.axis("off")

    fig.suptitle(f"{patient_id} {phase_name}: choose a slice with clear LV cavity")
    fig.tight_layout()

    montage_path = output_folder / f"{patient_id}_{phase_name.lower()}_seed_montage.png"
    fig.savefig(montage_path, dpi=130)

    print(f"  Seed montage saved: {montage_path}")
    print("  Inspect the montage, then close it and enter the slice number.")
    plt.show(block=True)
    plt.close(fig)
    return montage_path


def manual_seed(volume, patient_id, phase_name, output_folder):
    """Ask the user for a slice number and click inside the LV blood pool."""
    import matplotlib.pyplot as plt

    n_slices = volume.shape[2]
    default_slice_1based = n_slices // 2 + 1

    show_slice_montage(volume, patient_id, phase_name, output_folder)

    while True:
        answer = input(
            f"  {patient_id} {phase_name}: seed slice 1-{n_slices} "
            f"[default {default_slice_1based}]: "
        ).strip()
        if not answer:
            slice_1based = default_slice_1based
            break
        try:
            slice_1based = int(answer)
        except ValueError:
            print("  Please enter a whole-number slice index.")
            continue
        if 1 <= slice_1based <= n_slices:
            break
        print(f"  Slice must be between 1 and {n_slices}.")

    z = slice_1based - 1
    image = normalize_image(volume[:, :, z])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"{patient_id} {phase_name}, slice {slice_1based}: "
        "click inside LV blood pool"
    )
    ax.axis("off")
    fig.canvas.draw()
    plt.show(block=False)

    points = plt.ginput(1, timeout=0)
    plt.close(fig)
    if not points:
        raise RuntimeError("No seed click received.")

    x, y = points[0]
    seed = Seed(row=int(round(y)), col=int(round(x)), slice=z)
    print(
        f"  Seed saved for {patient_id} {phase_name}: "
        f"row={seed.row}, col={seed.col}, slice={seed.slice + 1}"
    )
    return seed


def gt_debug_seed(gt_volume):
    """Ground-truth centroid seed for debugging only."""
    lv = gt_volume == LABEL_LV
    if not lv.any():
        raise RuntimeError("GT debug seed requested, but no LV label exists.")
    areas = np.array([lv[:, :, z].sum() for z in range(lv.shape[2])])
    z = int(np.argmax(areas))
    rows, cols = np.where(lv[:, :, z])
    return Seed(
        row=int(round(float(rows.mean()))),
        col=int(round(float(cols.mean()))),
        slice=z,
    )


def choose_seed(volume, gt_volume, patient_id, phase_name, cfg, cache, output_folder, args):
    """Choose, cache, and return a seed for one patient phase."""
    cache_key = f"{patient_id}_{phase_name}"

    if args.seed_from_gt_debug:
        print("  WARNING: using GT-derived debug seed. Do not report this as final.")
        return gt_debug_seed(gt_volume), "gt_debug"

    if args.auto_seed:
        print("  WARNING: using auto-seed. Use manual seeds for final results.")
        return auto_seed(
            volume, cfg.seg.norm_low_pct, cfg.seg.norm_high_pct
        ), "auto"

    if not args.reset_seeds and cache_key in cache:
        seed = seed_from_dict(cache[cache_key])
        print(
            f"  Reusing cached seed for {cache_key}: "
            f"row={seed.row}, col={seed.col}, slice={seed.slice + 1}"
        )
        return seed, "manual_cached"

    seed = manual_seed(volume, patient_id, phase_name, output_folder)
    cache[cache_key] = seed_to_dict(seed)
    return seed, "manual"


def validate_if_available(predicted_mask, gt_volume, target_label):
    if gt_volume is None:
        return None
    return evaluate(predicted_mask, gt_volume == target_label)


def metric(metrics, key):
    if metrics is None:
        return float("nan")
    return float(metrics.get(key, float("nan")))


def process_patient(patient_id, cfg, show_results, cache, args):
    folder = cfg.paths.dataset_folder / patient_id
    data = load_patient(folder, patient_id)
    voxel_size = data.voxel_size_mm
    voxel_volume_ml = data.voxel_volume_ml
    out = cfg.paths.output_folder

    ed_frame = data.info.ed_frame
    es_frame = data.info.es_frame
    ed_vol = data.frame(ed_frame)
    es_vol = data.frame(es_frame)

    print(f"  group={data.info.group}  ED={ed_frame}  ES={es_frame}")

    ed_seed, ed_seed_mode = choose_seed(
        ed_vol, data.ed_gt, patient_id, "ED", cfg, cache, out, args
    )
    es_seed, es_seed_mode = choose_seed(
        es_vol, data.es_gt, patient_id, "ES", cfg, cache, out, args
    )

    ed_blood = segment_blood_volume(
        ed_vol, ed_seed, voxel_size, cfg.seg, cfg.prop
    )
    es_blood = segment_blood_volume(
        es_vol, es_seed, voxel_size, cfg.seg, cfg.prop
    )

    ed_myo, _ = segment_myo_volume(
        ed_vol, ed_blood, voxel_size, cfg.myo
    )
    es_myo, _ = segment_myo_volume(
        es_vol, es_blood, voxel_size, cfg.myo
    )

    lvedv_ml = float(ed_blood.sum() * voxel_volume_ml)
    lvesv_ml = float(es_blood.sum() * voxel_volume_ml)
    if lvesv_ml > lvedv_ml:
        lvedv_ml, lvesv_ml = lvesv_ml, lvedv_ml
    lvsv_ml = lvedv_ml - lvesv_ml
    lvef_pct = 100.0 * lvsv_ml / lvedv_ml if lvedv_ml > 0 else float("nan")
    edv_index = (
        lvedv_ml / data.info.bsa_m2
        if data.info.bsa_m2 and not np.isnan(data.info.bsa_m2)
        else float("nan")
    )

    cycle = segment_cycle(
        data.cine,
        ed_seed,
        voxel_volume_ml,
        voxel_size,
        cfg.seg,
        cfg.prop,
        store_masks=True,
        ed_frame_idx=ed_frame - 1,
    )

    wall, mid_z = diastolic_wall_thickness(
        ed_blood, ed_myo, voxel_size, ed_image_volume=ed_vol
    )

    blood_ed = validate_if_available(ed_blood, data.ed_gt, LABEL_LV)
    blood_es = validate_if_available(es_blood, data.es_gt, LABEL_LV)
    myo_ed = validate_if_available(ed_myo, data.ed_gt, LABEL_MYO)
    myo_es = validate_if_available(es_myo, data.es_gt, LABEL_MYO)

    cov = (
        assess_coverage(data.ed_gt, LABEL_LV)
        if data.ed_gt is not None
        else None
    )

    idx = CardiacIndices(
        lvedv_ml=lvedv_ml,
        lvesv_ml=lvesv_ml,
        lvsv_ml=lvsv_ml,
        lvef_pct=lvef_pct,
        ed_frame_idx=ed_frame - 1,
        es_frame_idx=es_frame - 1,
        edv_index_ml_m2=edv_index,
    )
    corr = correlate(
        data.info.group,
        idx.lvef_pct,
        wall.mean_mm,
        idx.edv_index_ml_m2,
        cfg.clinical,
    )

    plot_volume_curve(
        cycle.volumes_ml,
        patient_id,
        idx.ed_frame_idx,
        idx.es_frame_idx,
        cycle.interpolated_frames,
        save_path=out / f"{patient_id}_v2_volume_curve.png",
        show=show_results,
    )

    ed_areas = [ed_blood[:, :, z].sum() for z in range(ed_blood.shape[2])]
    ed_z = int(np.argmax(ed_areas)) if max(ed_areas) > 0 else mid_z
    gt_ed = data.ed_gt[:, :, ed_z] == LABEL_LV if data.ed_gt is not None else None
    plot_overlay(
        ed_vol[:, :, ed_z],
        ed_blood[:, :, ed_z],
        ed_myo[:, :, ed_z],
        gt_ed,
        title=f"{patient_id} ED v2: green=blood, blue=myo, red=GT blood",
        save_path=out / f"{patient_id}_v2_ed_overlay.png",
        show=show_results,
    )

    es_areas = [es_blood[:, :, z].sum() for z in range(es_blood.shape[2])]
    es_z = int(np.argmax(es_areas)) if max(es_areas) > 0 else es_blood.shape[2] // 2
    gt_es = data.es_gt[:, :, es_z] == LABEL_LV if data.es_gt is not None else None
    plot_overlay(
        es_vol[:, :, es_z],
        es_blood[:, :, es_z],
        es_myo[:, :, es_z],
        gt_es,
        title=f"{patient_id} ES v2: green=blood, blue=myo, red=GT blood",
        save_path=out / f"{patient_id}_v2_es_overlay.png",
        show=show_results,
    )

    return {
        "patient": patient_id,
        "group": data.info.group,
        "ed_frame": ed_frame,
        "es_frame": es_frame,
        "ed_seed_mode": ed_seed_mode,
        "ed_seed_row": ed_seed.row,
        "ed_seed_col": ed_seed.col,
        "ed_seed_slice_1based": ed_seed.slice + 1,
        "es_seed_mode": es_seed_mode,
        "es_seed_row": es_seed.row,
        "es_seed_col": es_seed.col,
        "es_seed_slice_1based": es_seed.slice + 1,
        "roi_radius_mm": cfg.seg.roi_radius_mm,
        "blood_frac": cfg.seg.blood_seed_frac,
        "lvedv_ml": round(idx.lvedv_ml, 2),
        "lvesv_ml": round(idx.lvesv_ml, 2),
        "lvsv_ml": round(idx.lvsv_ml, 2),
        "lvef_pct": round(idx.lvef_pct, 2),
        "edv_index_ml_m2": round(idx.edv_index_ml_m2, 2),
        "diastolic_wall_mean_mm": round(wall.mean_mm, 2),
        "diastolic_wall_max_mm": round(wall.max_mm, 2),
        "dice_lv_ed": round(metric(blood_ed, "dice"), 3),
        "dice_lv_es": round(metric(blood_es, "dice"), 3),
        "jaccard_lv_ed": round(metric(blood_ed, "jaccard"), 3),
        "jaccard_lv_es": round(metric(blood_es, "jaccard"), 3),
        "dice_myo_ed": round(metric(myo_ed, "dice"), 3),
        "dice_myo_es": round(metric(myo_es, "dice"), 3),
        "jaccard_myo_ed": round(metric(myo_ed, "jaccard"), 3),
        "jaccard_myo_es": round(metric(myo_es, "jaccard"), 3),
        "full_coverage": cov.full_coverage if cov else "unknown",
        "coverage_note": cov.reason if cov else "no ground truth",
        "n_interpolated_frames": len(cycle.interpolated_frames),
        "agrees_with_group": corr.agrees,
        "criteria_checked": " | ".join(corr.criteria_checked),
        "criteria_met": " | ".join(str(x) for x in corr.criteria_met),
        "ef_pct": round(idx.lvef_pct, 2),
        "wall_mm": round(wall.mean_mm, 2),
    }


def main():
    args = parse_args()

    cfg = Config()
    cfg.paths = Paths(dataset_folder=args.dataset, output_folder=args.output)
    if args.all_patients:
        found = sorted(
            d.name for d in args.dataset.iterdir()
            if d.is_dir() and d.name.startswith("patient")
            and (d / "Info.cfg").exists()
        )
        if not found:
            print(f"No patientNNN folders found in {args.dataset}")
            return 1
        cfg.patients = found
        print(f"--all-patients: found {len(found)} patients in {args.dataset}")
    else:
        cfg.patients = list(args.patients)
    cfg.interactive_seed = not args.auto_seed and not args.seed_from_gt_debug
    cfg.seg.roi_radius_mm = args.roi_radius_mm
    cfg.seg.blood_seed_frac = args.blood_frac
    cfg.paths.output_folder.mkdir(parents=True, exist_ok=True)

    seed_cache_path = cfg.paths.output_folder / "seeds.json"
    seed_cache = {} if args.reset_seeds else load_seed_cache(seed_cache_path)

    if args.auto_seed:
        print("\nWARNING: --auto-seed is for debugging only; it can select non-LV anatomy.")
    if args.seed_from_gt_debug:
        print("\nWARNING: --seed-from-gt-debug uses ground truth and is not a final method.")

    rows = []
    failures = []
    show_results = not args.no_show

    for patient_id in cfg.patients:
        print(f"\n=== {patient_id} ===")
        try:
            row = process_patient(patient_id, cfg, show_results, seed_cache, args)
            rows.append(row)
            save_seed_cache(seed_cache_path, seed_cache)
            print(
                f"  EF={row['lvef_pct']}%  wall={row['diastolic_wall_mean_mm']}mm  "
                f"Dice LV ED/ES={row['dice_lv_ed']}/{row['dice_lv_es']}  "
                f"Dice MYO ED/ES={row['dice_myo_ed']}/{row['dice_myo_es']}"
            )
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            failures.append((patient_id, str(exc)))
            traceback.print_exc()

    if not rows:
        print("\nNo rows were produced.")
        return 1

    csv_path = cfg.paths.output_folder / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_summary(
        rows, save_path=cfg.paths.output_folder / "summary_scatter.png"
    )

    print(f"\nWrote {csv_path}")
    print(f"Saved/reused manual seeds in {seed_cache_path}")
    print(f"\nProcessed {len(rows)} patients successfully.")
    if failures:
        print(f"{len(failures)} patient(s) FAILED:")
        for pid, msg in failures:
            print(f"  {pid}: {msg}")
    print(f"Figures in {cfg.paths.output_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
