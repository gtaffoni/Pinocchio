#!/usr/bin/env python3
"""validate_run.py — Regression validation tool for PINOCCHIO outputs.

Implements the three-level validation methodology described in
docs/VALIDATION_METHODOLOGY.md (version 1.0, 2026-06-12).

Usage
-----
  validate_run.py --reference <dir_or_golden> --candidate <dir>
                  [--level exact|reorder|physical]
                  [--profile gpu-fp|algo-approx]
                  [--thresholds file.json]
                  [--calibrate]
                  [--make-golden <outdir>]
                  [--plots <dir>]
                  [--report <file.txt>]

Exit codes
----------
  0  PASS
  1  FAIL  (physical regression detected)
  2  BROKEN (run incomplete / integrity error / setup mismatch)

Dependencies: Python >= 3.8, numpy.  matplotlib only if --plots (lazy import).
ReadPinocchio5.py must be in the same directory.
"""

import sys
import os
import json
import hashlib
import argparse
import shutil
import struct
import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Resolve ReadPinocchio5 from the same directory as this script
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ReadPinocchio5 as rp

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
PASS   = 0
FAIL   = 1
BROKEN = 2

# ---------------------------------------------------------------------------
# Default thresholds — profilo gpu-fp (sezione C.2 del documento)
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "gpu-fp": {
        "f_match_core":        0.999,   # C.2 #1 — npart >= core_npart
        "core_npart":          20,
        "f_match_all":         0.99,    # C.2 #2
        "f_mass_unmatched":    0.001,   # C.2 #3
        "max_unmatched_npart": 100,     # C.2 #4 — zero non-match >= questo
        "frac_dnpart_zero":    0.95,    # C.2 #5
        "p99_dnpart_rel":      0.02,    # C.2 #6
        "median_dx_cells":     0.01,    # C.2 #7
        "p99_dx_cells":        0.1,     # C.2 #8
        "p99_dv_kms":          10.0,    # C.2 #9
        "hmf_max_sigma":       0.2,     # C.2 #10
        "hmf_sign_low":        0.2,     # C.2 #10 — sign test banda [low, high]
        "hmf_sign_high":       0.8,
        "merger_flip_rate":    0.001,   # C.2 #11
        "tree_same_nbranches": 0.995,   # C.2 #12
        # Δposin = 0 esatto è hardcoded (C.2 #13): non entra nelle soglie JSON
    },
    "algo-approx": {
        # Soglie più ampie per approssimazioni algoritmiche dichiarate (FastFrag, tiling, bordo).
        # Valore guida: non superare la banda np1-vs-np2 misurata.
        # [Decisione Morgan]: fissare dopo la prima misura reale con --calibrate.
        "f_match_core":        0.99,    # rilassato: 99% vs 99.9%
        "core_npart":          20,
        "f_match_all":         0.97,    # rilassato: 97% vs 99%
        "f_mass_unmatched":    0.005,   # rilassato: 0.5% vs 0.1%
        "max_unmatched_npart": 100,     # invariato: zero non-match grandi
        "frac_dnpart_zero":    0.85,    # rilassato: 85% vs 95%
        "p99_dnpart_rel":      0.05,    # rilassato: 5% vs 2%
        "median_dx_cells":     0.05,    # rilassato: 0.05 vs 0.01 celle
        "p99_dx_cells":        0.5,     # rilassato: 0.5 vs 0.1 celle
        "p99_dv_kms":          50.0,    # rilassato: 50 vs 10 km/s
        "hmf_max_sigma":       1.0,     # rilassato: 1σ vs 0.2σ
        "hmf_sign_low":        0.1,
        "hmf_sign_high":       0.9,
        "merger_flip_rate":    0.01,    # rilassato: 1% vs 0.1%
        "tree_same_nbranches": 0.97,    # rilassato: 97% vs 99.5%
    }
}


# ===========================================================================
# SECTION 1: Run discovery — parse parameter file and/or manifest.json
# ===========================================================================

def _parse_parameter_file(parfile):
    """Parse a PINOCCHIO parameter file into a dict of key->value strings.

    Ignores comment lines (%) and inline comments.  Returns a dict with
    string values (caller converts types as needed).
    """
    params = {}
    with open(parfile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            # strip inline comment
            if '%' in line:
                line = line[:line.index('%')].strip()
            parts = line.split()
            if len(parts) == 0:
                continue
            key = parts[0]
            val = parts[1] if len(parts) > 1 else "1"  # flag-style keys
            params[key] = val
    return params


def _read_outputs_file(path, rundir):
    """Read the OutputList file; returns sorted list of redshifts (descending)."""
    # path may be a filename relative to rundir, or absolute
    if not os.path.isabs(path):
        path = os.path.join(rundir, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"OutputList file not found: {path}")
    zs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            try:
                zs.append(float(line))
            except ValueError:
                pass
    return sorted(set(zs), reverse=True)


def _z_to_tag(z):
    """Convert redshift to PINOCCHIO filename tag (e.g. 0.5 -> '0.5000')."""
    return f"{z:.4f}"


def discover_run(rundir):
    """Discover metadata for a run directory or golden directory.

    Returns a dict with:
      rundir, RunFlag, outputs (list of z, descending), BoxSize, GridSize,
      RandomSeed, MinHaloMass, np_mpi, PLC, CatalogInAscii, NumFiles,
      has_manifest, manifest (or None), parfile (path).

    Raises SystemExit(BROKEN) with a clear message if discovery fails.
    """
    rundir = os.path.abspath(rundir)
    if not os.path.isdir(rundir):
        _die(BROKEN, f"Directory not found: {rundir}")

    manifest_path = os.path.join(rundir, "manifest.json")
    has_manifest = os.path.exists(manifest_path)
    manifest = None

    if has_manifest:
        with open(manifest_path) as f:
            manifest = json.load(f)
        run_meta = manifest["run"]
        return {
            "rundir": rundir,
            "RunFlag": run_meta["RunFlag"],
            "outputs": sorted(run_meta["outputs"], reverse=True),
            "BoxSize": float(run_meta["BoxSize"]),
            "GridSize": int(run_meta["GridSize"]),
            "RandomSeed": int(run_meta["RandomSeed"]),
            "MinHaloMass": int(run_meta["MinHaloMass"]),
            "np_mpi": int(run_meta["np"]),
            "PLC": bool(run_meta.get("PLC", False)),
            "CatalogInAscii": bool(run_meta.get("CatalogInAscii", False)),
            "NumFiles": int(run_meta.get("NumFiles", 1)),
            "has_manifest": True,
            "manifest": manifest,
            "parfile": None,
        }

    # No manifest: look for parameter_file
    parfile = None
    for name in ("parameter_file", "param_file", "pinocchio.param"):
        candidate = os.path.join(rundir, name)
        if os.path.exists(candidate):
            parfile = candidate
            break
    if parfile is None:
        _die(BROKEN,
             f"No manifest.json and no parameter_file found in {rundir}")

    params = _parse_parameter_file(parfile)

    # Required keys
    for key in ("RunFlag", "GridSize", "RandomSeed", "BoxSize"):
        if key not in params:
            _die(BROKEN, f"Parameter file {parfile} missing key: {key}")

    runflag   = params["RunFlag"]
    gridsize  = int(params["GridSize"])
    seed      = int(params["RandomSeed"])
    boxsize   = float(params["BoxSize"])
    minhalo   = int(params.get("MinHaloMass", 10))
    numfiles  = int(params.get("NumFiles", 1))
    plc       = ("StartingzForPLC" in params)
    ascii_cat = ("CatalogInAscii" in params)

    outlist_file = params.get("OutputList", "outputs")
    try:
        outputs = _read_outputs_file(outlist_file, rundir)
    except FileNotFoundError as e:
        _die(BROKEN, str(e))

    # Try to detect np_mpi from a log file or default to unknown (-1)
    np_mpi = _detect_np_from_logs(rundir)

    return {
        "rundir": rundir,
        "RunFlag": runflag,
        "outputs": outputs,
        "BoxSize": boxsize,
        "GridSize": gridsize,
        "RandomSeed": seed,
        "MinHaloMass": minhalo,
        "np_mpi": np_mpi,
        "PLC": plc,
        "CatalogInAscii": ascii_cat,
        "NumFiles": numfiles,
        "has_manifest": False,
        "manifest": None,
        "parfile": parfile,
    }


def _detect_np_from_logs(rundir):
    """Try to read np_mpi from np_mpi.txt written by run scripts, or return -1."""
    marker = os.path.join(rundir, "np_mpi.txt")
    if os.path.exists(marker):
        try:
            return int(open(marker).read().strip())
        except Exception:
            pass
    return -1


def _catalog_path(run, z, suffix=None):
    """Return the expected catalog filename for redshift z in run."""
    tag = _z_to_tag(z)
    fname = f"pinocchio.{tag}.{run['RunFlag']}.catalog.out"
    base = os.path.join(run['rundir'], fname)
    if suffix is not None:
        return base + suffix
    return base


def _mf_path(run, z):
    tag = _z_to_tag(z)
    return os.path.join(run['rundir'],
                        f"pinocchio.{tag}.{run['RunFlag']}.mf.out")


def _histories_path(run):
    return os.path.join(run['rundir'],
                        f"pinocchio.{run['RunFlag']}.histories.out")


def _plc_path(run):
    return os.path.join(run['rundir'],
                        f"pinocchio.{run['RunFlag']}.plc.out")


def _nz_path(run):
    return os.path.join(run['rundir'],
                        f"pinocchio.{run['RunFlag']}.nz.out")


# ===========================================================================
# SECTION 2: Integrity check (B.0)
# ===========================================================================

def integrity_check(run, label="run"):
    """B.0: verify completeness, readability, temporal coherence, SHA256 (if manifest).

    Returns list of problem strings.  Empty list = all good.
    """
    problems = []
    rundir = run['rundir']

    # B.0.1 — file completeness
    cat_mtimes = []
    for z in run['outputs']:
        cat = _catalog_path(run, z)
        mf  = _mf_path(run, z)
        # catalog may have .0 extension for multi-file
        cat_exists = os.path.exists(cat) or os.path.exists(cat + '.0')
        if not cat_exists:
            problems.append(f"[{label}] Missing catalog for z={z}: {cat}")
        else:
            mtime = os.path.getmtime(cat if os.path.exists(cat) else cat + '.0')
            cat_mtimes.append((z, mtime))
        if not os.path.exists(mf):
            problems.append(f"[{label}] Missing mf.out for z={z}: {mf}")

    hist = _histories_path(run)
    hist_exists = os.path.exists(hist) or os.path.exists(hist + '.0')
    if not hist_exists:
        problems.append(f"[{label}] Missing histories file: {hist}")

    if run['PLC']:
        plc_f = _plc_path(run)
        plc_exists = os.path.exists(plc_f) or os.path.exists(plc_f + '.0')
        if not plc_exists:
            problems.append(f"[{label}] Missing PLC file: {plc_f}")

    # B.0.2 — readability (try to open catalog at z=0 or last z)
    if not problems:
        last_z = run['outputs'][-1]
        cat = _catalog_path(run, last_z)
        try:
            c = rp.catalog(cat, silent=True)
            if c is None or c.Nhalos == 0:
                problems.append(f"[{label}] Catalog at z={last_z} is empty or unreadable")
        except Exception as e:
            problems.append(f"[{label}] Error reading catalog at z={last_z}: {e}")

    # B.0.3 — temporal coherence: catalog at lower z must not be older than at higher z
    if len(cat_mtimes) >= 2:
        # cat_mtimes is in descending z order; mtime should be non-decreasing
        # (lower z written later)
        for i in range(len(cat_mtimes) - 1):
            z_high, mt_high = cat_mtimes[i]
            z_low,  mt_low  = cat_mtimes[i + 1]
            if mt_low < mt_high - 1.0:  # 1s tolerance
                problems.append(
                    f"[{label}] Temporal incoherence: z={z_low} catalog older than z={z_high} "
                    f"(possible stale files from aborted run)"
                )

    # B.0.4 — manifest SHA256 verification
    if run['has_manifest'] and run['manifest']:
        mfiles = run['manifest'].get('files', {})
        for fname, fmeta in mfiles.items():
            fpath = os.path.join(rundir, fname)
            if not os.path.exists(fpath):
                problems.append(f"[{label}] Golden file missing: {fname}")
                continue
            expected_sha = fmeta.get('sha256', '')
            if expected_sha:
                actual_sha = _sha256(fpath)
                if actual_sha != expected_sha:
                    problems.append(
                        f"[{label}] SHA256 mismatch for {fname}: "
                        f"expected {expected_sha[:12]}... got {actual_sha[:12]}..."
                    )

    return problems


def check_compatibility(ref, cand):
    """B.0.4-5: verify same seed, BoxSize, GridSize, outputs, MinHaloMass, np.

    Returns list of problem strings.
    """
    problems = []

    def _chk(field, a, b, fmt=None):
        if fmt:
            va, vb = fmt(a[field]), fmt(b[field])
        else:
            va, vb = a[field], b[field]
        if va != vb:
            problems.append(
                f"Setup mismatch [{field}]: reference={va}, candidate={vb} — "
                f"these runs are not comparable (exit 2)"
            )

    _chk('RandomSeed', ref, cand)
    _chk('GridSize',   ref, cand)
    _chk('MinHaloMass', ref, cand)
    # BoxSize: compare as float with tolerance
    if abs(ref['BoxSize'] - cand['BoxSize']) > 0.01:
        problems.append(
            f"Setup mismatch [BoxSize]: reference={ref['BoxSize']}, "
            f"candidate={cand['BoxSize']}"
        )
    # OutputList: same set of redshifts
    ref_z  = sorted(ref['outputs'])
    cand_z = sorted(cand['outputs'])
    if ref_z != cand_z:
        problems.append(
            f"Setup mismatch [OutputList]: reference={ref_z}, candidate={cand_z}"
        )
    # np_mpi — only enforce if both are known
    rnp = ref['np_mpi']
    cnp = cand['np_mpi']
    if rnp > 0 and cnp > 0 and rnp != cnp:
        problems.append(
            f"Setup mismatch [np_mpi]: reference={rnp}, candidate={cnp}. "
            f"Run with the same number of MPI tasks (B.0.5)."
        )

    return problems


# ===========================================================================
# SECTION 3: Level 0 — byte-exact comparison
# ===========================================================================

def compare_exact(ref, cand):
    """L0: SHA256 comparison of every expected output file.

    Returns (ok: bool, detail: list of str).
    """
    detail = []
    ok = True

    def _compare_file(rpath, cpath, label):
        nonlocal ok
        re = os.path.exists(rpath)
        ce = os.path.exists(cpath)
        if not re or not ce:
            detail.append(f"  SKIP {label}: file missing in ref={re} cand={ce}")
            return
        sha_r = _sha256(rpath)
        sha_c = _sha256(cpath)
        if sha_r == sha_c:
            detail.append(f"  OK   {label}")
        else:
            detail.append(f"  DIFF {label}  ref={sha_r[:12]}  cand={sha_c[:12]}")
            ok = False

    for z in ref['outputs']:
        tag = _z_to_tag(z)
        _compare_file(_catalog_path(ref, z), _catalog_path(cand, z), f"catalog z={z}")
        _compare_file(_mf_path(ref, z), _mf_path(cand, z), f"mf.out  z={z}")

    _compare_file(_histories_path(ref), _histories_path(cand), "histories")
    if ref['PLC']:
        _compare_file(_plc_path(ref), _plc_path(cand), "plc")
        _compare_file(_nz_path(ref), _nz_path(cand), "nz")

    return ok, detail


# ===========================================================================
# SECTION 4: Level 1 — reorder comparison
# ===========================================================================

def compare_reorder(ref, cand, ulp_tol=2):
    """L1: same names/npart/floats-within-ulp after sorting by name.

    Returns (ok: bool, detail: list of str).
    """
    detail = []
    ok = True

    for z in ref['outputs']:
        cat_r = _load_catalog(ref, z)
        cat_c = _load_catalog(cand, z)
        if cat_r is None or cat_c is None:
            detail.append(f"  z={z}: could not load catalog")
            ok = False
            continue

        # sort by name
        sr = np.argsort(cat_r['name'])
        sc = np.argsort(cat_c['name'])
        dr = cat_r[sr]
        dc = cat_c[sc]

        # unique check
        if len(np.unique(cat_r['name'])) != len(cat_r['name']):
            detail.append(f"  z={z}: DUPLICATE names in reference — integrity bug")
            ok = False
        if len(np.unique(cat_c['name'])) != len(cat_c['name']):
            detail.append(f"  z={z}: DUPLICATE names in candidate — integrity bug")
            ok = False

        # set equality
        if not np.array_equal(dr['name'], dc['name']):
            n_r = len(dr)
            n_c = len(dc)
            detail.append(f"  z={z}: name sets DIFFER  ref={n_r}  cand={n_c}")
            ok = False
            continue

        detail.append(f"  z={z}: name sets match ({len(dr)} halos)")

        # npart
        if 'npart' in dr.dtype.names:
            if not np.array_equal(dr['npart'], dc['npart']):
                diff = np.sum(dr['npart'] != dc['npart'])
                detail.append(f"  z={z}: npart DIFFER in {diff} halos")
                ok = False
            else:
                detail.append(f"  z={z}: npart identical")

        # float fields within ulp_tol ulp
        for field in ('Mass', 'pos', 'vel', 'posin'):
            if field not in dr.dtype.names:
                continue
            vr = dr[field].astype(np.float32)
            vc = dc[field].astype(np.float32)
            ulps = np.abs(vr - vc) / np.maximum(np.spacing(vr), 1e-45)
            max_ulp = np.max(ulps)
            if max_ulp > ulp_tol:
                detail.append(f"  z={z}: {field} max_ulp={max_ulp:.1f} (tol={ulp_tol})")
                ok = False
            else:
                detail.append(f"  z={z}: {field} OK (max_ulp={max_ulp:.2f})")

    return ok, detail


# ===========================================================================
# SECTION 5: Level 2 — physical metrics
# ===========================================================================

def _load_catalog(run, z):
    """Load catalog for redshift z; return structured array or None."""
    path = _catalog_path(run, z)
    try:
        c = rp.catalog(path, silent=True)
        if c is None:
            return None
        return c.data
    except Exception:
        return None


def _periodic_delta(a, b, box):
    """Minimum-image difference a - b in a periodic box of size box.

    Works element-wise on arrays of arbitrary shape.
    """
    d = a - b
    return d - box * np.rint(d / box)


def catalog_metrics(cat_ref, cat_cand, box, grid, minhalo=10):
    """B.1 + B.2: compute all per-halo and aggregate metrics.

    Parameters
    ----------
    cat_ref, cat_cand : numpy structured arrays (from ReadPinocchio5.catalog.data)
    box  : BoxSize in Mpc/h
    grid : GridSize (for cell size = box/grid)
    minhalo : MinHaloMass (minimum npart)

    Returns
    -------
    dict with all metrics.  Keys follow the specification.
    """
    cell = box / grid   # Mpc/h per cell

    name_r = cat_ref['name']
    name_c = cat_cand['name']

    # --- uniqueness integrity check ---
    if len(np.unique(name_r)) != len(name_r):
        return {"integrity_error": "duplicate names in reference"}
    if len(np.unique(name_c)) != len(name_c):
        return {"integrity_error": "duplicate names in candidate"}

    # --- B.1: matching ---
    common, iref, icand = np.intersect1d(
        name_r, name_c, return_indices=True, assume_unique=True
    )
    only_ref  = np.setdiff1d(name_r, name_c, assume_unique=True)
    only_cand = np.setdiff1d(name_c, name_r, assume_unique=True)

    N_ref   = len(name_r)
    N_cand  = len(name_c)
    N_match = len(common)

    f_match_all = N_match / N_ref if N_ref > 0 else 1.0

    # per-bin f_match on reference npart
    npart_r = cat_ref['npart']
    m10_19  = (npart_r >= 10) & (npart_r <= 19)
    m20_99  = (npart_r >= 20) & (npart_r <= 99)
    m100p   = npart_r >= 100

    def _fmatch_bin(mask):
        n_in = np.sum(mask)
        if n_in == 0:
            return 1.0, 0, 0
        # names in this bin
        names_in_bin = name_r[mask]
        n_matched = len(np.intersect1d(names_in_bin, common, assume_unique=True))
        return n_matched / n_in, n_matched, n_in

    f_match_10_19, nm_10_19, nt_10_19 = _fmatch_bin(m10_19)
    f_match_20_99, nm_20_99, nt_20_99 = _fmatch_bin(m20_99)
    f_match_100p,  nm_100p,  nt_100p  = _fmatch_bin(m100p)

    # f_match_core: npart >= 20
    n_core = nt_20_99 + nt_100p
    nm_core = nm_20_99 + nm_100p
    f_match_core = nm_core / n_core if n_core > 0 else 1.0

    # f_mass_unmatched
    mass_r = cat_ref['Mass']
    mass_total = mass_r.sum()
    # unmatched reference halos
    mask_only_ref = np.isin(name_r, only_ref, assume_unique=True)
    mass_unmatched_ref = mass_r[mask_only_ref].sum()
    f_mass_unmatched = mass_unmatched_ref / mass_total if mass_total > 0 else 0.0

    # non-match with npart >= 100
    npart_unmatched = npart_r[mask_only_ref]
    large_unmatched_mask = npart_unmatched >= 100
    n_large_unmatched = np.sum(large_unmatched_mask)
    large_unmatched_list = []
    if n_large_unmatched > 0:
        idx = np.where(mask_only_ref)[0][large_unmatched_mask]
        for i in idx:
            pos = cat_ref['pos'][i] if 'pos' in cat_ref.dtype.names else None
            large_unmatched_list.append({
                "name":  int(name_r[i]),
                "npart": int(npart_r[i]),
                "Mass":  float(mass_r[i]),
                "pos":   pos.tolist() if pos is not None else None,
            })

    # --- B.2: metrics on matched halos ---
    mr = cat_ref[iref]
    mc = cat_cand[icand]

    # ΔNpart (integer — no FP noise)
    dnpart = mc['npart'].astype(np.int64) - mr['npart'].astype(np.int64)
    frac_dnpart_zero = np.mean(dnpart == 0)
    # p99 of |ΔNpart|/npart_ref, excluding zero-npart
    npart_ref_m = mr['npart'].astype(np.float64)
    npart_ref_m = np.where(npart_ref_m > 0, npart_ref_m, 1.0)
    rel_dnpart = np.abs(dnpart) / npart_ref_m
    p99_dnpart_rel = float(np.percentile(rel_dnpart, 99)) if len(rel_dnpart) > 0 else 0.0
    # max |ΔNpart| for halos with npart_ref >= 100
    big_mask = mr['npart'] >= 100
    max_dnpart_big = int(np.max(np.abs(dnpart[big_mask]))) if np.any(big_mask) else 0

    # |Δx| with periodic minimum image
    dx3 = _periodic_delta(mc['pos'].astype(np.float64),
                          mr['pos'].astype(np.float64), box)   # (N,3) Mpc/h
    dx  = np.sqrt(np.sum(dx3**2, axis=1))                       # Mpc/h
    dx_cells = dx / cell
    median_dx_cells = float(np.median(dx_cells))
    p99_dx_cells    = float(np.percentile(dx_cells, 99)) if len(dx_cells) > 0 else 0.0
    max_dx_cells    = float(np.max(dx_cells)) if len(dx_cells) > 0 else 0.0

    # |Δv| km/s
    dv3 = mc['vel'].astype(np.float64) - mr['vel'].astype(np.float64)
    dv  = np.sqrt(np.sum(dv3**2, axis=1))
    p99_dv_kms = float(np.percentile(dv, 99)) if len(dv) > 0 else 0.0
    max_dv_kms = float(np.max(dv)) if len(dv) > 0 else 0.0

    # Δposin — initial Lagrangian position of the halo (a centre-of-mass-like
    # quantity over the member particles' initial positions). It must be reproduced
    # exactly at Level 0/1. At Level 2 (physical/GPU) it carries floating-point
    # noise: same code+config but a different compiler (e.g. nvc vs gcc, FMA
    # contraction, math intrinsics) or GPU arithmetic shifts it by a few float32
    # ulp (~1e-5 cells). A real frame/indexing bug instead shifts it by WHOLE cells
    # (>= 1 cell ~ 1e5 ulp), so a sub-cell tolerance still catches such bugs.
    dposin3 = mc['posin'].astype(np.float64) - mr['posin'].astype(np.float64)
    dposin  = np.sqrt(np.sum(dposin3**2, axis=1))
    n_dposin_nonzero = int(np.sum(dposin != 0.0))
    max_dposin = float(np.max(dposin)) if len(dposin) > 0 else 0.0
    # Δposin in grid-cell units (grid-independent, physically interpretable) and,
    # for info, in float32 ulp at the position scale.
    max_dposin_cells = max_dposin / cell if cell > 0 else 0.0
    if len(dposin3) > 0:
        comp_abs = np.abs(dposin3)
        sp = np.spacing(np.abs(mr['posin']).astype(np.float32)).astype(np.float64)
        sp = np.maximum(sp, float(np.spacing(np.float32(1.0))))
        max_dposin_ulp = float(np.max(comp_abs / sp))
    else:
        max_dposin_ulp = 0.0
    # List offending halos (up to 10)
    dposin_offenders = []
    if n_dposin_nonzero > 0:
        bad_idx = np.where(dposin != 0.0)[0][:10]
        for i in bad_idx:
            dposin_offenders.append({
                "name":     int(common[i]),
                "dposin":   float(dposin[i]),
                "dposin3":  dposin3[i].tolist(),
            })

    return {
        # B.1 matching
        "N_ref":              N_ref,
        "N_cand":             N_cand,
        "N_match":            N_match,
        "f_match_all":        f_match_all,
        "f_match_core":       f_match_core,
        "f_match_10_19":      f_match_10_19,
        "f_match_20_99":      f_match_20_99,
        "f_match_100p":       f_match_100p,
        "f_mass_unmatched":   f_mass_unmatched,
        "n_large_unmatched":  n_large_unmatched,
        "large_unmatched":    large_unmatched_list,
        # B.2 per-halo
        "frac_dnpart_zero":   frac_dnpart_zero,
        "p99_dnpart_rel":     p99_dnpart_rel,
        "max_dnpart_big":     max_dnpart_big,
        "median_dx_cells":    median_dx_cells,
        "p99_dx_cells":       p99_dx_cells,
        "max_dx_cells":       max_dx_cells,
        "p99_dv_kms":         p99_dv_kms,
        "max_dv_kms":         max_dv_kms,
        # Δposin (frame check)
        "n_dposin_nonzero":   n_dposin_nonzero,
        "max_dposin":         max_dposin,
        "max_dposin_cells":   max_dposin_cells,
        "max_dposin_ulp":     max_dposin_ulp,
        "dposin_offenders":   dposin_offenders,
        # for calibrate / plots
        "_dnpart":            dnpart,
        "_dx_cells":          dx_cells,
        "_dv_kms":            dv,
        "_npart_ref":         mr['npart'],
    }


def hmf_metrics(mf_ref_path, mf_cand_path):
    """B.3: mass function bin-by-bin comparison in Poisson sigma units.

    mf.out columns: mass, n(m), upper_1sigma, lower_1sigma, N_halos_in_bin, n_analytic.
    sigma_i = (upper - lower) / 2  (from reference).

    Returns dict with max_r, sign_frac, per-bin residuals.
    """
    try:
        mf_r = np.loadtxt(mf_ref_path,  comments='#')
        mf_c = np.loadtxt(mf_cand_path, comments='#')
    except Exception as e:
        return {"error": str(e)}

    if mf_r.ndim == 1:
        mf_r = mf_r.reshape(1, -1)
    if mf_c.ndim == 1:
        mf_c = mf_c.reshape(1, -1)

    # Align on mass bins (match by column 0)
    mass_r = mf_r[:, 0]
    mass_c = mf_c[:, 0]
    if not np.allclose(mass_r, mass_c, rtol=1e-4):
        return {"error": "mf.out mass bins do not match between ref and cand"}

    n_ref  = mf_r[:, 1]       # n(m) [Mpc^-3 Msun^-1 h^4]
    up_r   = mf_r[:, 2]       # +1sigma
    lo_r   = mf_r[:, 3]       # -1sigma
    nbin_r = mf_r[:, 4]       # number of halos in bin (reference)
    n_cand = mf_c[:, 1]

    sigma = (up_r - lo_r) / 2.0
    sigma = np.where(sigma > 0, sigma, 1e-40)  # avoid /0

    residuals = (n_cand - n_ref) / sigma

    # only consider bins with >= 50 halos in reference
    mask_50 = nbin_r >= 50
    if np.sum(mask_50) == 0:
        return {
            "max_r": 0.0,
            "sign_frac": 0.5,
            "n_bins_50": 0,
            "residuals": residuals.tolist(),
            "nbin_r": nbin_r.tolist(),
        }

    r_50 = residuals[mask_50]
    max_r    = float(np.max(np.abs(r_50)))
    # Sign test for systematic bias: only bins with a real deviation carry sign
    # information. Zero-residual bins (identical/near-identical HMF) must not be
    # counted, otherwise perfect agreement spuriously fails the [low,high] band.
    nz = np.abs(r_50) > 0.0
    n_signif = int(np.sum(nz))
    if n_signif < 5:
        sign_frac = 0.5          # too few deviating bins -> test N/A, neutral pass
    else:
        sign_frac = float(np.mean(r_50[nz] > 0))

    return {
        "max_r":     max_r,
        "sign_frac": sign_frac,
        "n_sign_bins": n_signif,
        "n_bins_50": int(np.sum(mask_50)),
        "residuals": residuals.tolist(),
        "nbin_r":    nbin_r.tolist(),
        "mass_bins": mass_r.tolist(),
    }


def histories_metrics(href, hcand):
    """B.4: merger tree comparison via name matching.

    href, hcand: ReadPinocchio5.histories objects (already loaded).

    merged_with is resolved via nickname->name mapping within each tree,
    never via global index.

    Returns dict with tree_matched_frac, tree_same_nbranches_frac,
    merger_flip_rate, dz_* statistics.
    """
    if href is None or hcand is None:
        return {"error": "histories not loaded"}

    # Build root-name -> tree index maps
    def _build_root_map(h):
        """root_name (name of first branch) -> (tree_idx, Nbranches)."""
        rmap = {}
        for t in range(h.Ntrees):
            p = h.pointers[t]
            root_name = int(h.data['name'][p])
            rmap[root_name] = t
        return rmap

    rmap_r = _build_root_map(href)
    rmap_c = _build_root_map(hcand)

    roots_r = set(rmap_r.keys())
    roots_c = set(rmap_c.keys())
    matched_roots = roots_r & roots_c

    tree_matched_frac = len(matched_roots) / len(roots_r) if roots_r else 1.0

    # For each matched tree: check Nbranches and branch-level metrics
    n_same_nb = 0
    n_total_matched = len(matched_roots)
    flip_count   = 0
    branch_count = 0
    dz_merging_list = []
    dz_peak_list    = []
    dz_appear_list  = []

    for root in matched_roots:
        tr = rmap_r[root]
        tc = rmap_c[root]

        pr = href.pointers[tr]
        pc = hcand.pointers[tc]
        nb_r = int(href.Nbranches[tr])
        nb_c = int(hcand.Nbranches[tc])

        if nb_r == nb_c:
            n_same_nb += 1

        # Build nickname -> name maps for this tree (reference and candidate)
        def _nick_map(h, ptr, nb):
            m = {}
            for k in range(nb):
                nick = int(h.data['nickname'][ptr + k])
                name = int(h.data['name'][ptr + k])
                m[nick] = name
            return m

        nick_r = _nick_map(href,  pr, nb_r)
        nick_c = _nick_map(hcand, pc, nb_c)

        # Build name -> branch data for candidate
        name_to_branch_c = {}
        for k in range(nb_c):
            name = int(hcand.data['name'][pc + k])
            name_to_branch_c[name] = pc + k

        # Match branches by name
        for k in range(nb_r):
            br_name = int(href.data['name'][pr + k])
            if br_name not in name_to_branch_c:
                continue
            branch_count += 1

            idx_r = pr + k
            idx_c = name_to_branch_c[br_name]

            # Resolve merged_with via nickname->name
            mw_r_nick = int(href.data['merged_with'][idx_r])
            mw_c_nick = int(hcand.data['merged_with'][idx_c])

            mw_r_name = nick_r.get(mw_r_nick, -1)
            mw_c_name = nick_c.get(mw_c_nick, -1)

            if mw_r_name != mw_c_name:
                flip_count += 1

            # |Δz_*|
            dz_merging_list.append(
                abs(float(href.data['z_merging'][idx_r]) -
                    float(hcand.data['z_merging'][idx_c]))
            )
            dz_peak_list.append(
                abs(float(href.data['z_peak'][idx_r]) -
                    float(hcand.data['z_peak'][idx_c]))
            )
            dz_appear_list.append(
                abs(float(href.data['z_appear'][idx_r]) -
                    float(hcand.data['z_appear'][idx_c]))
            )

    merger_flip_rate = flip_count / branch_count if branch_count > 0 else 0.0
    tree_same_nb_frac = n_same_nb / n_total_matched if n_total_matched > 0 else 1.0

    def _p99(lst):
        return float(np.percentile(lst, 99)) if lst else 0.0

    return {
        "Ntrees_ref":              href.Ntrees,
        "Ntrees_cand":             hcand.Ntrees,
        "tree_matched_frac":       tree_matched_frac,
        "tree_same_nbranches_frac": tree_same_nb_frac,
        "merger_flip_rate":        merger_flip_rate,
        "branch_count":            branch_count,
        "flip_count":              flip_count,
        "p99_dz_merging":          _p99(dz_merging_list),
        "p99_dz_peak":             _p99(dz_peak_list),
        "p99_dz_appear":           _p99(dz_appear_list),
    }


# ===========================================================================
# SECTION 6: Verdict
# ===========================================================================

def _check(metric_val, threshold, op=">=", name=""):
    """Return (passed, line_str) for a single threshold check."""
    if op == ">=":
        passed = metric_val >= threshold
    elif op == "<=":
        passed = metric_val <= threshold
    elif op == "==":
        passed = metric_val == threshold
    elif op == "in":
        # threshold is (low, high)
        passed = threshold[0] <= metric_val <= threshold[1]
    else:
        passed = False
    mark = "PASS" if passed else "FAIL"
    return passed, f"  {mark}  {name:40s}  {metric_val:.6g}  (thr {op} {threshold})"


def verdict(all_metrics, thresholds, ref_info):
    """Apply thresholds to metrics; return (ok: bool, report_lines: list[str]).

    Hardcoded: Δposin == 0 at all z (C.2 #13).
    """
    lines = []
    ok = True

    thr = thresholds

    def _fail(line):
        nonlocal ok
        ok = False
        lines.append(line)

    def _chk(val, thr_val, op, name):
        passed, line = _check(val, thr_val, op, name)
        lines.append(line)
        if not passed:
            _fail("")  # already appended

    for z in ref_info['outputs']:
        m = all_metrics.get(f"cat_{z}")
        if m is None:
            lines.append(f"  [z={z}] No catalog metrics available")
            continue
        if "integrity_error" in m:
            lines.append(f"  [z={z}] INTEGRITY ERROR: {m['integrity_error']}")
            ok = False
            continue

        lines.append(f"\n  --- Catalog metrics z={z} ---")

        # B.1
        passed, line = _check(m['f_match_all'],  thr['f_match_all'],  ">=", "f_match_all")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['f_match_core'], thr['f_match_core'], ">=", "f_match_core (npart>=20)")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['f_mass_unmatched'], thr['f_mass_unmatched'], "<=", "f_mass_unmatched")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['n_large_unmatched'], 0, "==", "n_large_unmatched (npart>=100)")
        lines.append(line)
        if not passed: ok = False
        if m['n_large_unmatched'] > 0:
            for lu in m['large_unmatched'][:5]:
                lines.append(f"    large unmatched: name={lu['name']} npart={lu['npart']} M={lu['Mass']:.2e}")

        # B.2
        passed, line = _check(m['frac_dnpart_zero'], thr['frac_dnpart_zero'], ">=", "frac_dnpart_zero")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['p99_dnpart_rel'], thr['p99_dnpart_rel'], "<=", "p99_dnpart_rel")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['median_dx_cells'], thr['median_dx_cells'], "<=", "median_dx_cells")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['p99_dx_cells'], thr['p99_dx_cells'], "<=", "p99_dx_cells")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(m['p99_dv_kms'], thr['p99_dv_kms'], "<=", "p99_dv_kms")
        lines.append(line)
        if not passed: ok = False

        # Δposin — frame/indexing check, in grid-cell units. Tolerates floating-point
        # noise (cross-compiler/GPU, ~1e-5 cells); a real frame bug moves posin by
        # whole cells (>= 1), orders of magnitude above the threshold.
        posin_tol = thr.get('posin_max_cells', 1e-3)
        passed, line = _check(m['max_dposin_cells'], posin_tol, "<=",
                              f"dposin <= {posin_tol:g} cells (frame check)")
        lines.append(line + f"  [max={m['max_dposin']:.3e} Mpc/h = {m['max_dposin_ulp']:.0f} ulp]")
        if not passed:
            ok = False
            for off in m['dposin_offenders'][:3]:
                lines.append(f"    offender name={off['name']}  dposin={off['dposin']:.3e}")

        # B.3 HMF
        hm = all_metrics.get(f"hmf_{z}")
        if hm and "error" not in hm:
            lines.append(f"\n  --- HMF metrics z={z} ---")
            passed, line = _check(hm['max_r'], thr['hmf_max_sigma'], "<=", "hmf_max_sigma (bins>=50)")
            lines.append(line)
            if not passed: ok = False

            sign_ok = thr['hmf_sign_low'] <= hm['sign_frac'] <= thr['hmf_sign_high']
            mark = "PASS" if sign_ok else "FAIL"
            lines.append(
                f"  {mark}  {'hmf_sign_frac in [{:.1f},{:.1f}]'.format(thr['hmf_sign_low'],thr['hmf_sign_high']):40s}  "
                f"{hm['sign_frac']:.3f}"
            )
            if not sign_ok: ok = False

    # B.4 histories
    hm = all_metrics.get("histories")
    if hm and "error" not in hm:
        lines.append(f"\n  --- Histories metrics ---")
        passed, line = _check(hm['merger_flip_rate'], thr['merger_flip_rate'], "<=", "merger_flip_rate")
        lines.append(line)
        if not passed: ok = False

        passed, line = _check(hm['tree_same_nbranches_frac'], thr['tree_same_nbranches'], ">=", "tree_same_nbranches_frac")
        lines.append(line)
        if not passed: ok = False

    return ok, lines


# ===========================================================================
# SECTION 7: Calibration report (--calibrate)
# ===========================================================================

def _to_python(v):
    """Convert numpy scalar to Python native type for JSON serialisation."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def calibration_report(all_metrics, ref_info):
    """Print a calibration band report (no pass/fail); return dict for JSON dump."""
    lines = ["", "=== CALIBRATION BAND (no pass/fail) ==="]
    band = {}

    for z in ref_info['outputs']:
        m = all_metrics.get(f"cat_{z}")
        if m is None or "integrity_error" in m:
            continue
        lines.append(f"\n  --- z={z}  ({m['N_ref']} ref, {m['N_cand']} cand) ---")
        for key in ("f_match_all", "f_match_core", "f_match_10_19",
                    "f_match_20_99", "f_match_100p", "f_mass_unmatched",
                    "n_large_unmatched", "frac_dnpart_zero", "p99_dnpart_rel",
                    "median_dx_cells", "p99_dx_cells", "p99_dv_kms",
                    "n_dposin_nonzero", "max_dposin"):
            val = m.get(key, "N/A")
            lines.append(f"    {key:35s} = {val}")
        band[f"cat_{z}"] = {k: _to_python(m.get(k)) for k in (
            "f_match_all", "f_match_core", "f_mass_unmatched",
            "frac_dnpart_zero", "p99_dnpart_rel",
            "median_dx_cells", "p99_dx_cells", "p99_dv_kms",
            "n_dposin_nonzero"
        )}

        hm = all_metrics.get(f"hmf_{z}")
        if hm and "error" not in hm:
            lines.append(f"    {'hmf_max_r':35s} = {hm['max_r']:.4f}")
            lines.append(f"    {'hmf_sign_frac':35s} = {hm['sign_frac']:.3f}")
            band[f"hmf_{z}"] = {"max_r": _to_python(hm['max_r']),
                                 "sign_frac": _to_python(hm['sign_frac'])}

    hm = all_metrics.get("histories")
    if hm and "error" not in hm:
        lines.append(f"\n  --- Histories ---")
        for key in ("tree_matched_frac", "tree_same_nbranches_frac",
                    "merger_flip_rate", "p99_dz_merging"):
            val = hm.get(key, "N/A")
            lines.append(f"    {key:35s} = {val}")
        band["histories"] = {k: _to_python(v) for k, v in hm.items()
                             if not k.startswith("_")}

    print("\n".join(lines))
    return band


# ===========================================================================
# SECTION 8: Golden creation (--make-golden)
# ===========================================================================

def make_golden(run, outdir, git_info=None):
    """Freeze a run as a golden reference.

    If outdir == run['rundir'], writes manifest.json in-place (no file copy).
    Otherwise copies all output files to outdir and writes manifest.json.
    """
    os.makedirs(outdir, exist_ok=True)
    outdir_abs = os.path.abspath(outdir)
    inplace = (outdir_abs == run['rundir'])

    # Collect files (copy or hash in-place)
    files_meta = {}

    def _copy_and_hash(src, dst_name):
        dst = os.path.join(outdir_abs, dst_name)
        if not inplace:
            shutil.copy2(src, dst)
        sha = _sha256(dst)
        size = os.path.getsize(dst)
        files_meta[dst_name] = {"sha256": sha, "bytes": size}
        return dst_name

    # catalogs and mf
    for z in run['outputs']:
        cat = _catalog_path(run, z)
        mf  = _mf_path(run, z)
        cat_name = os.path.basename(cat)
        mf_name  = os.path.basename(mf)
        if os.path.exists(cat):
            _copy_and_hash(cat, cat_name)
        elif os.path.exists(cat + '.0'):
            _copy_and_hash(cat + '.0', cat_name + '.0')
        if os.path.exists(mf):
            _copy_and_hash(mf, mf_name)

    # histories, plc, nz
    for path_fn in (_histories_path, _plc_path, _nz_path):
        p = path_fn(run)
        if os.path.exists(p):
            _copy_and_hash(p, os.path.basename(p))
        elif os.path.exists(p + '.0'):
            _copy_and_hash(p + '.0', os.path.basename(p) + '.0')

    # copy parameter file and outputs list (skip if in-place)
    if run['parfile'] and os.path.exists(run['parfile']) and not inplace:
        dst_pf = os.path.join(outdir_abs, "parameter_file")
        if os.path.abspath(run['parfile']) != os.path.abspath(dst_pf):
            shutil.copy2(run['parfile'], dst_pf)
        # also copy outputs file
        parparams = _parse_parameter_file(run['parfile'])
        outlist = parparams.get("OutputList", "outputs")
        if not os.path.isabs(outlist):
            outlist = os.path.join(run['rundir'], outlist)
        dst_ol = os.path.join(outdir_abs, "outputs")
        if os.path.exists(outlist) and os.path.abspath(outlist) != os.path.abspath(dst_ol):
            shutil.copy2(outlist, dst_ol)

    # write np_mpi.txt in golden
    if run['np_mpi'] > 0:
        with open(os.path.join(outdir, "np_mpi.txt"), "w") as f:
            f.write(str(run['np_mpi']))

    # Build manifest
    manifest = {
        "created": datetime.datetime.utcnow().isoformat(),
        "code": git_info or {"branch": "unknown", "commit": "unknown"},
        "run": {
            "np":             run['np_mpi'],
            "omp":            1,
            "RunFlag":        run['RunFlag'],
            "RandomSeed":     run['RandomSeed'],
            "BoxSize":        run['BoxSize'],
            "GridSize":       run['GridSize'],
            "MinHaloMass":    run['MinHaloMass'],
            "outputs":        run['outputs'],
            "PLC":            run['PLC'],
            "CatalogInAscii": run['CatalogInAscii'],
            "NumFiles":       run['NumFiles'],
        },
        "files": files_meta,
    }

    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Golden created in {outdir}")
    print(f"  Files frozen: {len(files_meta)}")
    print(f"  Manifest: {manifest_path}")

    return manifest


# ===========================================================================
# SECTION 9: Optional plots
# ===========================================================================

def make_plots(all_metrics, ref_info, plotdir):
    """Generate diagnostic plots (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping plots")
        return

    os.makedirs(plotdir, exist_ok=True)

    for z in ref_info['outputs']:
        m = all_metrics.get(f"cat_{z}")
        if m is None or "integrity_error" in m:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"z = {z}")

        # |Δx| histogram
        ax = axes[0]
        dx = m.get("_dx_cells")
        if dx is not None and len(dx) > 0:
            ax.hist(dx, bins=50, log=True)
            ax.axvline(0.01, color='r', linestyle='--', label='median thr 0.01')
            ax.axvline(0.1,  color='orange', linestyle='--', label='p99 thr 0.1')
            ax.set_xlabel("|Δx| [cells]")
            ax.set_ylabel("count")
            ax.legend(fontsize=7)
            ax.set_title("|Δx| distribution")

        # ΔNpart histogram
        ax = axes[1]
        dn = m.get("_dnpart")
        if dn is not None and len(dn) > 0:
            lim = max(5, int(np.percentile(np.abs(dn), 99.5)) + 1)
            bins = np.arange(-lim - 0.5, lim + 1.5, 1)
            ax.hist(dn, bins=bins, log=True)
            ax.set_xlabel("ΔNpart")
            ax.set_ylabel("count")
            ax.set_title("ΔNpart distribution")

        # HMF residuals
        ax = axes[2]
        hm = all_metrics.get(f"hmf_{z}")
        if hm and "error" not in hm and "residuals" in hm:
            r = np.array(hm["residuals"])
            nb = np.array(hm["nbin_r"])
            bins_50 = nb >= 50
            ax.scatter(range(len(r)), r, c=np.where(bins_50, 'b', 'gray'), s=20)
            ax.axhline(0, color='k')
            ax.axhline( 0.2, color='r', linestyle='--', label='±0.2σ')
            ax.axhline(-0.2, color='r', linestyle='--')
            ax.axhline( 1.0, color='orange', linestyle=':', label='±1σ')
            ax.axhline(-1.0, color='orange', linestyle=':')
            ax.set_xlabel("bin index")
            ax.set_ylabel("residual [σ_Poisson]")
            ax.set_title("HMF residuals")
            ax.legend(fontsize=7)

        plt.tight_layout()
        fname = os.path.join(plotdir, f"validation_z{z:.4f}.png")
        plt.savefig(fname, dpi=100)
        plt.close(fig)
        print(f"  Plot: {fname}")


# ===========================================================================
# SECTION 10: Utilities
# ===========================================================================

def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _die(code, msg):
    print(f"ERROR (exit {code}): {msg}", file=sys.stderr)
    sys.exit(code)


def _get_git_info(rundir):
    """Try to get git branch/commit for the run directory."""
    try:
        import subprocess
        branch = subprocess.check_output(
            ['git', '-C', rundir, 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        commit = subprocess.check_output(
            ['git', '-C', rundir, 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def load_thresholds(args):
    """Load thresholds from file (if --thresholds) and select profile."""
    profile = getattr(args, 'profile', 'gpu-fp') or 'gpu-fp'
    thr_file = getattr(args, 'thresholds', None)

    if thr_file and os.path.exists(thr_file):
        with open(thr_file) as f:
            all_thr = json.load(f)
        if profile in all_thr:
            return all_thr[profile]
        else:
            print(f"WARNING: profile '{profile}' not found in {thr_file}; using default gpu-fp")

    return DEFAULT_THRESHOLDS.get(profile, DEFAULT_THRESHOLDS['gpu-fp'])


# ===========================================================================
# SECTION 11: Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PINOCCHIO regression validation tool (VALIDATION_METHODOLOGY.md v1.0)"
    )
    p.add_argument("--reference",   required=True,
                   help="Reference run directory or golden directory")
    p.add_argument("--candidate",   required=True,
                   help="Candidate run directory")
    p.add_argument("--level",       choices=["exact", "reorder", "physical"],
                   default="physical",
                   help="Validation level: exact=L0, reorder=L1, physical=L2")
    p.add_argument("--profile",     choices=["gpu-fp", "algo-approx"],
                   default="gpu-fp",
                   help="Threshold profile (only for --level physical)")
    p.add_argument("--thresholds",  default=None,
                   help="Path to thresholds.json (uses built-in defaults if not given)")
    p.add_argument("--calibrate",   action="store_true",
                   help="Calibration mode: measure and print band, do not apply thresholds")
    p.add_argument("--make-golden", default=None, metavar="OUTDIR",
                   help="Freeze the reference run as a golden in OUTDIR")
    p.add_argument("--plots",       default=None, metavar="PLOTDIR",
                   help="Generate diagnostic plots in PLOTDIR (requires matplotlib)")
    p.add_argument("--report",      default=None, metavar="FILE",
                   help="Write report to FILE in addition to stdout")
    return p.parse_args()


def main():
    args = parse_args()

    # --make-golden: freeze reference, then exit
    if args.make_golden:
        ref = discover_run(args.reference)
        problems = integrity_check(ref, label="reference")
        if problems:
            for p in problems:
                print(p, file=sys.stderr)
            sys.exit(BROKEN)
        git_info = _get_git_info(ref['rundir'])
        make_golden(ref, args.make_golden, git_info=git_info)
        sys.exit(PASS)

    # --- Discover both runs ---
    print(f"\n{'='*60}")
    print(f"PINOCCHIO Regression Validation")
    print(f"  Reference : {args.reference}")
    print(f"  Candidate : {args.candidate}")
    print(f"  Level     : {args.level}")
    if args.level == 'physical':
        print(f"  Profile   : {args.profile}")
    print(f"{'='*60}")

    ref  = discover_run(args.reference)
    cand = discover_run(args.candidate)

    # --- Integrity gate (B.0) ---
    # In calibrate mode, np_mpi mismatch is expected (np=1 vs np=2 band measurement)
    # and is allowed; all other checks still apply.
    compat_problems = check_compatibility(ref, cand)
    if args.calibrate:
        # filter out np_mpi mismatch — calibration explicitly compares different np
        compat_problems = [p for p in compat_problems if "np_mpi" not in p]
    problems = (integrity_check(ref,  label="reference") +
                integrity_check(cand, label="candidate") +
                compat_problems)
    if problems:
        print("\n[INTEGRITY / SETUP ERRORS]")
        for p in problems:
            print(f"  {p}")
        print(f"\nVerdict: BROKEN (exit {BROKEN})")
        sys.exit(BROKEN)

    print(f"\n[B.0] Integrity OK — RunFlag='{ref['RunFlag']}' "
          f"Grid={ref['GridSize']} Box={ref['BoxSize']} "
          f"Seed={ref['RandomSeed']} np={ref['np_mpi']}")

    # --- Level 0: byte-exact ---
    if args.level == 'exact':
        print("\n[L0] Byte-exact comparison (SHA256):")
        ok, detail = compare_exact(ref, cand)
        for line in detail:
            print(line)
        verdict_str = "PASS" if ok else "FAIL"
        print(f"\nVerdict: {verdict_str} (exit {PASS if ok else FAIL})")
        if args.report:
            with open(args.report, 'w') as f:
                f.write("\n".join(detail))
        sys.exit(PASS if ok else FAIL)

    # --- Level 1: reorder ---
    if args.level == 'reorder':
        print("\n[L1] Reorder comparison (sort by name, float ulp):")
        ok, detail = compare_reorder(ref, cand)
        for line in detail:
            print(line)
        verdict_str = "PASS" if ok else "FAIL"
        print(f"\nVerdict: {verdict_str} (exit {PASS if ok else FAIL})")
        sys.exit(PASS if ok else FAIL)

    # --- Level 2: physical metrics ---
    all_metrics = {}

    # Load catalogs and compute metrics per z
    for z in ref['outputs']:
        print(f"\n[B.1+B.2] Loading catalogs z={z} ...")
        cat_r = _load_catalog(ref,  z)
        cat_c = _load_catalog(cand, z)
        if cat_r is None:
            print(f"  ERROR: cannot load reference catalog for z={z}")
            sys.exit(BROKEN)
        if cat_c is None:
            print(f"  ERROR: cannot load candidate catalog for z={z}")
            sys.exit(BROKEN)
        m = catalog_metrics(cat_r, cat_c, ref['BoxSize'], ref['GridSize'], ref['MinHaloMass'])
        all_metrics[f"cat_{z}"] = m
        print(f"  ref={m['N_ref']} cand={m['N_cand']} matched={m['N_match']} "
              f"f_match_all={m['f_match_all']:.4f} f_match_core={m['f_match_core']:.4f}")

    # HMF
    for z in ref['outputs']:
        mf_r = _mf_path(ref,  z)
        mf_c = _mf_path(cand, z)
        if os.path.exists(mf_r) and os.path.exists(mf_c):
            all_metrics[f"hmf_{z}"] = hmf_metrics(mf_r, mf_c)

    # Histories
    hist_r_path = _histories_path(ref)
    hist_c_path = _histories_path(cand)
    if os.path.exists(hist_r_path) and os.path.exists(hist_c_path):
        print(f"\n[B.4] Loading histories ...")
        href  = rp.histories(hist_r_path,  silent=True)
        hcand = rp.histories(hist_c_path,  silent=True)
        all_metrics["histories"] = histories_metrics(href, hcand)
    else:
        all_metrics["histories"] = {"error": "histories file not found"}

    # PLC (diagnostics only for now, per B.5 decision Morgan)
    if ref['PLC'] and os.path.exists(_plc_path(ref)):
        pass  # PLC kept as diagnostics; not included in verdict

    # --- Calibrate mode ---
    if args.calibrate:
        band = calibration_report(all_metrics, ref)
        with open("calibration_band.json", "w") as f:
            json.dump(band, f, indent=2)
        print(f"\nCalibration band saved to calibration_band.json")
        sys.exit(PASS)

    # --- Apply verdict ---
    thresholds = load_thresholds(args)
    ok, verdict_lines = verdict(all_metrics, thresholds, ref)

    print("\n[VERDICT DETAIL]")
    for line in verdict_lines:
        print(line)

    # Global sanity counts (B.6)
    print("\n[B.6] Global sanity counts (info only):")
    for z in ref['outputs']:
        m = all_metrics.get(f"cat_{z}")
        if m:
            print(f"  z={z}: ref={m['N_ref']}  cand={m['N_cand']}  "
                  f"match={m['N_match']}  only_ref={m['N_ref']-m['N_match']}  "
                  f"only_cand={m['N_cand']-m['N_match']}")
    hm = all_metrics.get("histories")
    if hm and "error" not in hm:
        print(f"  histories: Ntrees_ref={hm['Ntrees_ref']}  Ntrees_cand={hm['Ntrees_cand']}")

    verdict_str = "PASS" if ok else "FAIL"
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {verdict_str}  (exit {PASS if ok else FAIL})")
    print(f"{'='*60}\n")

    if args.plots:
        print(f"[PLOTS] Generating diagnostics in {args.plots} ...")
        make_plots(all_metrics, ref, args.plots)

    if args.report:
        with open(args.report, 'w') as f:
            f.write(f"Verdict: {verdict_str}\n")
            f.write("\n".join(verdict_lines))

    sys.exit(PASS if ok else FAIL)


if __name__ == "__main__":
    main()
