#!/usr/bin/env python3
"""
compare_catalogs.py — Confronto statistico tra due run PINOCCHIO

Uso:
    python compare_catalogs.py <test_dir> [ref_dir] [--redshift z] [--verbose] [--mf]

Parametri:
    test_dir    directory con i cataloghi del run da testare
    ref_dir     directory con i cataloghi di riferimento
                (default: state/reference_catalogs/ relativo alla repo root)
    --redshift  redshift da confrontare (default: 0.0)
    --verbose   mostra dettagli sugli aloni con grandi discrepanze
    --mf        stampa tabella comparativa della mass function

Output:
    - Summary statistico su stdout
    - Exit code 0 = PASS, 1 = FAIL (per uso in script CI)

Metriche calcolate:
    - N_halos: numero totale di aloni
    - Fraction matched by name (seed particle ID)
    - Mass ratio distribution: mean e std di M_test/M_ref
    - Mass conservation: sum(M_test) / sum(M_ref)
    - Mass function chi²/bin

Formati supportati:
    - ASCII  (parameter_file con CatalogInAscii, 12 colonne)
    - Binario (via ReadPinocchio5.py)

Cataloghi di riferimento (src_Leonardo, 4 MPI, GridSize=128, RandomSeed=486604):
    N_good_halos = 88982
    Rigenerazione: ./scripts/generate_reference_catalogs.sh
"""

import sys
import os
import argparse
import numpy as np

# path per ReadPinocchio5 (binario)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Thresholds pass/fail
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "matched_fraction":    0.95,   # >= 95% aloni matchati per nome
    "mass_ratio_mean_tol": 0.02,   # |<M_test/M_ref> - 1| < 2%
    "mass_ratio_std_max":  0.10,   # std(M_test/M_ref) < 10%
    "mass_conserv_tol":    0.01,   # |ΣM_test/ΣM_ref - 1| < 1%
    "nhalo_rel_tol":       0.02,   # |N_test - N_ref| / N_ref < 2%
}


# ─────────────────────────────────────────────────────────────────────────────
# I/O: rilevamento formato e lettura
# ─────────────────────────────────────────────────────────────────────────────

def is_ascii_catalog(path):
    """Ritorna True se il file è un catalogo ASCII PINOCCHIO."""
    try:
        with open(path, "rb") as f:
            first = f.read(1)
        return first == b"#"
    except Exception:
        return False


def read_ascii_catalog(path):
    """
    Legge un catalogo ASCII PINOCCHIO (formato CatalogInAscii).

    Colonne (12):
        1: group_ID (name)   int64
        2: mass              float32  [Msun/h]
        3-5: posin           float32  [Mpc/h]
        6-8: pos             float32  [Mpc/h]
        9-11: vel            float32  [km/s]
        12: npart            int32

    Ritorna un array numpy strutturato con gli stessi campi di ReadPinocchio5.
    """
    dtype = np.dtype([
        ("name",  np.int64),
        ("Mass",  np.float32),
        ("posin", np.float32, 3),
        ("pos",   np.float32, 3),
        ("vel",   np.float32, 3),
        ("npart", np.int32),
    ])

    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    if raw.shape[0] == 0:
        return np.zeros(0, dtype=dtype)

    out = np.zeros(len(raw), dtype=dtype)
    out["name"]     = raw[:, 0].astype(np.int64)
    out["Mass"]     = raw[:, 1].astype(np.float32)
    out["posin"]    = raw[:, 2:5].astype(np.float32)
    out["pos"]      = raw[:, 5:8].astype(np.float32)
    out["vel"]      = raw[:, 8:11].astype(np.float32)
    out["npart"]    = raw[:, 11].astype(np.int32)
    return out


def read_binary_catalog(path, verbose=False):
    """Legge un catalogo binario PINOCCHIO via ReadPinocchio5."""
    import ReadPinocchio5 as rp
    cat = rp.catalog(path, silent=not verbose)
    if cat is None:
        raise RuntimeError(f"Catalogo binario non leggibile: {path}")
    return cat.data


def load_catalog(path, verbose=False):
    """Carica un catalogo (auto-detect ASCII vs binario)."""
    if is_ascii_catalog(path):
        if verbose:
            print(f"  formato: ASCII")
        data = read_ascii_catalog(path)
    else:
        if verbose:
            print(f"  formato: binario")
        data = read_binary_catalog(path, verbose=verbose)

    if len(data) == 0:
        raise RuntimeError(f"Catalogo vuoto: {path}")
    if "posin" not in data.dtype.names:
        raise RuntimeError(f"Campo 'posin' assente: {path}")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Ricerca file catalogo
# ─────────────────────────────────────────────────────────────────────────────

def find_catalog(directory, redshift):
    """Trova il file catalog per un dato redshift nella directory."""
    z_str = f"{redshift:.4f}"
    candidates = [f for f in os.listdir(directory)
                  if "catalog" in f and f.endswith(".out") and z_str in f]
    if not candidates:
        all_cats = [f for f in os.listdir(directory)
                    if "catalog" in f and f.endswith(".out")]
        raise FileNotFoundError(
            f"Nessun catalogo z={z_str} in {directory}\n"
            f"Cataloghi disponibili: {all_cats}"
        )
    if len(candidates) > 1:
        print(f"  WARN: {len(candidates)} candidati, uso: {candidates[0]}")
    return os.path.join(directory, sorted(candidates)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Comparazione
# ─────────────────────────────────────────────────────────────────────────────

def mass_function_bins(masses, Mmin=None, Mmax=None, Nbins=15):
    if Mmin is None: Mmin = float(masses.min())
    if Mmax is None: Mmax = float(masses.max())
    bins = np.logspace(np.log10(Mmin), np.log10(Mmax), Nbins + 1)
    counts, _ = np.histogram(masses, bins=bins)
    centers = np.sqrt(bins[:-1] * bins[1:])
    return centers, counts, bins


def compare(ref_data, test_data, verbose=False):
    """
    Confronta ref_data e test_data (array strutturati).
    Ritorna dict con metriche.
    """
    res = {}

    # 1. Numero di aloni
    N_ref  = len(ref_data)
    N_test = len(test_data)
    res["N_ref"]      = N_ref
    res["N_test"]     = N_test
    res["delta_N_rel"] = (N_test - N_ref) / N_ref

    # 2. Match per nome (seed particle ID)
    ref_idx = {int(n): i for i, n in enumerate(ref_data["name"])}

    matched   = []      # (i_test, i_ref)
    unmatched_test = []

    for i, n in enumerate(test_data["name"]):
        ni = int(n)
        if ni in ref_idx:
            matched.append((i, ref_idx[ni]))
        else:
            unmatched_test.append(i)

    test_names = set(int(n) for n in test_data["name"])
    unmatched_ref = [i for i, n in enumerate(ref_data["name"])
                     if int(n) not in test_names]

    matched = np.array(matched, dtype=np.int64) if matched else np.empty((0, 2), dtype=np.int64)
    Nmatched = len(matched)

    res["N_matched"]        = Nmatched
    res["matched_fraction"] = Nmatched / N_ref
    res["N_unmatched_test"] = len(unmatched_test)
    res["N_unmatched_ref"]  = len(unmatched_ref)

    # 3. Mass ratio per aloni matchati
    if Nmatched > 0:
        M_test = test_data["Mass"][matched[:, 0]].astype(float)
        M_ref  = ref_data["Mass"][matched[:, 1]].astype(float)
        ratio  = M_test / M_ref
        res["mass_ratio_mean"] = float(np.mean(ratio))
        res["mass_ratio_std"]  = float(np.std(ratio))
        res["mass_ratio_p10"]  = float(np.percentile(ratio, 10))
        res["mass_ratio_p90"]  = float(np.percentile(ratio, 90))

        big_diff = np.abs(ratio - 1.0) > 0.10
        res["N_mass_change_10pct"] = int(np.sum(big_diff))

        if verbose and np.sum(big_diff) > 0:
            ii = matched[big_diff, 0]
            ji = matched[big_diff, 1]
            print(f"\n  Aloni con |ΔM/M| > 10% ({len(ii)} totali, max 10):")
            for k, (it, ir) in enumerate(zip(ii[:10], ji[:10])):
                print(f"    name={test_data['name'][it]}  "
                      f"M_test={test_data['Mass'][it]:.3e}  "
                      f"M_ref={ref_data['Mass'][ir]:.3e}  "
                      f"ratio={ratio[big_diff][k]:.3f}  "
                      f"posin_ref={ref_data['posin'][ir]}")
    else:
        res["mass_ratio_mean"]    = float("nan")
        res["mass_ratio_std"]     = float("nan")
        res["mass_ratio_p10"]     = float("nan")
        res["mass_ratio_p90"]     = float("nan")
        res["N_mass_change_10pct"] = 0

    # 4. Conservazione massa totale
    res["mass_conservation"] = (float(test_data["Mass"].sum()) /
                                 float(ref_data["Mass"].sum()))

    # 5. Mass function chi²/bin
    Mmin = min(float(ref_data["Mass"].min()),  float(test_data["Mass"].min()))
    Mmax = max(float(ref_data["Mass"].max()),  float(test_data["Mass"].max()))
    centers, cnt_ref,  _ = mass_function_bins(ref_data["Mass"],  Mmin, Mmax)
    _,       cnt_test, _ = mass_function_bins(test_data["Mass"], Mmin, Mmax)
    mask = cnt_ref > 0
    res["mf_chi2_per_bin"] = (float(np.sum((cnt_test[mask] - cnt_ref[mask])**2 /
                                            cnt_ref[mask]) / mask.sum())
                               if mask.sum() > 0 else float("nan"))
    res["mf_centers"]     = centers
    res["mf_counts_ref"]  = cnt_ref
    res["mf_counts_test"] = cnt_test

    # 6. Diagnostica unmatched
    if len(unmatched_test) > 0:
        res["unmatched_test_mass_mean"]  = float(test_data["Mass"][unmatched_test].mean())
        res["unmatched_test_posin_mean"] = test_data["posin"][unmatched_test].mean(axis=0)
    if len(unmatched_ref) > 0:
        res["unmatched_ref_mass_mean"]   = float(ref_data["Mass"][unmatched_ref].mean())
        res["unmatched_ref_posin_mean"]  = ref_data["posin"][unmatched_ref].mean(axis=0)

    return res


# ─────────────────────────────────────────────────────────────────────────────
# Report e thresholds
# ─────────────────────────────────────────────────────────────────────────────

def report(res, thr=THRESHOLDS):
    """Stampa il report. Ritorna True se PASS."""
    sep = "─" * 64
    print(sep)
    print("  PINOCCHIO catalog comparison")
    print(sep)

    checks = []

    def check(name, value, threshold, fmt, ok_fn, label=""):
        ok = ok_fn(value, threshold)
        checks.append(ok)
        flag = "✓" if ok else "✗"
        thr_str = f"thr {label}{threshold}"
        print(f"  {flag} {name:<22} {fmt.format(value)}   ({thr_str})")
        return ok

    # N halos
    dN = res["delta_N_rel"]
    ok = abs(dN) < thr["nhalo_rel_tol"]
    checks.append(ok)
    print(f"  {'✓' if ok else '✗'} N_halos               "
          f"ref={res['N_ref']:6d}  test={res['N_test']:6d}  "
          f"ΔN/N={dN:+.4f}   (thr ±{thr['nhalo_rel_tol']:.2f})")

    # Match fraction
    mf = res["matched_fraction"]
    ok = mf >= thr["matched_fraction"]
    checks.append(ok)
    print(f"  {'✓' if ok else '✗'} matched fraction       "
          f"{res['N_matched']}/{res['N_ref']} = {100*mf:.2f}%   "
          f"unmatched_test={res['N_unmatched_test']}  "
          f"unmatched_ref={res['N_unmatched_ref']}   "
          f"(thr ≥{100*thr['matched_fraction']:.0f}%)")

    # Mass ratio
    mr_mean = res["mass_ratio_mean"]
    mr_std  = res["mass_ratio_std"]
    if not np.isnan(mr_mean):
        ok_m = abs(mr_mean - 1.0) < thr["mass_ratio_mean_tol"]
        ok_s = mr_std < thr["mass_ratio_std_max"]
        checks += [ok_m, ok_s]
        flag = "✓" if (ok_m and ok_s) else "✗"
        print(f"  {flag} mass ratio (matched)  "
              f"mean={mr_mean:.5f}  std={mr_std:.5f}  "
              f"p10={res['mass_ratio_p10']:.3f}  p90={res['mass_ratio_p90']:.3f}   "
              f"(mean ±{thr['mass_ratio_mean_tol']:.2f}, std <{thr['mass_ratio_std_max']:.2f})")
        print(f"    halos |ΔM/M|>10%:    {res['N_mass_change_10pct']}")

    # Mass conservation
    mc = res["mass_conservation"]
    ok = abs(mc - 1.0) < thr["mass_conserv_tol"]
    checks.append(ok)
    print(f"  {'✓' if ok else '✗'} mass conservation      "
          f"ΣM_test/ΣM_ref = {mc:.6f}   "
          f"(thr ±{thr['mass_conserv_tol']:.3f})")

    # Mass function chi²
    chi2 = res["mf_chi2_per_bin"]
    if not np.isnan(chi2):
        print(f"    MF χ²/bin:           {chi2:.3f}")

    print(sep)
    passed = all(checks)
    print(f"  Overall:   {'PASS' if passed else 'FAIL'}")
    print(sep)
    return passed


def print_mf_table(res):
    print("\n  Mass function comparison (N halos per bin):")
    print(f"  {'log10(M)':>10}  {'N_ref':>8}  {'N_test':>8}  {'ratio':>8}")
    for c, r, t in zip(res["mf_centers"], res["mf_counts_ref"], res["mf_counts_test"]):
        ratio = t / r if r > 0 else float("nan")
        print(f"  {np.log10(c):>10.2f}  {r:>8d}  {t:>8d}  {ratio:>8.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Confronta cataloghi PINOCCHIO con riferimento src_Leonardo"
    )
    parser.add_argument("test_dir",
                        help="Directory con cataloghi da testare")
    parser.add_argument("ref_dir", nargs="?", default=None,
                        help="Directory riferimento (default: state/reference_catalogs/)")
    parser.add_argument("--redshift", "-z", type=float, default=0.0,
                        help="Redshift da confrontare (default: 0.0)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--mf", action="store_true",
                        help="Stampa tabella mass function")
    args = parser.parse_args()

    if args.ref_dir is None:
        repo_root = os.path.dirname(SCRIPT_DIR)
        args.ref_dir = os.path.join(repo_root, "state", "reference_catalogs")

    for d in [args.ref_dir, args.test_dir]:
        if not os.path.isdir(d):
            print(f"ERROR: directory non trovata: {d}")
            sys.exit(2)

    print(f"Reference: {args.ref_dir}")
    print(f"Test:      {args.test_dir}")
    print(f"Redshift:  z={args.redshift:.4f}\n")

    ref_path  = find_catalog(args.ref_dir,  args.redshift)
    test_path = find_catalog(args.test_dir, args.redshift)

    print(f"Ref  catalog: {os.path.basename(ref_path)}")
    print(f"Test catalog: {os.path.basename(test_path)}\n")

    ref_data  = load_catalog(ref_path,  verbose=args.verbose)
    test_data = load_catalog(test_path, verbose=args.verbose)

    res = compare(ref_data, test_data, verbose=args.verbose)
    passed = report(res)

    if args.mf:
        print_mf_table(res)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
