#!/usr/bin/env python3
"""
PINOCCHIO Statistical Validation Script

Compares two PINOCCHIO runs and determines whether they are
statistically equivalent. Implements the validation plan
documented in state/validation_plan.md.

Usage:
    python validate_run.py --ref-dir REF --test-dir TEST --run-name NAME \
        --ref-log REF_LOG --test-log TEST_LOG \
        [--redshifts Z1 Z2 ...] [--use-hardcoded-ref] [--plot]

Exit codes:
    0 = PASS
    1 = FAIL
    2 = BORDERLINE (manual inspection needed)
"""

import argparse
import re
import sys
import os
import numpy as np
from scipy import stats


# ============================================================
# Reference values from CLAUDE.md (1 MPI task, GridSize=128,
# RandomSeed=486604)
# ============================================================
HARDCODED_REF = {
    "peaks":                107684,
    "good_halos":           88981,
    "neighbours":           [331378, 117995, 13584, 430, 4, 0],
    "accretion_events":     268658,
    "accretion_before":     79071,
    "accretion_after":      364,
    "filament_accretion":   48327,
}


def parse_log_scalars(logfile):
    """
    Parse fragmentation scalars from PINOCCHIO stdout log.

    Parameters
    ----------
    logfile : str
        Path to the log file (stdout capture of pinocchio.x).

    Returns
    -------
    dict
        Dictionary with scalar names as keys and integer values.
    """
    scalars = {}
    with open(logfile, 'r') as f:
        text = f.read()

    m = re.search(r'Total number of peaks:\s+(\d+)', text)
    if m:
        scalars['peaks'] = int(m.group(1))

    m = re.search(r'Total number of good halos:\s+(\d+)', text)
    if m:
        scalars['good_halos'] = int(m.group(1))

    m = re.search(
        r'Particles with N neighbouring groups:\s+'
        r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)',
        text
    )
    if m:
        scalars['neighbours'] = [int(m.group(i)) for i in range(1, 7)]

    m = re.search(r'Total number of accretion events:\s+(\d+)', text)
    if m:
        scalars['accretion_events'] = int(m.group(1))

    m = re.search(r'Accretion before evaluating merger:\s+(\d+)', text)
    if m:
        scalars['accretion_before'] = int(m.group(1))

    m = re.search(r'Accretion after evaluating merger:\s+(\d+)', text)
    if m:
        scalars['accretion_after'] = int(m.group(1))

    m = re.search(r'Accretion of filament particles:\s+(\d+)', text)
    if m:
        scalars['filament_accretion'] = int(m.group(1))

    m = re.search(r'Total number of merger events:\s+(\d+)', text)
    if m:
        scalars['merger_events'] = int(m.group(1))

    m = re.search(r'Total number of major merger events:\s+(\d+)', text)
    if m:
        scalars['major_mergers'] = int(m.group(1))

    m = re.search(r'Total number of collapsed particles:\s+(\d+)', text)
    if m:
        scalars['collapsed_particles'] = int(m.group(1))

    return scalars


def compare_scalars(ref, test, same_mpi_config=True):
    """
    Compare scalar counters between reference and test runs.

    Parameters
    ----------
    ref : dict
        Reference scalars.
    test : dict
        Test scalars.
    same_mpi_config : bool
        If True, use tight tolerances (same MPI decomposition expected).

    Returns
    -------
    str
        'PASS', 'BORDERLINE', or 'FAIL'
    list of str
        List of diagnostic messages.
    """
    messages = []
    worst = 'PASS'

    # Strict invariants: peaks and collapsed_particles
    strict_keys = ['peaks']
    if 'collapsed_particles' in ref and 'collapsed_particles' in test:
        strict_keys.append('collapsed_particles')

    for key in strict_keys:
        if key not in ref or key not in test:
            continue
        if ref[key] != test[key]:
            messages.append(
                f"[FAIL] STRICT: {key} differs: ref={ref[key]}, test={test[key]}"
            )
            worst = 'FAIL'
        else:
            messages.append(f"[PASS] STRICT: {key} = {ref[key]}")

    # Statistical invariants
    stat_keys = [
        'good_halos', 'accretion_events', 'accretion_before',
        'accretion_after', 'filament_accretion',
        'merger_events', 'major_mergers'
    ]

    tol_rel = 0.001 if same_mpi_config else 0.01
    tol_abs = 50
    border_rel = 0.01 if same_mpi_config else 0.05
    border_abs = 200

    for key in stat_keys:
        if key not in ref or key not in test:
            continue
        diff = abs(test[key] - ref[key])
        rel = diff / max(ref[key], 1)

        if diff <= tol_abs and rel <= tol_rel:
            messages.append(
                f"[PASS] STAT: {key}: ref={ref[key]}, test={test[key]}, "
                f"diff={diff}, rel={rel:.4%}"
            )
        elif diff <= border_abs and rel <= border_rel:
            messages.append(
                f"[BORDERLINE] STAT: {key}: ref={ref[key]}, test={test[key]}, "
                f"diff={diff}, rel={rel:.4%}"
            )
            if worst == 'PASS':
                worst = 'BORDERLINE'
        else:
            messages.append(
                f"[FAIL] STAT: {key}: ref={ref[key]}, test={test[key]}, "
                f"diff={diff}, rel={rel:.4%}"
            )
            worst = 'FAIL'

    # Neighbours array
    if 'neighbours' in ref and 'neighbours' in test:
        for i in range(6):
            r = ref['neighbours'][i]
            t = test['neighbours'][i]
            if r == 0 and t == 0:
                continue
            diff = abs(t - r)
            rel = diff / max(r, 1)
            label = f"neighbours[{i}]"
            if diff <= tol_abs and rel <= tol_rel:
                messages.append(
                    f"[PASS] STAT: {label}: ref={r}, test={t}, "
                    f"diff={diff}, rel={rel:.4%}"
                )
            elif diff <= border_abs and rel <= border_rel:
                messages.append(
                    f"[BORDERLINE] STAT: {label}: ref={r}, test={t}, "
                    f"diff={diff}, rel={rel:.4%}"
                )
                if worst == 'PASS':
                    worst = 'BORDERLINE'
            else:
                messages.append(
                    f"[FAIL] STAT: {label}: ref={r}, test={t}, "
                    f"diff={diff}, rel={rel:.4%}"
                )
                worst = 'FAIL'

    return worst, messages


def read_mass_function(filepath):
    """
    Read a PINOCCHIO mass function file.

    Parameters
    ----------
    filepath : str
        Path to pinocchio.ZZZZ.RunName.mf.out

    Returns
    -------
    mass : ndarray
        Halo mass in Msun/h (bin center).
    nm : ndarray
        n(m) differential number density.
    n_halos : ndarray
        Number of halos in the bin (integer counts).
    nm_analytic : ndarray
        Analytic prediction (Watson et al. 2013).
    """
    data = np.loadtxt(filepath, comments='#')
    mass = data[:, 0]
    nm = data[:, 1]
    n_halos = data[:, 4].astype(int)
    nm_analytic = data[:, 5]
    return mass, nm, n_halos, nm_analytic


def compare_mass_function(ref_file, test_file, redshift_label="z=?"):
    """
    Compare mass functions from two runs.

    Parameters
    ----------
    ref_file : str
        Path to reference MF file.
    test_file : str
        Path to test MF file.
    redshift_label : str
        Label for log messages.

    Returns
    -------
    str
        'PASS', 'BORDERLINE', or 'FAIL'
    list of str
        Diagnostic messages.
    dict
        Data for plotting.
    """
    messages = []
    worst = 'PASS'

    mass_ref, nm_ref, n_ref, _ = read_mass_function(ref_file)
    mass_test, nm_test, n_test, _ = read_mass_function(test_file)

    # Total halo count comparison
    N_tot_ref = n_ref.sum()
    N_tot_test = n_test.sum()
    rel_diff_total = abs(N_tot_test - N_tot_ref) / max(N_tot_ref, 1)

    messages.append(
        f"[{redshift_label}] Total halos: ref={N_tot_ref}, test={N_tot_test}, "
        f"rel_diff={rel_diff_total:.4%}"
    )
    if rel_diff_total > 0.01:
        messages.append(f"  [FAIL] Total halo count differs by more than 1%")
        worst = 'FAIL'
    elif rel_diff_total > 0.005:
        messages.append(f"  [BORDERLINE] Total halo count differs by 0.5-1%")
        if worst == 'PASS':
            worst = 'BORDERLINE'

    # Match bins by mass (require same binning)
    if len(mass_ref) != len(mass_test):
        messages.append(
            f"  [WARN] Different number of MF bins: ref={len(mass_ref)}, "
            f"test={len(mass_test)}. Using overlapping range."
        )
        m_min = max(mass_ref.min(), mass_test.min())
        m_max = min(mass_ref.max(), mass_test.max())
        idx_ref = (mass_ref >= m_min * 0.99) & (mass_ref <= m_max * 1.01)
        idx_test = (mass_test >= m_min * 0.99) & (mass_test <= m_max * 1.01)
        n_ref_matched = n_ref[idx_ref]
        n_test_matched = n_test[idx_test]
        mass_matched = mass_ref[idx_ref]
        if len(n_ref_matched) != len(n_test_matched):
            messages.append(f"  [FAIL] Cannot match MF bins between ref and test")
            return 'FAIL', messages, {}
    else:
        n_ref_matched = n_ref
        n_test_matched = n_test
        mass_matched = mass_ref

    # Chi-square test on bins with N_ref >= 10
    mask_chi2 = n_ref_matched >= 10
    if mask_chi2.sum() > 0:
        chi2 = np.sum(
            (n_test_matched[mask_chi2] - n_ref_matched[mask_chi2])**2
            / n_ref_matched[mask_chi2]
        )
        ndof = mask_chi2.sum()
        chi2_red = chi2 / ndof
        messages.append(
            f"  Chi2/Ndof = {chi2:.1f}/{ndof} = {chi2_red:.3f}"
        )
        if chi2_red > 5.0:
            messages.append(f"  [FAIL] Chi2/Ndof > 5.0")
            worst = 'FAIL'
        elif chi2_red > 2.0:
            messages.append(f"  [BORDERLINE] Chi2/Ndof between 2.0 and 5.0")
            if worst == 'PASS':
                worst = 'BORDERLINE'
        else:
            messages.append(f"  [PASS] Chi2/Ndof < 2.0")

    # Ratio test on bins with N_ref > 5
    mask_ratio = n_ref_matched > 5
    if mask_ratio.sum() > 0:
        ratio = np.where(
            n_ref_matched[mask_ratio] > 0,
            n_test_matched[mask_ratio] / n_ref_matched[mask_ratio],
            1.0
        )
        tol = np.maximum(3.0 / np.sqrt(n_ref_matched[mask_ratio]), 0.05)

        outliers = np.abs(ratio - 1.0) > tol
        n_outliers = outliers.sum()
        if n_outliers > 0:
            messages.append(
                f"  [WARN] {n_outliers}/{mask_ratio.sum()} MF bins outside "
                f"tolerance in ratio test"
            )
            extreme = (n_ref_matched[mask_ratio] > 20) & (
                (ratio > 2.0) | (ratio < 0.5)
            )
            if extreme.any():
                messages.append(
                    f"  [FAIL] {extreme.sum()} bins with N>20 have ratio "
                    f"outside [0.5, 2.0]"
                )
                worst = 'FAIL'
            elif n_outliers > mask_ratio.sum() * 0.3:
                messages.append(
                    f"  [BORDERLINE] More than 30% of bins are outliers"
                )
                if worst == 'PASS':
                    worst = 'BORDERLINE'
        else:
            messages.append(f"  [PASS] All MF bins within tolerance")

    plot_data = {
        'mass_ref': mass_ref, 'n_ref': n_ref,
        'mass_test': mass_test, 'n_test': n_test,
    }

    return worst, messages, plot_data


def read_catalog_ascii(filepath):
    """
    Read an ASCII halo catalog.

    Parameters
    ----------
    filepath : str
        Path to pinocchio.ZZZZ.RunName.catalog.out

    Returns
    -------
    dict with keys:
        'id': ndarray int64
        'mass': ndarray float64  (Msun/h)
        'pos_init': ndarray (N,3) float64  (Mpc/h)
        'pos_final': ndarray (N,3) float64  (Mpc/h)
        'vel': ndarray (N,3) float64  (km/s)
        'npart': ndarray int32
    """
    data = np.loadtxt(filepath, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return {
        'id': data[:, 0].astype(np.int64),
        'mass': data[:, 1],
        'pos_init': data[:, 2:5],
        'pos_final': data[:, 5:8],
        'vel': data[:, 8:11],
        'npart': data[:, 11].astype(np.int32),
    }


def compare_catalogs(ref_file, test_file, redshift_label="z=?"):
    """
    Compare halo catalogs from two runs using KS tests.

    Parameters
    ----------
    ref_file, test_file : str
        Paths to catalog files.
    redshift_label : str
        Label for messages.

    Returns
    -------
    str
        'PASS', 'BORDERLINE', or 'FAIL'
    list of str
        Diagnostic messages.
    """
    messages = []
    worst = 'PASS'

    ref = read_catalog_ascii(ref_file)
    test = read_catalog_ascii(test_file)

    N_ref = len(ref['mass'])
    N_test = len(test['mass'])
    messages.append(
        f"[{redshift_label}] Catalog: ref={N_ref} halos, test={N_test} halos"
    )

    if N_ref == 0 or N_test == 0:
        messages.append(f"  [WARN] One catalog is empty, skipping KS tests")
        return worst, messages

    rel_diff = abs(N_test - N_ref) / N_ref
    if rel_diff > 0.01:
        messages.append(
            f"  [FAIL] Number of halos differs by {rel_diff:.2%}"
        )
        worst = 'FAIL'

    # KS test on mass distribution
    ks_stat, ks_pval = stats.ks_2samp(ref['mass'], test['mass'])
    N_eff = min(N_ref, N_test)
    threshold_5pct = 1.36 / np.sqrt(N_eff)
    threshold_1pct = 1.63 / np.sqrt(N_eff)

    messages.append(
        f"  KS test (mass): D={ks_stat:.6f}, p={ks_pval:.4f}, "
        f"threshold_5%={threshold_5pct:.6f}"
    )
    if ks_stat > threshold_1pct:
        messages.append(f"  [FAIL] KS mass: D > 1% threshold")
        worst = 'FAIL'
    elif ks_stat > threshold_5pct:
        messages.append(f"  [BORDERLINE] KS mass: D between 5% and 1% threshold")
        if worst == 'PASS':
            worst = 'BORDERLINE'
    else:
        messages.append(f"  [PASS] KS mass")

    # KS test on velocity magnitude distribution
    v_ref = np.sqrt(np.sum(ref['vel']**2, axis=1))
    v_test = np.sqrt(np.sum(test['vel']**2, axis=1))
    ks_stat_v, ks_pval_v = stats.ks_2samp(v_ref, v_test)
    messages.append(
        f"  KS test (|v|): D={ks_stat_v:.6f}, p={ks_pval_v:.4f}"
    )
    if ks_stat_v > threshold_1pct:
        messages.append(f"  [FAIL] KS velocity: D > 1% threshold")
        worst = 'FAIL'
    elif ks_stat_v > threshold_5pct:
        messages.append(f"  [BORDERLINE] KS velocity")
        if worst == 'PASS':
            worst = 'BORDERLINE'
    else:
        messages.append(f"  [PASS] KS velocity")

    return worst, messages


def make_plots(ref_dir, test_dir, run_name, redshifts, output_dir="."):
    """
    Generate comparison plots (MF ratio, cumulative MF, velocity distribution).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for zstr in redshifts:
        ref_mf = os.path.join(ref_dir, f"pinocchio.{zstr}.{run_name}.mf.out")
        test_mf = os.path.join(test_dir, f"pinocchio.{zstr}.{run_name}.mf.out")

        if not (os.path.exists(ref_mf) and os.path.exists(test_mf)):
            continue

        m_ref, nm_ref, n_ref, ana_ref = read_mass_function(ref_mf)
        m_test, nm_test, n_test, ana_test = read_mass_function(test_mf)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1]})

        # Upper panel: MF
        ax = axes[0]
        ax.plot(m_ref, m_ref * nm_ref, 'o-', label='Reference', color='blue',
                markersize=3)
        ax.plot(m_test, m_test * nm_test, 's--', label='Test', color='red',
                markersize=3)
        ax.plot(m_ref, m_ref * ana_ref, '-', label='Watson 2013', color='gray',
                alpha=0.5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylabel(r'$M \, n(M)$ [Mpc$^{-3}$]')
        ax.set_title(f'Mass Function at z={zstr}')
        ax.legend()

        # Lower panel: ratio
        ax = axes[1]
        if len(m_ref) == len(m_test):
            mask = n_ref > 5
            ratio = np.where(n_ref > 0, n_test / np.maximum(n_ref, 1), 1.0)
            ax.plot(m_ref[mask], ratio[mask], 'ko', markersize=4)
            ax.axhline(1.0, color='gray', ls='--')
            tol = np.maximum(3.0 / np.sqrt(np.maximum(n_ref[mask], 1)), 0.05)
            ax.fill_between(m_ref[mask], 1 - tol, 1 + tol,
                           alpha=0.2, color='green', label='Tolerance')
        ax.set_xscale('log')
        ax.set_ylabel('Test / Reference')
        ax.set_xlabel(r'M [M$_\odot$/h]')
        ax.set_ylim([0.5, 1.5])
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'validation_mf_z{zstr}.png'),
                    dpi=150)
        plt.close()

    # Catalog comparison plots at z=0
    zstr = redshifts[0] if redshifts else '0.0000'
    ref_cat = os.path.join(ref_dir, f"pinocchio.{zstr}.{run_name}.catalog.out")
    test_cat = os.path.join(test_dir, f"pinocchio.{zstr}.{run_name}.catalog.out")

    if os.path.exists(ref_cat) and os.path.exists(test_cat):
        ref = read_catalog_ascii(ref_cat)
        test = read_catalog_ascii(test_cat)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Cumulative MF
        ax = axes[0]
        m_sorted_ref = np.sort(ref['mass'])[::-1]
        m_sorted_test = np.sort(test['mass'])[::-1]
        ax.plot(m_sorted_ref, np.arange(1, len(m_sorted_ref) + 1),
                label='Reference', color='blue')
        ax.plot(m_sorted_test, np.arange(1, len(m_sorted_test) + 1),
                label='Test', color='red', ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'M [M$_\odot$/h]')
        ax.set_ylabel('N(>M)')
        ax.set_title(f'Cumulative MF at z={zstr}')
        ax.legend()

        # Velocity distribution
        ax = axes[1]
        v_ref = np.sqrt(np.sum(ref['vel']**2, axis=1))
        v_test = np.sqrt(np.sum(test['vel']**2, axis=1))
        bins = np.linspace(0, max(v_ref.max(), v_test.max()), 50)
        ax.hist(v_ref, bins=bins, alpha=0.5, label='Reference', color='blue',
                density=True)
        ax.hist(v_test, bins=bins, alpha=0.5, label='Test', color='red',
                density=True)
        ax.set_xlabel('|v| [km/s]')
        ax.set_ylabel('PDF')
        ax.set_title(f'Velocity distribution at z={zstr}')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'validation_catalog_z{zstr}.png'),
                    dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='PINOCCHIO statistical validation'
    )
    parser.add_argument('--ref-dir', type=str, default=None,
                       help='Directory with reference output files')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Directory with test output files')
    parser.add_argument('--run-name', type=str, default='example',
                       help='Run flag name (default: example)')
    parser.add_argument('--ref-log', type=str, default=None,
                       help='Path to reference log file')
    parser.add_argument('--test-log', type=str, required=True,
                       help='Path to test log file')
    parser.add_argument('--redshifts', nargs='+',
                       default=['0.0000', '0.5000', '1.0000', '2.0000'],
                       help='Redshift labels for MF files')
    parser.add_argument('--use-hardcoded-ref', action='store_true',
                       help='Use hardcoded reference values from CLAUDE.md')
    parser.add_argument('--same-mpi', action='store_true', default=False,
                       help='Expect identical MPI decomposition (tight tol)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--plot-dir', type=str, default='.',
                       help='Directory for output plots')

    args = parser.parse_args()

    overall = 'PASS'

    def update_overall(result):
        nonlocal overall
        if result == 'FAIL':
            overall = 'FAIL'
        elif result == 'BORDERLINE' and overall == 'PASS':
            overall = 'BORDERLINE'

    # ---- 1. Scalar comparison ----
    print("=" * 60)
    print("SECTION 1: Scalar invariants")
    print("=" * 60)

    test_scalars = parse_log_scalars(args.test_log)

    if args.use_hardcoded_ref:
        ref_scalars = HARDCODED_REF
        same_mpi = False
    elif args.ref_log:
        ref_scalars = parse_log_scalars(args.ref_log)
        same_mpi = args.same_mpi
    else:
        print("[SKIP] No reference log provided and --use-hardcoded-ref not set")
        ref_scalars = None

    if ref_scalars:
        result, msgs = compare_scalars(ref_scalars, test_scalars, same_mpi)
        update_overall(result)
        for m in msgs:
            print(f"  {m}")
        print(f"\n  Section 1 result: {result}\n")

    # ---- 2. Mass function comparison ----
    print("=" * 60)
    print("SECTION 2: Mass function comparison")
    print("=" * 60)

    for zstr in args.redshifts:
        test_mf = os.path.join(
            args.test_dir, f"pinocchio.{zstr}.{args.run_name}.mf.out"
        )
        if args.ref_dir:
            ref_mf = os.path.join(
                args.ref_dir, f"pinocchio.{zstr}.{args.run_name}.mf.out"
            )
        else:
            ref_mf = None

        if ref_mf and os.path.exists(ref_mf) and os.path.exists(test_mf):
            result, msgs, _ = compare_mass_function(
                ref_mf, test_mf, f"z={zstr}"
            )
            update_overall(result)
            for m in msgs:
                print(f"  {m}")
            print(f"  Section 2 (z={zstr}) result: {result}\n")
        else:
            print(f"  [SKIP] MF file not found for z={zstr}")

    # ---- 3. Catalog comparison ----
    print("=" * 60)
    print("SECTION 3: Catalog comparison (KS tests)")
    print("=" * 60)

    for zstr in args.redshifts:
        test_cat = os.path.join(
            args.test_dir, f"pinocchio.{zstr}.{args.run_name}.catalog.out"
        )
        if args.ref_dir:
            ref_cat = os.path.join(
                args.ref_dir, f"pinocchio.{zstr}.{args.run_name}.catalog.out"
            )
        else:
            ref_cat = None

        if ref_cat and os.path.exists(ref_cat) and os.path.exists(test_cat):
            result, msgs = compare_catalogs(ref_cat, test_cat, f"z={zstr}")
            update_overall(result)
            for m in msgs:
                print(f"  {m}")
            print(f"  Section 3 (z={zstr}) result: {result}\n")
        else:
            print(f"  [SKIP] Catalog not found for z={zstr}")

    # ---- 4. Generate plots ----
    if args.plot and args.ref_dir:
        print("=" * 60)
        print("SECTION 4: Generating comparison plots")
        print("=" * 60)
        make_plots(args.ref_dir, args.test_dir, args.run_name,
                  args.redshifts, args.plot_dir)
        print(f"  Plots saved to {args.plot_dir}/")

    # ---- Final verdict ----
    print("\n" + "=" * 60)
    print(f"OVERALL RESULT: {overall}")
    print("=" * 60)

    if overall == 'PASS':
        print("The test run is statistically equivalent to the reference.")
        sys.exit(0)
    elif overall == 'BORDERLINE':
        print("Some tests are borderline. Manual inspection recommended.")
        print("Check the plots and the diagnostic messages above.")
        sys.exit(2)
    else:
        print("The test run FAILS validation. Investigate the issues above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
