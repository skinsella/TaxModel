#!/usr/bin/env python3
"""
Irish Tax Costing Model
========================
Replicates Revenue's Ready Reckoner for income tax and USC costings.

Based on:
- Revenue Ready Reckoner Post-Budget 2026 (October 2025)
- Individualised Gross Income data (2023)
- USC Standard Rates and Thresholds 2026

Two-component approach:
  1. USC: Microsimulation on individual income distribution (scaled to 2026)
  2. Income Tax: Interpolation from Ready Reckoner unit costs

Run directly for validation and examples:
    python tax_model.py
"""

import numpy as np
from dataclasses import dataclass

# ============================================================
# DATA: 2023 Individualised Gross Income (All Taxpayers)
# Source: Revenue - Individualised Gross Incomes (2023)
# (lower, upper, num_individuals, total_gross_income_€m)
# ============================================================

INDIVIDUALS_2023 = [
    (0,       10_000,  557_390,  2_622.84),
    (10_000,  12_000,  122_705,  1_361.31),
    (12_000,  15_000,  343_481,  4_663.45),
    (15_000,  17_000,  187_734,  2_985.64),
    (17_000,  20_000,  195_215,  3_607.05),
    (20_000,  25_000,  310_555,  6_990.34),
    (25_000,  27_000,  129_091,  3_360.14),
    (27_000,  30_000,  193_791,  5_521.52),
    (30_000,  35_000,  304_175,  9_869.89),
    (35_000,  40_000,  272_633, 10_216.77),
    (40_000,  50_000,  420_841, 18_788.30),
    (50_000,  60_000,  272_555, 14_904.12),
    (60_000,  70_000,  182_260, 11_786.48),
    (70_000,  75_000,   66_742,  4_832.93),
    (75_000,  80_000,   53_842,  4_168.64),
    (80_000,  90_000,   82_592,  6_995.57),
    (90_000, 100_000,   55_976,  5_300.31),
    (100_000, 150_000, 118_067, 14_099.60),
    (150_000, 200_000,  35_912,  6_143.20),
    (200_000, 275_000,  20_798,  4_814.03),
    (275_000, 550_000,  21_842, 11_788.31),  # "Over 275k"; avg ~€540k
]

# 2026 Ready Reckoner projected taxpayer units (page 3)
# (lower, upper, num_units, total_income_€m, combined_it_usc_€m)
TAXPAYER_UNITS_2026 = [
    (0,       10_000,  481_000,   2_219,     0.1),
    (10_000,  13_000,  138_800,   1_614,     0.1),
    (13_000,  15_000,  154_000,   2_174,     9.4),
    (15_000,  18_000,  189_000,   3_097,    17),
    (18_000,  20_000,   94_000,   1_786,    16),
    (20_000,  25_000,  230_600,   5_176,   129),
    (25_000,  27_000,   86_800,   2_256,    99),
    (27_000,  30_000,  148_100,   4_226,   214),
    (30_000,  35_000,  194_500,   6_301,   409),
    (35_000,  40_000,  205_800,   7_701,   642),
    (40_000,  50_000,  351_100,  15_669, 1_641),
    (50_000,  60_000,  249_300,  13_719, 1_909),
    (60_000,  70_000,  196_600,  12_739, 2_049),
    (70_000,  75_000,   82_800,   5_989, 1_033),
    (75_000,  80_000,   70_900,   5_493,   998),
    (80_000,  90_000,  111_200,   9_427, 1_781),
    (90_000, 100_000,   89_200,   8_445, 1_686),
    (100_000, 150_000, 243_400,  29_280, 7_122),
    (150_000, 200_000,  85_800,  14_679, 4_455),
    (200_000, 275_000,  46_600,  10_769, 3_752),
    (275_000, 550_000,  43_800,  22_650, 9_526),
]

# ============================================================
# TAX PARAMETERS
# ============================================================

@dataclass
class USCParams:
    """USC rate and band structure."""
    exemption_threshold: float = 13_000.0
    # Band boundaries (upper limits of each band)
    band1_limit: float = 12_012.0       # first €12,012
    band2_upper: float = 28_700.0       # €12,012 to €28,700
    band3_upper: float = 70_044.0       # €28,700 to €70,044
    # Rates
    band1_rate: float = 0.005           # 0.5%
    band2_rate: float = 0.02            # 2%
    band3_rate: float = 0.03            # 3%
    band4_rate: float = 0.08            # 8%


# 2024 parameters (base for pre-budget 2025 costings)
USC_2024 = USCParams(
    band1_limit=12_012, band2_upper=25_760, band3_upper=70_044,
    band1_rate=0.005, band2_rate=0.02, band3_rate=0.04, band4_rate=0.08,
)

# 2025 parameters
USC_2025 = USCParams(
    band1_limit=12_012, band2_upper=27_382, band3_upper=70_044,
    band1_rate=0.005, band2_rate=0.02, band3_rate=0.03, band4_rate=0.08,
)

# 2026 parameters (current)
USC_2026 = USCParams(
    band1_limit=12_012, band2_upper=28_700, band3_upper=70_044,
    band1_rate=0.005, band2_rate=0.02, band3_rate=0.03, band4_rate=0.08,
)


# ============================================================
# USC MICROSIMULATION
# ============================================================

def build_synthetic_population(dist_data, n_points=500):
    """
    Build a synthetic population from grouped income distribution.

    Within each band, distributes individuals using the actual average
    income (from total income / count) as the distribution centre.
    Uses a triangular-like spread within the band bounds.

    Returns: (incomes array, weights array)
    """
    all_incomes = []
    all_weights = []

    for lower, upper, count, total_income_m, *_ in dist_data:
        if count == 0:
            continue

        avg = (total_income_m * 1e6) / count
        n = n_points

        # Spread uniformly within band
        incomes = np.linspace(lower + 0.5, upper - 0.5, n)

        # Shift to match observed average income
        current_avg = np.mean(incomes)
        if current_avg > 0:
            # Scale incomes to match the actual average
            # But keep within bounds
            scale = avg / current_avg
            incomes = incomes * scale
            # Clip to reasonable bounds (allow some overshoot for top band)
            incomes = np.clip(incomes, lower, max(upper, avg * 1.5))

        weight_per_point = count / n
        all_incomes.append(incomes)
        all_weights.append(np.full(n, weight_per_point))

    return np.concatenate(all_incomes), np.concatenate(all_weights)


def calc_usc(incomes, weights, params):
    """
    Vectorised USC calculation.

    Returns total USC yield in euros.
    """
    exempt = incomes <= params.exemption_threshold
    liable = ~exempt

    # Band 1: 0 to band1_limit
    b1 = np.minimum(incomes, params.band1_limit) * params.band1_rate

    # Band 2: band1_limit to band2_upper
    b2_width = params.band2_upper - params.band1_limit
    b2 = np.clip(incomes - params.band1_limit, 0, b2_width) * params.band2_rate

    # Band 3: band2_upper to band3_upper
    b3_width = params.band3_upper - params.band2_upper
    b3 = np.clip(incomes - params.band2_upper, 0, b3_width) * params.band3_rate

    # Band 4: above band3_upper
    b4 = np.clip(incomes - params.band3_upper, 0, None) * params.band4_rate

    usc_per_person = (b1 + b2 + b3 + b4) * liable
    return float(np.sum(usc_per_person * weights))


def cost_usc_change(baseline, counterfactual, dist_data=None,
                    scale_to_2026=True, calibrate=True):
    """
    Cost a USC parameter change via microsimulation.

    Uses 2023 individual data scaled to approximate 2026 levels.
    Applies calibration factors derived from Ready Reckoner 2026 to correct
    for individual-vs-unit differences and income projection uncertainty.

    Returns dict with full_year cost in €m and component breakdown.
    """
    if dist_data is None:
        dist_data = INDIVIDUALS_2023

    incomes, weights = build_synthetic_population(dist_data)

    # Scale to 2026: income growth factor from 2023 to 2026
    # 2023 total: €154,820m; 2026 projected: €185,410m
    if scale_to_2026:
        income_growth = 185_410 / 154_820
        incomes = incomes * income_growth
        # Population growth: 3,948,197 → ~4,100,000 (estimate)
        pop_growth = 4_100_000 / 3_948_197
        weights = weights * pop_growth

    baseline_yield = calc_usc(incomes, weights, baseline) / 1e6
    new_yield = calc_usc(incomes, weights, counterfactual) / 1e6
    raw_cost = baseline_yield - new_yield  # positive = costs exchequer

    if calibrate and raw_cost != 0:
        # Decompose the cost by band to apply per-band calibration.
        # Calibration factors: Ready Reckoner value / raw model value
        # Derived from: 0.5%→0% (180/209), 2%→1% (455/485),
        #               3%→2% (482/579), 8%→7% (360/406)
        # These correct for reduced-rate USC, surcharge, and data differences.
        cal = _calibrated_usc_cost(incomes, weights, baseline, counterfactual)
        cost = cal
    else:
        cost = raw_cost

    return {
        'baseline_usc_€m': round(baseline_yield, 1),
        'new_usc_€m': round(new_yield, 1),
        'full_year_cost_€m': round(cost, 0),
    }


def _calibrated_usc_cost(incomes, weights, baseline, counterfactual):
    """
    Calculate calibrated USC cost by decomposing into per-band contributions
    and applying Ready-Reckoner-derived calibration factors.
    """
    # Calibration factors per band (RR target / raw model output)
    # Band 1 (0.5%): 180/209 = 0.861
    # Band 2 (2%):   455/485 = 0.938
    # Band 3 (3%):   482/579 = 0.832
    # Band 4 (8%):   360/406 = 0.887
    CAL = {1: 0.861, 2: 0.938, 3: 0.832, 4: 0.887}

    exempt_b = incomes <= baseline.exemption_threshold
    exempt_c = incomes <= counterfactual.exemption_threshold
    liable_b = ~exempt_b
    liable_c = ~exempt_c

    def band_usc(inc, params, liable_mask):
        b1 = np.minimum(inc, params.band1_limit) * params.band1_rate * liable_mask
        b2_w = params.band2_upper - params.band1_limit
        b2 = np.clip(inc - params.band1_limit, 0, b2_w) * params.band2_rate * liable_mask
        b3_w = params.band3_upper - params.band2_upper
        b3 = np.clip(inc - params.band2_upper, 0, b3_w) * params.band3_rate * liable_mask
        b4 = np.clip(inc - params.band3_upper, 0, None) * params.band4_rate * liable_mask
        return b1, b2, b3, b4

    bb1, bb2, bb3, bb4 = band_usc(incomes, baseline, liable_b)
    cb1, cb2, cb3, cb4 = band_usc(incomes, counterfactual, liable_c)

    diff1 = np.sum((bb1 - cb1) * weights) / 1e6
    diff2 = np.sum((bb2 - cb2) * weights) / 1e6
    diff3 = np.sum((bb3 - cb3) * weights) / 1e6
    diff4 = np.sum((bb4 - cb4) * weights) / 1e6

    calibrated = diff1 * CAL[1] + diff2 * CAL[2] + diff3 * CAL[3] + diff4 * CAL[4]
    return calibrated


# ============================================================
# INCOME TAX: READY RECKONER INTERPOLATION
# ============================================================

# All values from Ready Reckoner Post-Budget 2026 (€ million)

def cost_it_rate_change(rate, change_pp):
    """
    Cost an income tax rate change.

    rate: '20' or '40'
    change_pp: percentage points change (negative = cut, positive = increase)

    Returns: dict with first_year and full_year in €m
    Positive = costs exchequer (cut), negative = yields revenue (increase).
    Sign convention matches all other cost functions.
    """
    # RR data: cost of a 1pp CUT (positive = costs exchequer)
    costs_per_1pp = {
        '20': {'first_year': 936, 'full_year': 1_070},
        '40': {'first_year': 482, 'full_year': 567},
    }
    # RR data: yield of a 1pp INCREASE (positive = yields for exchequer)
    yields_per_1pp = {
        '20': {'first_year': 948, 'full_year': 1_085},
        '40': {'first_year': 482, 'full_year': 567},
    }

    if change_pp < 0:
        # Cut: costs the exchequer → positive result
        ref = costs_per_1pp[rate]
        return {
            'first_year_€m': round(abs(change_pp) * ref['first_year'], 0),
            'full_year_€m': round(abs(change_pp) * ref['full_year'], 0),
        }
    else:
        # Increase: yields for exchequer → negative result
        ref = yields_per_1pp[rate]
        return {
            'first_year_€m': round(-change_pp * ref['first_year'], 0),
            'full_year_€m': round(-change_pp * ref['full_year'], 0),
        }


def cost_it_band_change(increase):
    """
    Cost of widening the standard rate band (all categories combined).

    increase: € amount of band widening (e.g., 1000 for +€1,000)
    Proportionate increase across single, married one-earner, married two-earner.

    Returns: dict with first_year and full_year in €m (positive = costs exchequer)
    """
    # Known data points: (increase, first_year, full_year)
    points = [(100, 24, 27), (500, 117, 134), (1000, 232, 265), (1500, 344, 393)]
    amounts = [p[0] for p in points]
    fy = [p[1] for p in points]
    full = [p[2] for p in points]

    if increase <= 1500:
        fy_cost = float(np.interp(increase, amounts, fy))
        full_cost = float(np.interp(increase, amounts, full))
    else:
        # Extrapolate using marginal cost from last segment
        # (diminishing returns at higher levels)
        slope_fy = (fy[-1] - fy[-2]) / (amounts[-1] - amounts[-2])
        slope_full = (full[-1] - full[-2]) / (amounts[-1] - amounts[-2])
        fy_cost = fy[-1] + slope_fy * (increase - 1500)
        full_cost = full[-1] + slope_full * (increase - 1500)

    return {'first_year_€m': round(fy_cost, 0), 'full_year_€m': round(full_cost, 0)}


# Band widening broken down by taxpayer category
def cost_it_band_change_detail(increase):
    """Band widening with breakdown by category."""
    single_pts = [(100, 11, 12), (500, 52, 59), (1000, 103, 116), (1500, 152, 173)]
    married_one_pts = [(100, 2.5, 2.8), (500, 12, 14), (1000, 24, 28), (1500, 36, 42)]
    married_two_pts = [(100, 11, 12), (500, 53, 61), (1000, 105, 121), (1500, 156, 179)]

    def interp(pts, inc):
        a = [p[0] for p in pts]
        f = [p[1] for p in pts]
        g = [p[2] for p in pts]
        if inc <= a[-1]:
            return float(np.interp(inc, a, f)), float(np.interp(inc, a, g))
        # Extrapolate from last two points
        sf = (f[-1] - f[-2]) / (a[-1] - a[-2])
        sg = (g[-1] - g[-2]) / (a[-1] - a[-2])
        return f[-1] + sf * (inc - a[-1]), g[-1] + sg * (inc - a[-1])

    s_fy, s_full = interp(single_pts, increase)
    m1_fy, m1_full = interp(married_one_pts, increase)
    m2_fy, m2_full = interp(married_two_pts, increase)

    return {
        'single': {'first_year_€m': round(s_fy, 1), 'full_year_€m': round(s_full, 1)},
        'married_one_earner': {'first_year_€m': round(m1_fy, 1), 'full_year_€m': round(m1_full, 1)},
        'married_two_earner': {'first_year_€m': round(m2_fy, 1), 'full_year_€m': round(m2_full, 1)},
        'total': {
            'first_year_€m': round(s_fy + m1_fy + m2_fy, 0),
            'full_year_€m': round(s_full + m1_full + m2_full, 0),
        },
    }


def cost_credit_change(credit, amount):
    """
    Cost of increasing a tax credit.

    credit: one of 'single_person', 'married', 'employee', 'earned_income',
            'home_carer', 'age', 'rent'
    amount: € increase

    Returns: dict with first_year and full_year in €m
    """
    # (unit_increase, first_year_per_unit, full_year_per_unit)
    credit_data = {
        'single_person':  (100, 112, 127),
        'married':        (200, 150, 172),
        'employee':       (50,  110, 125),
        'earned_income':  (50,    8,  12),
        'home_carer':     (50,  3.0, 3.4),
        'age':            (50,   20,  24),     # per €50 single / €100 joint
        'rent':           (100,  25,  25),     # per €100 single / €200 joint
    }

    if credit not in credit_data:
        raise ValueError(f"Unknown credit: {credit}. Options: {list(credit_data.keys())}")

    unit, fy_per_unit, full_per_unit = credit_data[credit]
    multiplier = amount / unit

    return {
        'first_year_€m': round(multiplier * fy_per_unit, 1),
        'full_year_€m': round(multiplier * full_per_unit, 1),
    }


def cost_usc_band_change(band, amount):
    """
    Cost of widening a USC rate band.

    band: '0.5%_upper', '2%_both', '2%_upper', '3%', '8%_lower'
    amount: € increase in band width

    Returns: dict with first_year and full_year in €m
    """
    band_data = {
        'exemption': [(100, 0.4, 0.5), (500, 2.2, 2.5), (1000, 5, 6), (1500, 8, 9)],
        '0.5%_upper': [(100, 5, 5), (500, 20, 23), (1000, 39, 45), (1500, 59, 68)],
        '2%_both': [(100, 6, 7), (500, 29, 33), (1000, 56, 64), (1500, 84, 97)],
        '2%_upper': [(100, 1.7, 1.9), (500, 8, 10), (1000, 17, 19), (1500, 25, 29)],
        '3%': [(100, 4.2, 4.8), (500, 21, 24), (1000, 42, 48), (1500, 62, 71)],
        '8%_lower': [(500, 13, 14), (1000, 25, 29), (2000, 49, 56), (5000, 117, 135)],
    }

    if band not in band_data:
        raise ValueError(f"Unknown band: {band}. Options: {list(band_data.keys())}")

    pts = band_data[band]
    amounts = [p[0] for p in pts]
    fy_vals = [p[1] for p in pts]
    full_vals = [p[2] for p in pts]

    if amount <= amounts[-1]:
        fy = float(np.interp(amount, amounts, fy_vals))
        full = float(np.interp(amount, amounts, full_vals))
    else:
        # Linear extrapolation from last two data points
        slope_fy = (fy_vals[-1] - fy_vals[-2]) / (amounts[-1] - amounts[-2])
        slope_full = (full_vals[-1] - full_vals[-2]) / (amounts[-1] - amounts[-2])
        fy = fy_vals[-1] + slope_fy * (amount - amounts[-1])
        full = full_vals[-1] + slope_full * (amount - amounts[-1])

    return {'first_year_€m': round(fy, 1), 'full_year_€m': round(full, 1)}


def cost_indexation(pct):
    """
    Cost of indexing the tax system by a given percentage.

    pct: percentage (e.g., 1.0 for 1%, 2.5 for 2.5%)

    Based on Ready Reckoner "Cost of Indexation at 1%".
    Scales linearly.
    """
    # Per 1% indexation (full year €m)
    components = {
        'Personal credits + rate bands + SPCCC': (164, 188),
        'Employee credit + personal credits + bands': (208, 238),
        'Earned Income Credit': (3.1, 4.3),
        'USC bands and exemption limits': (29, 33),
    }

    scale = pct / 1.0
    results = {}
    for desc, (fy, full) in components.items():
        results[desc] = {
            'first_year_€m': round(fy * scale, 1),
            'full_year_€m': round(full * scale, 1),
        }
    return results


# ============================================================
# PACKAGE COSTING ENGINE
# ============================================================

def cost_package(changes):
    """
    Cost a package of tax changes.

    changes: list of dicts, each with 'type' and parameters:

    USC rate change:
        {'type': 'usc_rate', 'band': 3, 'new_rate': 0.02}
        Band numbers: 1 (0.5%), 2 (2%), 3 (3%), 4 (8%)

    USC band widening:
        {'type': 'usc_band', 'band': '2%_upper', 'amount': 1000}

    Income tax rate change:
        {'type': 'it_rate', 'rate': '40', 'change_pp': -1}
        Negative = cut (costs), positive = increase (yields)

    Income tax band widening:
        {'type': 'it_band', 'increase': 1000}

    Tax credit change:
        {'type': 'credit', 'credit': 'employee', 'amount': 100}

    Returns: dict with itemised costs and totals
    """
    items = []
    total_fy = 0
    total_full = 0

    # First-year / full-year ratios by USC band (from pre-budget analysis)
    usc_fy_ratios = {1: 0.867, 2: 0.870, 3: 0.874, 4: 0.831}

    for c in changes:
        if c['type'] == 'usc_rate':
            band_num = c['band']
            new_rate = c['new_rate']
            band_attr = {1: 'band1_rate', 2: 'band2_rate',
                         3: 'band3_rate', 4: 'band4_rate'}
            rate_labels = {1: '0.5%', 2: '2%', 3: '3%', 4: '8%'}

            baseline = USCParams()
            counterfactual = USCParams()
            setattr(counterfactual, band_attr[band_num], new_rate)

            result = cost_usc_change(baseline, counterfactual)
            full = result['full_year_cost_€m']
            ratio = usc_fy_ratios.get(band_num, 0.87)
            fy = round(full * ratio, 0)

            desc = f"USC {rate_labels[band_num]} rate → {new_rate*100:.1f}%"
            items.append({'description': desc, 'first_year_€m': fy, 'full_year_€m': full})
            total_fy += fy
            total_full += full

        elif c['type'] == 'usc_band':
            result = cost_usc_band_change(c['band'], c['amount'])
            desc = f"USC {c['band']} band +€{c['amount']:,}"
            items.append({'description': desc, **result})
            total_fy += result['first_year_€m']
            total_full += result['full_year_€m']

        elif c['type'] == 'it_rate':
            result = cost_it_rate_change(c['rate'], c['change_pp'])
            direction = "cut" if c['change_pp'] < 0 else "increase"
            desc = f"IT {c['rate']}% rate {direction} {abs(c['change_pp'])}pp"
            items.append({'description': desc, **result})
            total_fy += result['first_year_€m']
            total_full += result['full_year_€m']

        elif c['type'] == 'it_band':
            result = cost_it_band_change(c['increase'])
            desc = f"IT standard band +€{c['increase']:,}"
            items.append({'description': desc, **result})
            total_fy += result['first_year_€m']
            total_full += result['full_year_€m']

        elif c['type'] == 'credit':
            result = cost_credit_change(c['credit'], c['amount'])
            desc = f"{c['credit'].replace('_', ' ').title()} credit +€{c['amount']}"
            items.append({'description': desc, **result})
            total_fy += result['first_year_€m']
            total_full += result['full_year_€m']

        else:
            raise ValueError(f"Unknown change type: {c['type']}")

    return {
        'items': items,
        'total_first_year_€m': round(total_fy, 0),
        'total_full_year_€m': round(total_full, 0),
    }


def print_result(result):
    """Pretty-print a package costing result."""
    print()
    print("=" * 72)
    print(f"  {'MEASURE':<42} {'FIRST YR':>10} {'FULL YR':>10}")
    print("=" * 72)

    for item in result['items']:
        fy = item['first_year_€m']
        full = item['full_year_€m']
        print(f"  {item['description']:<42} €{abs(fy):>7.0f}m   €{abs(full):>7.0f}m")

    print("-" * 72)
    fy_tot = result['total_first_year_€m']
    full_tot = result['total_full_year_€m']
    label = "TOTAL COST" if full_tot > 0 else "TOTAL YIELD"
    print(f"  {label:<42} €{abs(fy_tot):>7.0f}m   €{abs(full_tot):>7.0f}m")
    print("=" * 72)


# ============================================================
# INDIVIDUAL TAX CALCULATIONS (2026)
# ============================================================

@dataclass
class IncomeTaxParams:
    """Income tax parameters for individual tax calculation."""
    standard_rate: float = 0.20
    higher_rate: float = 0.40
    single_band: float = 44_000
    married_one_earner_band: float = 53_000
    married_two_earner_band: float = 35_000  # max per spouse
    single_person_credit: float = 2_000
    married_credit: float = 4_000
    employee_credit: float = 2_000
    earned_income_credit: float = 2_000
    widowed_credit: float = 2_540


IT_2026 = IncomeTaxParams()

PRSI_RATE = 0.04
PRSI_ANNUAL_THRESHOLD = 18_304  # €352/week
PRSI_CREDIT_MAX = 624  # €12/week, tapers to zero at €22,048

# 2023 individualised gross income by personal status
# Source: Revenue - Individualised Gross Incomes (2023), Page 3
# Each entry: (lower, upper, count, total_gross_income_€m)
INDIVIDUALS_BY_STATUS_2023 = {
    'single_male': [
        (0, 10_000, 213_062, 979.83), (10_000, 12_000, 37_172, 410.47),
        (12_000, 15_000, 78_426, 1_059.12), (15_000, 17_000, 49_852, 790.47),
        (17_000, 20_000, 53_247, 983.86), (20_000, 25_000, 86_468, 1_948.20),
        (25_000, 27_000, 35_811, 931.76), (27_000, 30_000, 55_036, 1_568.65),
        (30_000, 35_000, 88_222, 2_862.58), (35_000, 40_000, 78_638, 2_946.65),
        (40_000, 50_000, 107_010, 4_765.12), (50_000, 60_000, 61_756, 3_371.84),
        (60_000, 70_000, 38_669, 2_500.66), (70_000, 75_000, 14_155, 1_025.11),
        (75_000, 80_000, 11_399, 882.73), (80_000, 90_000, 17_013, 1_440.34),
        (90_000, 100_000, 11_350, 1_074.16), (100_000, 150_000, 22_367, 2_661.04),
        (150_000, 200_000, 5_851, 999.41), (200_000, 275_000, 2_885, 664.18),
        (275_000, 550_000, 2_388, 1_187.22),
    ],
    'single_female': [
        (0, 10_000, 219_330, 999.20), (10_000, 12_000, 41_283, 457.25),
        (12_000, 15_000, 89_865, 1_217.25), (15_000, 17_000, 59_166, 940.77),
        (17_000, 20_000, 56_601, 1_045.79), (20_000, 25_000, 87_378, 1_967.74),
        (25_000, 27_000, 35_839, 931.86), (27_000, 30_000, 52_423, 1_493.01),
        (30_000, 35_000, 72_955, 2_363.50), (35_000, 40_000, 59_395, 2_223.84),
        (40_000, 50_000, 90_302, 4_023.99), (50_000, 60_000, 54_224, 2_963.16),
        (60_000, 70_000, 33_261, 2_147.23), (70_000, 75_000, 11_287, 817.12),
        (75_000, 80_000, 8_558, 662.33), (80_000, 90_000, 12_171, 1_030.20),
        (90_000, 100_000, 7_768, 734.68), (100_000, 150_000, 14_356, 1_700.11),
        (150_000, 200_000, 3_663, 624.30), (200_000, 275_000, 1_824, 420.14),
        (275_000, 550_000, 1_479, 695.03),
    ],
    'married_both': [
        (0, 10_000, 86_603, 457.09), (10_000, 12_000, 32_919, 367.05),
        (12_000, 15_000, 136_348, 1_856.21), (15_000, 17_000, 45_108, 720.32),
        (17_000, 20_000, 59_262, 1_094.91), (20_000, 25_000, 94_666, 2_130.74),
        (25_000, 27_000, 39_042, 1_015.69), (27_000, 30_000, 60_404, 1_722.54),
        (30_000, 35_000, 104_644, 3_398.71), (35_000, 40_000, 99_747, 3_740.59),
        (40_000, 50_000, 168_752, 7_549.30), (50_000, 60_000, 117_965, 6_455.39),
        (60_000, 70_000, 83_530, 5_404.43), (70_000, 75_000, 30_974, 2_243.08),
        (75_000, 80_000, 25_427, 1_968.48), (80_000, 90_000, 39_788, 3_370.44),
        (90_000, 100_000, 27_255, 2_581.36), (100_000, 150_000, 58_485, 6_991.52),
        (150_000, 200_000, 18_342, 3_137.54), (200_000, 275_000, 10_862, 2_515.21),
        (275_000, 550_000, 11_701, 6_241.42),
    ],
    'married_one': [
        (0, 10_000, 31_535, 152.03), (10_000, 12_000, 6_967, 77.38),
        (12_000, 15_000, 21_724, 295.86), (15_000, 17_000, 10_670, 169.60),
        (17_000, 20_000, 12_843, 238.25), (20_000, 25_000, 25_181, 566.15),
        (25_000, 27_000, 12_788, 335.06), (27_000, 30_000, 18_608, 528.88),
        (30_000, 35_000, 27_634, 897.45), (35_000, 40_000, 26_149, 980.09),
        (40_000, 50_000, 42_366, 1_894.86), (50_000, 60_000, 29_880, 1_636.46),
        (60_000, 70_000, 21_709, 1_405.72), (70_000, 75_000, 8_496, 615.18),
        (75_000, 80_000, 7_059, 546.78), (80_000, 90_000, 11_727, 994.19),
        (90_000, 100_000, 8_361, 792.43), (100_000, 150_000, 20_487, 2_465.09),
        (150_000, 200_000, 7_383, 1_266.61), (200_000, 275_000, 4_843, 1_125.55),
        (275_000, 550_000, 5_859, 3_413.74),
    ],
    'widower': [
        (0, 10_000, 2_414, 12.02), (10_000, 12_000, 969, 10.87),
        (12_000, 15_000, 4_802, 65.68), (15_000, 17_000, 5_491, 87.26),
        (17_000, 20_000, 3_682, 67.80), (20_000, 25_000, 4_608, 103.02),
        (25_000, 27_000, 1_400, 36.40), (27_000, 30_000, 1_985, 56.58),
        (30_000, 35_000, 3_020, 98.13), (35_000, 40_000, 2_730, 102.21),
        (40_000, 50_000, 4_330, 194.43), (50_000, 60_000, 3_343, 182.60),
        (60_000, 70_000, 1_950, 125.97), (70_000, 75_000, 732, 52.95),
        (75_000, 80_000, 531, 41.08), (80_000, 90_000, 754, 63.95),
        (90_000, 100_000, 522, 49.51), (100_000, 150_000, 1_109, 132.38),
        (150_000, 200_000, 328, 56.23), (200_000, 275_000, 187, 43.21),
        (275_000, 550_000, 238, 143.94),
    ],
    'widow': [
        (0, 10_000, 4_446, 22.66), (10_000, 12_000, 3_395, 38.28),
        (12_000, 15_000, 12_316, 169.33), (15_000, 17_000, 17_447, 277.21),
        (17_000, 20_000, 9_580, 176.44), (20_000, 25_000, 12_254, 274.48),
        (25_000, 27_000, 4_211, 109.37), (27_000, 30_000, 5_335, 151.85),
        (30_000, 35_000, 7_700, 249.52), (35_000, 40_000, 5_974, 223.40),
        (40_000, 50_000, 8_081, 360.61), (50_000, 60_000, 5_387, 294.66),
        (60_000, 70_000, 3_141, 202.48), (70_000, 75_000, 1_098, 79.49),
        (75_000, 80_000, 868, 67.25), (80_000, 90_000, 1_139, 96.45),
        (90_000, 100_000, 720, 68.16), (100_000, 150_000, 1_263, 149.46),
        (150_000, 200_000, 345, 59.11), (200_000, 275_000, 197, 45.74),
        (275_000, 550_000, 177, 106.97),
    ],
}

# Map status keys to tax calculation status
_STATUS_TAX_MAP = {
    'single_male': 'single',
    'single_female': 'single',
    'married_both': 'married_two_earner',
    'married_one': 'married_one_earner',
    'widower': 'widowed',
    'widow': 'widowed',
}


def calc_individual_it(gross, status='single', params=None, employment='paye'):
    """
    Calculate income tax for an individual.

    status: 'single', 'married_one_earner', 'married_two_earner', 'widowed'
    employment: 'paye' or 'self_employed'
    """
    if params is None:
        params = IT_2026

    # Standard rate band
    bands = {
        'single': params.single_band,
        'married_one_earner': params.married_one_earner_band,
        'married_two_earner': params.married_two_earner_band,
        'widowed': params.single_band,
    }
    band = bands.get(status, params.single_band)

    # Gross tax
    standard_portion = min(gross, band)
    higher_portion = max(gross - band, 0)
    gross_tax = standard_portion * params.standard_rate + higher_portion * params.higher_rate

    # Credits
    if status == 'single':
        credits = params.single_person_credit
    elif status == 'married_one_earner':
        credits = params.married_credit
    elif status == 'married_two_earner':
        credits = params.married_credit / 2  # split between spouses
    elif status == 'widowed':
        credits = params.widowed_credit
    else:
        credits = params.single_person_credit

    if employment == 'paye':
        credits += params.employee_credit
    else:
        credits += params.earned_income_credit

    return max(gross_tax - credits, 0)


def calc_individual_usc(gross, params=None):
    """Calculate USC for an individual (scalar version)."""
    if params is None:
        params = USC_2026

    if gross <= params.exemption_threshold:
        return 0.0

    b1 = min(gross, params.band1_limit) * params.band1_rate
    b2 = max(min(gross, params.band2_upper) - params.band1_limit, 0) * params.band2_rate
    b3 = max(min(gross, params.band3_upper) - params.band2_upper, 0) * params.band3_rate
    b4 = max(gross - params.band3_upper, 0) * params.band4_rate
    return b1 + b2 + b3 + b4


def calc_individual_prsi(gross, employment='paye'):
    """
    Calculate PRSI.

    PAYE (Class A): 4% on all earnings if above €352/week (€18,304/year),
    with tapered credit up to €22,048.
    Self-employed (Class S): 4% with minimum €500/year, no tapered credit.
    """
    if employment == 'self_employed':
        # Class S: 4% with minimum €500
        if gross <= 5_000:
            return 0.0
        return max(gross * PRSI_RATE, 500.0)

    # Class A (PAYE)
    if gross <= PRSI_ANNUAL_THRESHOLD:
        return 0.0

    prsi_gross = gross * PRSI_RATE
    credit = max(0, PRSI_CREDIT_MAX - (gross - PRSI_ANNUAL_THRESHOLD) / 6)
    return max(prsi_gross - credit, 0)


def calc_take_home(gross, status='single', employment='paye',
                   it_params=None, usc_params=None):
    """
    Calculate net take-home pay with full breakdown.

    Returns dict with gross, all deductions, net pay, and effective rate.
    """
    it = calc_individual_it(gross, status, it_params, employment)
    usc = calc_individual_usc(gross, usc_params)
    prsi = calc_individual_prsi(gross, employment)
    total_ded = it + usc + prsi
    net = gross - total_ded

    return {
        'gross': gross,
        'income_tax': round(it, 2),
        'usc': round(usc, 2),
        'prsi': round(prsi, 2),
        'total_deductions': round(total_ded, 2),
        'net_pay': round(net, 2),
        'effective_rate': round(total_ded / gross * 100, 1) if gross > 0 else 0,
        'marginal_rate': _marginal_rate(gross, status, employment, it_params, usc_params),
    }


def _marginal_rate(gross, status='single', employment='paye',
                   it_params=None, usc_params=None):
    """Calculate marginal tax rate at a given income level.

    Uses €100 delta to smooth out cliff edges (e.g. USC exemption threshold).
    """
    delta = 100.0
    base = calc_individual_it(gross, status, it_params, employment) + \
           calc_individual_usc(gross, usc_params) + \
           calc_individual_prsi(gross, employment)
    higher = calc_individual_it(gross + delta, status, it_params, employment) + \
             calc_individual_usc(gross + delta, usc_params) + \
             calc_individual_prsi(gross + delta, employment)
    return round((higher - base) / delta * 100, 1)


def distributional_analysis(changes):
    """
    Calculate per-person tax savings by income band from a package of changes.

    Uses 2023 individual data by status, scaled to 2026 income levels.

    changes: same format as cost_package() — list of change dicts.

    Returns list of dicts: band label, count, avg saving, total saving.
    """
    # Build counterfactual parameters from changes
    new_it = IncomeTaxParams()
    new_usc = USCParams()

    for c in changes:
        if c['type'] == 'it_band':
            new_it.single_band += c['increase']
            new_it.married_one_earner_band += c['increase']
            new_it.married_two_earner_band += c['increase']
        elif c['type'] == 'it_rate':
            if c['rate'] == '20':
                new_it.standard_rate += c['change_pp'] / 100
            elif c['rate'] == '40':
                new_it.higher_rate += c['change_pp'] / 100
        elif c['type'] == 'credit':
            credit_attr_map = {
                'single_person': 'single_person_credit',
                'married': 'married_credit',
                'employee': 'employee_credit',
                'earned_income': 'earned_income_credit',
                'home_carer': None,  # not modelled per-individual
            }
            attr = credit_attr_map.get(c['credit'])
            if attr:
                setattr(new_it, attr, getattr(new_it, attr) + c['amount'])
                # Also increase widowed credit when single_person changes
                if c['credit'] == 'single_person':
                    new_it.widowed_credit += c['amount']
        elif c['type'] == 'usc_rate':
            band_attr = {1: 'band1_rate', 2: 'band2_rate',
                         3: 'band3_rate', 4: 'band4_rate'}
            setattr(new_usc, band_attr[c['band']], c['new_rate'])
        elif c['type'] == 'usc_band':
            if c['band'] == 'exemption':
                new_usc.exemption_threshold += c['amount']
            elif c['band'] == '0.5%_upper':
                new_usc.band1_limit += c['amount']
            elif c['band'] in ('2%_upper', '2%_both'):
                new_usc.band2_upper += c['amount']
            elif c['band'] == '3%':
                new_usc.band3_upper += c['amount']
            elif c['band'] == '8%_lower':
                new_usc.band3_upper += c['amount']

    base_it = IT_2026
    base_usc = USC_2026

    # Income growth factor 2023 → 2026
    growth = 185_410 / 154_820

    n_bands = len(INDIVIDUALS_BY_STATUS_2023['single_male'])
    results = []

    for i in range(n_bands):
        lower, upper = INDIVIDUALS_BY_STATUS_2023['single_male'][i][:2]
        total_saving = 0.0
        total_count = 0

        for status_key, tax_status in _STATUS_TAX_MAP.items():
            row = INDIVIDUALS_BY_STATUS_2023[status_key][i]
            count = row[2]
            income_m = row[3]
            if count == 0:
                continue

            avg_income = (income_m * 1e6 / count) * growth

            # Current tax (IT + USC only — PRSI doesn't change)
            it_curr = calc_individual_it(avg_income, tax_status, base_it)
            usc_curr = calc_individual_usc(avg_income, base_usc)

            # New tax
            it_new = calc_individual_it(avg_income, tax_status, new_it)
            usc_new = calc_individual_usc(avg_income, new_usc)

            saving = (it_curr + usc_curr) - (it_new + usc_new)
            total_saving += saving * count
            total_count += count

        avg_saving = total_saving / total_count if total_count > 0 else 0

        if upper >= 550_000:
            label = f"€{lower // 1000}k+"
        else:
            label = f"€{lower // 1000}k–€{upper // 1000}k"

        results.append({
            'band': label,
            'lower': lower,
            'upper': upper,
            'count': total_count,
            'avg_saving': round(avg_saving, 0),
            'total_saving_€m': round(total_saving / 1e6, 1),
        })

    return results


# ============================================================
# VALIDATION
# ============================================================

def validate():
    """Validate model against Ready Reckoner 2026 values."""

    print("\n" + "=" * 72)
    print("  VALIDATION: USC Rate Changes vs Ready Reckoner 2026")
    print("=" * 72)
    print(f"  {'Change':<30} {'Model':>8} {'RR':>8} {'Diff':>8} {'Error':>7}")
    print("-" * 72)

    # Ready Reckoner targets (full year, €m)
    rr_targets = [
        ("0.5% → 0%",   1, 0.000,  180),
        ("2% → 1%",     2, 0.010,  455),   # includes reduced rate
        ("3% → 2%",     3, 0.020,  482),
        ("8% → 7%",     4, 0.070,  360),   # includes surcharge
    ]

    band_attr = {1: 'band1_rate', 2: 'band2_rate', 3: 'band3_rate', 4: 'band4_rate'}

    for desc, band_num, new_rate, rr_val in rr_targets:
        baseline = USCParams()
        counter = USCParams()
        setattr(counter, band_attr[band_num], new_rate)

        result = cost_usc_change(baseline, counter)
        model_val = result['full_year_cost_€m']
        diff = model_val - rr_val
        err = (diff / rr_val) * 100

        print(f"  {desc:<30} {model_val:>7.0f}  {rr_val:>7.0f}  {diff:>+7.0f}  {err:>+6.1f}%")

    # Income tax band widening
    print()
    print("=" * 72)
    print("  VALIDATION: IT Band Widening vs Ready Reckoner 2026")
    print("=" * 72)
    print(f"  {'Change':<30} {'Model FY':>10} {'RR FY':>10} {'Model Full':>12} {'RR Full':>10}")
    print("-" * 72)

    for inc, rr_fy, rr_full in [(500, 117, 134), (1000, 232, 265), (1500, 344, 393)]:
        r = cost_it_band_change(inc)
        print(f"  Band +€{inc:<22,} €{r['first_year_€m']:>7.0f}m  €{rr_fy:>7}m"
              f"    €{r['full_year_€m']:>7.0f}m  €{rr_full:>7}m")

    # USC band widening
    print()
    print("=" * 72)
    print("  VALIDATION: USC Band Widening vs Ready Reckoner 2026")
    print("=" * 72)

    usc_band_tests = [
        ("0.5% upper +€1,000", '0.5%_upper', 1000, 39, 45),
        ("2% upper +€1,000", '2%_upper', 1000, 17, 19),
        ("3% band +€1,000", '3%', 1000, 42, 48),
        ("8% lower +€1,000", '8%_lower', 1000, 25, 29),
    ]

    print(f"  {'Change':<30} {'Model FY':>10} {'RR FY':>10} {'Model Full':>12} {'RR Full':>10}")
    print("-" * 72)
    for desc, band, amt, rr_fy, rr_full in usc_band_tests:
        r = cost_usc_band_change(band, amt)
        print(f"  {desc:<30} €{r['first_year_€m']:>7.1f}m  €{rr_fy:>7}m"
              f"    €{r['full_year_€m']:>7.1f}m  €{rr_full:>7}m")

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    validate()

    # ----------------------------------------------------------
    # EXAMPLE 1: Typical budget tax package
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  EXAMPLE 1: Typical Budget Tax Package")
    print("=" * 72)

    package = [
        {'type': 'it_band', 'increase': 1000},
        {'type': 'credit', 'credit': 'single_person', 'amount': 100},
        {'type': 'credit', 'credit': 'employee', 'amount': 100},
        {'type': 'credit', 'credit': 'earned_income', 'amount': 100},
    ]
    print_result(cost_package(package))

    # ----------------------------------------------------------
    # EXAMPLE 2: USC reform - abolish 3% band
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  EXAMPLE 2: Reduce USC 3% rate to 2%")
    print("=" * 72)

    package2 = [
        {'type': 'usc_rate', 'band': 3, 'new_rate': 0.02},
    ]
    print_result(cost_package(package2))

    # ----------------------------------------------------------
    # EXAMPLE 3: Combined IT and USC reform
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  EXAMPLE 3: Budget Package with IT + USC + Credits")
    print("=" * 72)

    package3 = [
        {'type': 'it_band', 'increase': 2000},
        {'type': 'credit', 'credit': 'single_person', 'amount': 150},
        {'type': 'credit', 'credit': 'employee', 'amount': 150},
        {'type': 'credit', 'credit': 'earned_income', 'amount': 150},
        {'type': 'usc_rate', 'band': 3, 'new_rate': 0.025},
        {'type': 'usc_band', 'band': '2%_upper', 'amount': 500},
    ]
    print_result(cost_package(package3))

    # ----------------------------------------------------------
    # EXAMPLE 4: USC rate sensitivity table
    # ----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  USC RATE SENSITIVITY (Full Year Cost €m)")
    print("=" * 72)
    print(f"  {'Band':<15} {'Cut 0.5pp':>10} {'Cut 1pp':>10} {'Cut 2pp':>10}")
    print("-" * 72)

    band_map = {1: 'band1_rate', 2: 'band2_rate', 3: 'band3_rate', 4: 'band4_rate'}
    labels = {1: '0.5% band', 2: '2% band', 3: '3% band', 4: '8% band'}
    current_rates = {1: 0.005, 2: 0.02, 3: 0.03, 4: 0.08}

    for band_num in [1, 2, 3, 4]:
        row = f"  {labels[band_num]:<15}"
        for cut in [0.005, 0.01, 0.02]:
            new_rate = current_rates[band_num] - cut
            if new_rate < 0:
                row += f"{'N/A':>10}"
                continue
            baseline = USCParams()
            counter = USCParams()
            setattr(counter, band_map[band_num], new_rate)
            result = cost_usc_change(baseline, counter)
            row += f"  €{result['full_year_cost_€m']:>6.0f}m"
        print(row)
    print()
