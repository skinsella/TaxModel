#!/usr/bin/env python3
"""
Irish Tax Costing Model — Interactive Interface
================================================
Streamlit app wrapping the tax_model engine.

Run with:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from tax_model import (
    USCParams, cost_usc_change, cost_it_rate_change, cost_it_band_change,
    cost_it_band_change_detail, cost_credit_change, cost_usc_band_change,
    cost_indexation, cost_package, TAXPAYER_UNITS_2026,
    calc_take_home, distributional_analysis, IncomeTaxParams, IT_2026,
    USC_2026, calc_individual_it, calc_individual_usc, calc_individual_prsi,
)

# ── Page config ──────────────────────────────────────────────

st.set_page_config(
    page_title="Irish Tax Costing Model",
    page_icon="🏛️",
    layout="wide",
)

# ── Sidebar: current parameters ──────────────────────────────

with st.sidebar:
    st.markdown("## Current 2026 Parameters")

    st.markdown("**USC Bands**")
    usc_df = pd.DataFrame({
        'Band': ['0.5%', '2%', '3%', '8%'],
        'From': ['€0', '€12,012', '€28,700', '€70,044'],
        'To': ['€12,012', '€28,700', '€70,044', '—'],
    })
    st.dataframe(usc_df, hide_index=True, use_container_width=True)

    st.markdown("**Income Tax**")
    it_df = pd.DataFrame({
        'Status': ['Single', 'Married (1 earner)', 'Married (2 earners)'],
        'Standard Band': ['€44,000', '€53,000', '€53k / €35k'],
    })
    st.dataframe(it_df, hide_index=True, use_container_width=True)

    st.markdown("**Key Credits (2026)**")
    cr_df = pd.DataFrame({
        'Credit': ['Single Person', 'Married', 'Employee (PAYE)', 'Earned Income'],
        'Amount': ['€2,000', '€4,000', '€2,000', '€2,000'],
    })
    st.dataframe(cr_df, hide_index=True, use_container_width=True)

    st.markdown("**Taxpayer Units:** 3,493,400")
    st.markdown("**Total Income:** €185.4bn")

    st.divider()
    st.caption("Based on Revenue Ready Reckoner Post-Budget 2026 (Oct 2025) "
               "and Individualised Gross Income data (2023).")


# ── Main content ─────────────────────────────────────────────

st.title("Irish Tax Costing Model")
st.markdown("Build a budget package and see the Exchequer cost instantly. "
            "All figures are **€ million** based on 2026 parameters.")

# ── Tabs ─────────────────────────────────────────────────────

tab_pkg, tab_take_home, tab_usc, tab_it, tab_credits, tab_index, tab_dist = st.tabs([
    "Package Builder",
    "Take-Home Calculator",
    "USC Rates & Bands",
    "Income Tax",
    "Tax Credits",
    "Indexation",
    "Income Distribution",
])

# ==============================================================
# TAB 1: PACKAGE BUILDER
# ==============================================================

with tab_pkg:
    st.header("Budget Package Builder")
    st.markdown("Toggle measures on/off and set values. The total cost updates live.")

    changes = []

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income Tax")

        # Band widening
        do_band = st.checkbox("Widen standard rate band", key="pkg_band")
        if do_band:
            band_inc = st.slider("Band increase (€)", 0, 5000, 1000, 100,
                                 key="pkg_band_amt")
            if band_inc > 0:
                changes.append({'type': 'it_band', 'increase': band_inc})

        st.divider()

        # Rate changes
        do_rate20 = st.checkbox("Change 20% rate", key="pkg_rate20")
        if do_rate20:
            rate20_chg = st.slider("20% rate change (pp)", -3.0, 3.0, 0.0, 0.5,
                                   key="pkg_rate20_val")
            if rate20_chg != 0:
                changes.append({'type': 'it_rate', 'rate': '20',
                                'change_pp': rate20_chg})

        do_rate40 = st.checkbox("Change 40% rate", key="pkg_rate40")
        if do_rate40:
            rate40_chg = st.slider("40% rate change (pp)", -3.0, 3.0, 0.0, 0.5,
                                   key="pkg_rate40_val")
            if rate40_chg != 0:
                changes.append({'type': 'it_rate', 'rate': '40',
                                'change_pp': rate40_chg})

        st.divider()

        # Credits
        st.subheader("Tax Credits")

        do_single = st.checkbox("Increase Single Person Credit", key="pkg_cr_s")
        if do_single:
            single_amt = st.slider("Single Person Credit increase (€)",
                                   0, 500, 100, 25, key="pkg_cr_s_val")
            if single_amt > 0:
                changes.append({'type': 'credit', 'credit': 'single_person',
                                'amount': single_amt})

        do_employee = st.checkbox("Increase Employee (PAYE) Credit", key="pkg_cr_e")
        if do_employee:
            emp_amt = st.slider("Employee Credit increase (€)",
                                0, 500, 100, 25, key="pkg_cr_e_val")
            if emp_amt > 0:
                changes.append({'type': 'credit', 'credit': 'employee',
                                'amount': emp_amt})

        do_earned = st.checkbox("Increase Earned Income Credit", key="pkg_cr_ei")
        if do_earned:
            earned_amt = st.slider("Earned Income Credit increase (€)",
                                   0, 500, 100, 25, key="pkg_cr_ei_val")
            if earned_amt > 0:
                changes.append({'type': 'credit', 'credit': 'earned_income',
                                'amount': earned_amt})

        do_hcarer = st.checkbox("Increase Home Carer Credit", key="pkg_cr_hc")
        if do_hcarer:
            hc_amt = st.slider("Home Carer Credit increase (€)",
                               0, 500, 100, 25, key="pkg_cr_hc_val")
            if hc_amt > 0:
                changes.append({'type': 'credit', 'credit': 'home_carer',
                                'amount': hc_amt})

    with col2:
        st.subheader("USC Rates")

        do_usc1 = st.checkbox("Change 0.5% USC rate", key="pkg_usc1")
        if do_usc1:
            usc1_new = st.slider("New 0.5% band rate (%)", 0.0, 0.5, 0.0, 0.1,
                                 key="pkg_usc1_val")
            changes.append({'type': 'usc_rate', 'band': 1,
                            'new_rate': usc1_new / 100})

        do_usc2 = st.checkbox("Change 2% USC rate", key="pkg_usc2")
        if do_usc2:
            usc2_new = st.slider("New 2% band rate (%)", 0.0, 3.0, 1.0, 0.25,
                                 key="pkg_usc2_val")
            changes.append({'type': 'usc_rate', 'band': 2,
                            'new_rate': usc2_new / 100})

        do_usc3 = st.checkbox("Change 3% USC rate", key="pkg_usc3")
        if do_usc3:
            usc3_new = st.slider("New 3% band rate (%)", 0.0, 4.0, 2.0, 0.25,
                                 key="pkg_usc3_val")
            changes.append({'type': 'usc_rate', 'band': 3,
                            'new_rate': usc3_new / 100})

        do_usc4 = st.checkbox("Change 8% USC rate", key="pkg_usc4")
        if do_usc4:
            usc4_new = st.slider("New 8% band rate (%)", 4.0, 10.0, 7.0, 0.5,
                                 key="pkg_usc4_val")
            changes.append({'type': 'usc_rate', 'band': 4,
                            'new_rate': usc4_new / 100})

        st.divider()
        st.subheader("USC Bands")

        do_usc_b2 = st.checkbox("Widen 2% band (upper end)", key="pkg_usc_b2")
        if do_usc_b2:
            usc_b2_amt = st.slider("2% band upper increase (€)",
                                   0, 3000, 500, 100, key="pkg_usc_b2_val")
            if usc_b2_amt > 0:
                changes.append({'type': 'usc_band', 'band': '2%_upper',
                                'amount': usc_b2_amt})

        do_usc_b3 = st.checkbox("Widen 3% band", key="pkg_usc_b3")
        if do_usc_b3:
            usc_b3_amt = st.slider("3% band increase (€)",
                                   0, 3000, 500, 100, key="pkg_usc_b3_val")
            if usc_b3_amt > 0:
                changes.append({'type': 'usc_band', 'band': '3%',
                                'amount': usc_b3_amt})

        do_usc_b8 = st.checkbox("Raise 8% threshold", key="pkg_usc_b8")
        if do_usc_b8:
            usc_b8_amt = st.slider("8% threshold increase (€)",
                                   0, 10000, 1000, 500, key="pkg_usc_b8_val")
            if usc_b8_amt > 0:
                changes.append({'type': 'usc_band', 'band': '8%_lower',
                                'amount': usc_b8_amt})

    # ── Results ──────────────────────────────────────────────

    st.divider()

    if changes:
        result = cost_package(changes)

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("First Year Cost",
                       f"€{abs(result['total_first_year_€m']):,.0f}m",
                       delta=None)
        with mc2:
            st.metric("Full Year Cost",
                       f"€{abs(result['total_full_year_€m']):,.0f}m",
                       delta=None)
        with mc3:
            n_measures = len(result['items'])
            st.metric("Measures", n_measures)

        # Detail table
        rows = []
        for item in result['items']:
            rows.append({
                'Measure': item['description'],
                'First Year (€m)': abs(item['first_year_€m']),
                'Full Year (€m)': abs(item['full_year_€m']),
                'Direction': 'Cost' if item['full_year_€m'] > 0 else 'Yield',
            })
        rows.append({
            'Measure': 'TOTAL',
            'First Year (€m)': abs(result['total_first_year_€m']),
            'Full Year (€m)': abs(result['total_full_year_€m']),
            'Direction': 'Cost' if result['total_full_year_€m'] > 0 else 'Yield',
        })

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.apply(
                lambda row: ['font-weight: bold'] * len(row)
                if row['Measure'] == 'TOTAL' else [''] * len(row),
                axis=1
            ),
            hide_index=True,
            use_container_width=True,
        )

        # Bar chart
        chart_df = pd.DataFrame([
            {'Measure': item['description'],
             'Full Year (€m)': abs(item['full_year_€m'])}
            for item in result['items']
        ])
        st.bar_chart(chart_df.set_index('Measure'), horizontal=True)

        # ── Distributional Analysis ──────────────────────────────
        st.divider()
        st.subheader("Who Benefits? — Distributional Analysis")
        st.markdown("Average annual saving per person, by gross income band. "
                     "Based on 2023 individual data scaled to 2026 income levels.")

        dist_results = distributional_analysis(changes)

        # Filter to bands with non-zero savings
        dist_df = pd.DataFrame(dist_results)
        dist_df = dist_df[dist_df['avg_saving'] != 0]

        if not dist_df.empty:
            dc1, dc2 = st.columns([2, 1])
            with dc1:
                chart_data = dist_df[['band', 'avg_saving']].copy()
                chart_data.columns = ['Income Band', 'Avg Saving (€/year)']
                st.bar_chart(chart_data.set_index('Income Band'))

            with dc2:
                st.markdown("**Per-person savings by income band**")
                display_df = dist_df[['band', 'count', 'avg_saving', 'total_saving_€m']].copy()
                display_df.columns = ['Income Band', 'Individuals', 'Avg Saving (€)', 'Total (€m)']
                display_df['Individuals'] = display_df['Individuals'].apply(lambda x: f"{x:,.0f}")
                display_df['Avg Saving (€)'] = display_df['Avg Saving (€)'].apply(lambda x: f"€{x:,.0f}")
                display_df['Total (€m)'] = display_df['Total (€m)'].apply(lambda x: f"€{x:,.1f}m")
                st.dataframe(display_df, hide_index=True, use_container_width=True,
                             height=min(len(display_df) * 35 + 38, 700))

            # Summary metrics
            total_dist_cost = dist_df['total_saving_€m'].sum()
            max_band = dist_df.loc[dist_df['avg_saving'].idxmax()]
            st.markdown(f"**Largest per-person benefit:** {max_band['band']} "
                        f"(€{max_band['avg_saving']:,.0f}/year)")
        else:
            st.info("No distributional impact for the selected measures.")

    else:
        st.info("Select measures above to build a budget package. "
                "The cost will appear here.")


# ==============================================================
# TAB 2: TAKE-HOME CALCULATOR
# ==============================================================

with tab_take_home:
    st.header("Net Take-Home Pay Calculator")
    st.markdown("Calculate your net pay under current 2026 parameters, "
                "or compare current vs proposed changes.")

    th_col1, th_col2 = st.columns(2)

    with th_col1:
        st.subheader("Your Details")
        gross_salary = st.number_input("Annual gross salary (€)", 0, 1_000_000,
                                        50_000, 1_000, key="th_gross")
        th_status = st.selectbox("Filing status", [
            'Single', 'Married (one earner)', 'Married (two earners)', 'Widowed',
        ], key="th_status")
        th_employment = st.selectbox("Employment type", ['PAYE', 'Self-employed'],
                                      key="th_employment")

        status_map = {
            'Single': 'single',
            'Married (one earner)': 'married_one_earner',
            'Married (two earners)': 'married_two_earner',
            'Widowed': 'widowed',
        }
        emp_map = {'PAYE': 'paye', 'Self-employed': 'self_employed'}

    with th_col2:
        st.subheader("Proposed Changes (optional)")
        st.markdown("Adjust parameters to compare current vs proposed take-home.")

        th_band_chg = st.number_input("Standard rate band increase (€)",
                                       0, 10_000, 0, 500, key="th_band_chg")
        th_credit_chg = st.number_input("Personal credit increase (€)",
                                         0, 1000, 0, 50, key="th_credit_chg")
        th_emp_credit_chg = st.number_input("Employee/Earned Income credit increase (€)",
                                             0, 1000, 0, 50, key="th_emp_credit_chg")
        th_usc3_new = st.slider("USC 3% band rate (%)", 0.0, 4.0, 3.0, 0.25,
                                 key="th_usc3")

    st.divider()

    if gross_salary > 0:
        status_val = status_map[th_status]
        emp_val = emp_map[th_employment]

        # Current
        current = calc_take_home(gross_salary, status_val, emp_val)

        # Build proposed params
        has_changes = (th_band_chg > 0 or th_credit_chg > 0 or
                       th_emp_credit_chg > 0 or th_usc3_new != 3.0)

        proposed = None
        if has_changes:
            prop_it = IncomeTaxParams()
            prop_it.single_band += th_band_chg
            prop_it.married_one_earner_band += th_band_chg
            prop_it.married_two_earner_band += th_band_chg
            prop_it.single_person_credit += th_credit_chg
            prop_it.married_credit += th_credit_chg * 2
            prop_it.widowed_credit += th_credit_chg
            if emp_val == 'paye':
                prop_it.employee_credit += th_emp_credit_chg
            else:
                prop_it.earned_income_credit += th_emp_credit_chg

            prop_usc = USCParams()
            prop_usc.band3_rate = th_usc3_new / 100

            proposed = calc_take_home(gross_salary, status_val, emp_val,
                                       prop_it, prop_usc)

        # Display
        if proposed:
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown("### Current (2026)")
            with r2:
                st.markdown("### Proposed")
            with r3:
                st.markdown("### Difference")

            rows = [
                ('Gross Income', 'gross'),
                ('Income Tax', 'income_tax'),
                ('USC', 'usc'),
                ('PRSI', 'prsi'),
                ('Total Deductions', 'total_deductions'),
                ('Net Take-Home', 'net_pay'),
            ]
            compare_rows = []
            for label, key in rows:
                curr_val = current[key]
                prop_val = proposed[key]
                diff = prop_val - curr_val
                compare_rows.append({
                    'Item': label,
                    'Current': f"€{curr_val:,.2f}",
                    'Proposed': f"€{prop_val:,.2f}",
                    'Change': f"€{diff:+,.2f}",
                })
            compare_rows.append({
                'Item': 'Effective Rate',
                'Current': f"{current['effective_rate']}%",
                'Proposed': f"{proposed['effective_rate']}%",
                'Change': f"{proposed['effective_rate'] - current['effective_rate']:+.1f}pp",
            })
            compare_rows.append({
                'Item': 'Marginal Rate',
                'Current': f"{current['marginal_rate']}%",
                'Proposed': f"{proposed['marginal_rate']}%",
                'Change': f"{proposed['marginal_rate'] - current['marginal_rate']:+.1f}pp",
            })
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True,
                         use_container_width=True)

            saving = proposed['net_pay'] - current['net_pay']
            if saving > 0:
                st.success(f"You would save **€{saving:,.2f}/year** "
                           f"(€{saving/12:,.2f}/month) under the proposed changes.")
            elif saving < 0:
                st.warning(f"You would pay **€{abs(saving):,.2f}/year** more "
                           f"under the proposed changes.")
        else:
            # Just show current payslip
            st.subheader("Your 2026 Payslip")

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Net Take-Home", f"€{current['net_pay']:,.0f}")
            with m2:
                st.metric("Monthly Net", f"€{current['net_pay']/12:,.0f}")
            with m3:
                st.metric("Effective Rate", f"{current['effective_rate']}%")
            with m4:
                st.metric("Marginal Rate", f"{current['marginal_rate']}%")

            payslip_rows = [
                {'Item': 'Gross Income', 'Annual': f"€{current['gross']:,.2f}",
                 'Monthly': f"€{current['gross']/12:,.2f}"},
                {'Item': 'Income Tax', 'Annual': f"-€{current['income_tax']:,.2f}",
                 'Monthly': f"-€{current['income_tax']/12:,.2f}"},
                {'Item': 'USC', 'Annual': f"-€{current['usc']:,.2f}",
                 'Monthly': f"-€{current['usc']/12:,.2f}"},
                {'Item': 'PRSI', 'Annual': f"-€{current['prsi']:,.2f}",
                 'Monthly': f"-€{current['prsi']/12:,.2f}"},
                {'Item': 'Total Deductions', 'Annual': f"-€{current['total_deductions']:,.2f}",
                 'Monthly': f"-€{current['total_deductions']/12:,.2f}"},
                {'Item': 'Net Take-Home', 'Annual': f"€{current['net_pay']:,.2f}",
                 'Monthly': f"€{current['net_pay']/12:,.2f}"},
            ]
            st.dataframe(pd.DataFrame(payslip_rows), hide_index=True,
                         use_container_width=True)

        # Effective rate chart across income levels
        st.divider()
        st.subheader("Effective Tax Rate by Income")
        incomes = list(range(10_000, 200_001, 5_000))
        rate_rows = []
        for inc in incomes:
            th = calc_take_home(inc, status_val, emp_val)
            row = {'Income': inc, 'Current': th['effective_rate']}
            if has_changes:
                th_p = calc_take_home(inc, status_val, emp_val, prop_it, prop_usc)
                row['Proposed'] = th_p['effective_rate']
            rate_rows.append(row)
        rate_df = pd.DataFrame(rate_rows).set_index('Income')
        st.line_chart(rate_df)


# ==============================================================
# TAB 3: USC EXPLORER
# ==============================================================

with tab_usc:
    st.header("USC Rate & Band Explorer")

    col_rate, col_band = st.columns(2)

    with col_rate:
        st.subheader("Rate Changes")
        st.markdown("Cost of changing a single USC rate (full year, €m)")

        usc_band_sel = st.selectbox("Select USC band",
                                    ['0.5% band', '2% band', '3% band', '8% band'],
                                    key="usc_rate_band")
        band_map = {'0.5% band': (1, 0.005), '2% band': (2, 0.02),
                    '3% band': (3, 0.03), '8% band': (4, 0.08)}
        band_num, current_rate = band_map[usc_band_sel]

        band_attr = {1: 'band1_rate', 2: 'band2_rate',
                     3: 'band3_rate', 4: 'band4_rate'}

        # Generate a range of new rates
        if band_num == 1:
            test_rates = [0.0, 0.1, 0.2, 0.3, 0.4]
        elif band_num == 2:
            test_rates = [0.0, 0.5, 1.0, 1.5]
        elif band_num == 3:
            test_rates = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        else:
            test_rates = [4.0, 5.0, 6.0, 7.0]

        rows = []
        for r in test_rates:
            baseline = USCParams()
            counter = USCParams()
            setattr(counter, band_attr[band_num], r / 100)
            res = cost_usc_change(baseline, counter)
            direction = "Cost" if res['full_year_cost_€m'] > 0 else "Yield"
            rows.append({
                'New Rate': f"{r:.1f}%",
                'Change': f"{(r/100 - current_rate)*100:+.1f}pp",
                'Full Year (€m)': abs(res['full_year_cost_€m']),
                'Direction': direction,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

    with col_band:
        st.subheader("Band Widening")
        st.markdown("Cost of widening USC rate bands (€m)")

        usc_band_type = st.selectbox(
            "Select band boundary",
            ['0.5% upper', '2% upper only', '3% band', '8% lower threshold'],
            key="usc_band_type"
        )
        band_key_map = {
            '0.5% upper': '0.5%_upper',
            '2% upper only': '2%_upper',
            '3% band': '3%',
            '8% lower threshold': '8%_lower',
        }

        test_amounts = [500, 1000, 1500, 2000, 3000, 5000]
        rows = []
        for amt in test_amounts:
            try:
                res = cost_usc_band_change(band_key_map[usc_band_type], amt)
                rows.append({
                    'Increase': f"+€{amt:,}",
                    'First Year (€m)': res['first_year_€m'],
                    'Full Year (€m)': res['full_year_€m'],
                })
            except Exception:
                pass

        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True,
                         use_container_width=True)


# ==============================================================
# TAB 3: INCOME TAX
# ==============================================================

with tab_it:
    st.header("Income Tax Explorer")

    col_rates, col_bands = st.columns(2)

    with col_rates:
        st.subheader("Rate Changes")
        st.markdown("Cost/yield per 1 percentage point change (€m)")

        rate_data = [
            {'Rate': '20%', 'Cut 1pp (Cost)': '€1,070m', 'Increase 1pp (Yield)': '€1,085m'},
            {'Rate': '40%', 'Cut 1pp (Cost)': '€567m', 'Increase 1pp (Yield)': '€567m'},
        ]
        st.dataframe(pd.DataFrame(rate_data), hide_index=True,
                     use_container_width=True)

        st.divider()
        st.subheader("Custom Rate Change")
        rate_sel = st.selectbox("Rate", ['20%', '40%'], key="it_rate_sel")
        rate_chg = st.slider("Change (pp)", -5.0, 5.0, -1.0, 0.5, key="it_rate_chg")
        if rate_chg != 0:
            r = cost_it_rate_change(rate_sel.replace('%', ''), rate_chg)
            label = "Cost" if r['full_year_€m'] < 0 else "Yield"
            st.metric(f"Full Year {label}", f"€{abs(r['full_year_€m']):,.0f}m")
            st.metric("First Year", f"€{abs(r['first_year_€m']):,.0f}m")

    with col_bands:
        st.subheader("Standard Rate Band Widening")
        band_inc_it = st.slider("Band increase (€)", 0, 5000, 1000, 100,
                                key="it_band_exp")
        if band_inc_it > 0:
            detail = cost_it_band_change_detail(band_inc_it)
            total = cost_it_band_change(band_inc_it)

            detail_rows = [
                {'Category': 'Single / Widowed',
                 'First Year (€m)': detail['single']['first_year_€m'],
                 'Full Year (€m)': detail['single']['full_year_€m']},
                {'Category': 'Married (one earner)',
                 'First Year (€m)': detail['married_one_earner']['first_year_€m'],
                 'Full Year (€m)': detail['married_one_earner']['full_year_€m']},
                {'Category': 'Married (two earners)',
                 'First Year (€m)': detail['married_two_earner']['first_year_€m'],
                 'Full Year (€m)': detail['married_two_earner']['full_year_€m']},
                {'Category': 'TOTAL',
                 'First Year (€m)': total['first_year_€m'],
                 'Full Year (€m)': total['full_year_€m']},
            ]
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True,
                         use_container_width=True)


# ==============================================================
# TAB 4: TAX CREDITS
# ==============================================================

with tab_credits:
    st.header("Tax Credit Explorer")

    st.markdown("Select a credit and amount to see the cost.")

    cc1, cc2 = st.columns(2)

    with cc1:
        credit_sel = st.selectbox("Credit", [
            'single_person', 'married', 'employee', 'earned_income',
            'home_carer', 'age', 'rent',
        ], format_func=lambda x: x.replace('_', ' ').title(),
            key="credit_sel")

        credit_amt = st.slider("Increase (€)", 0, 1000, 100, 25,
                               key="credit_amt")

    with cc2:
        if credit_amt > 0:
            cr_res = cost_credit_change(credit_sel, credit_amt)
            st.metric("First Year Cost", f"€{cr_res['first_year_€m']:,.1f}m")
            st.metric("Full Year Cost", f"€{cr_res['full_year_€m']:,.1f}m")

    st.divider()
    st.subheader("All Credits — Cost per Standard Increment")

    all_credits = [
        ('Single Person', 'single_person', 100),
        ('Married', 'married', 200),
        ('Employee (PAYE)', 'employee', 50),
        ('Earned Income', 'earned_income', 50),
        ('Home Carer', 'home_carer', 50),
        ('Age', 'age', 50),
        ('Rent', 'rent', 100),
    ]
    cr_rows = []
    for label, key, unit in all_credits:
        r = cost_credit_change(key, unit)
        cr_rows.append({
            'Credit': label,
            'Unit Increase': f"€{unit}",
            'First Year (€m)': r['first_year_€m'],
            'Full Year (€m)': r['full_year_€m'],
        })
    st.dataframe(pd.DataFrame(cr_rows), hide_index=True,
                 use_container_width=True)


# ==============================================================
# TAB 5: INDEXATION
# ==============================================================

with tab_index:
    st.header("Indexation Calculator")
    st.markdown("Cost of indexing the tax system (credits, bands, USC limits) "
                "by a given percentage.")

    idx_pct = st.slider("Indexation rate (%)", 0.0, 10.0, 2.0, 0.5,
                        key="idx_pct")

    if idx_pct > 0:
        idx_res = cost_indexation(idx_pct)
        rows = []
        total_fy = 0
        total_full = 0
        for desc, vals in idx_res.items():
            rows.append({
                'Component': desc,
                'First Year (€m)': vals['first_year_€m'],
                'Full Year (€m)': vals['full_year_€m'],
            })
            total_fy += vals['first_year_€m']
            total_full += vals['full_year_€m']
        rows.append({
            'Component': 'TOTAL',
            'First Year (€m)': round(total_fy, 1),
            'Full Year (€m)': round(total_full, 1),
        })
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     use_container_width=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric(f"Total First Year ({idx_pct}%)",
                      f"€{total_fy:,.0f}m")
        with mc2:
            st.metric(f"Total Full Year ({idx_pct}%)",
                      f"€{total_full:,.0f}m")


# ==============================================================
# TAB 6: INCOME DISTRIBUTION
# ==============================================================

with tab_dist:
    st.header("Income Distribution (2026 Projected)")
    st.markdown("Revenue Ready Reckoner projected taxpayer units by gross income range.")

    dist_rows = []
    for lower, upper, units, income_m, tax_m in TAXPAYER_UNITS_2026:
        upper_label = f"€{upper:,}" if upper < 550_000 else "—"
        dist_rows.append({
            'Income Range': f"€{lower:,} – {upper_label}",
            'Taxpayer Units': f"{units:,}",
            'Total Income (€m)': f"{income_m:,}",
            'IT + USC (€m)': f"{tax_m:,}",
            'Avg Income': f"€{int(income_m * 1e6 / units):,}" if units > 0 else "—",
            'Effective Rate': f"{(tax_m / income_m * 100):.1f}%" if income_m > 0 else "—",
        })

    st.dataframe(pd.DataFrame(dist_rows), hide_index=True,
                 use_container_width=True, height=700)

    # Chart: taxpayer units by income range
    chart_data = pd.DataFrame([
        {'Income Range': f"€{r[0]//1000}k–€{r[1]//1000}k"
         if r[1] < 550_000 else f"€{r[0]//1000}k+",
         'Taxpayer Units': r[2]}
        for r in TAXPAYER_UNITS_2026
    ])
    st.bar_chart(chart_data.set_index('Income Range'))
