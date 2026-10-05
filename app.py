"""
Repair, Replace, or Retire? Boston gas pipe triage
Part 4: Streamlit app. Run from the project folder:  streamlit run app.py
Reads the files saved in Parts 2 and 3 (data_processed/).
"""
import json
import math
import os

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="Boston gas pipe triage", page_icon="🔥", layout="wide")

CREATOR = "Cathy Kam"   # app creator, shown in the header, sidebar, and Methods tab
DATA = "data_processed"
SEG_FILE = os.path.join(DATA, "segments_scored.geojson")
SUMMARY_FILE = os.path.join(DATA, "plan_summary.json")
GSEP_FILE = os.path.join(DATA, "gsep_annual.csv")

ACTION_COLORS = {
    "Monitor (lower risk)": [205, 208, 212],
    "Replace or repair pipe": [59, 110, 143],
    "Electrify homes (non-pipe alternative)": [39, 174, 96],
}
TYPE_COLORS = {
    "Older owner-occupied neighborhoods": [230, 126, 34],
    "Dense old rental neighborhoods": [192, 57, 43],
    "Through streets, few homes": [149, 165, 166],
    "Newer low-density streets": [59, 110, 143],
    "Large-building streets": [142, 68, 173],
    "Heat-pump streets": [39, 174, 96],
}
BASEMAP = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading street data...")
def load_data():
    seg = gpd.read_file(SEG_FILE)
    for c in ["likely_leak_prone", "is_ej"]:
        seg[c] = seg[c].astype(bool)
    if "miles" not in seg:
        seg["miles"] = seg["length_m"] / 1609.34
    seg["units_per_100m"] = seg["res_units"] / (seg["length_m"] / 100)
    seg["street"] = (seg["ST_NAME"].fillna("").str.title() + " " + seg["ST_TYPE"].fillna("").str.title()).str.strip()
    if "archetype_name" not in seg:
        seg["archetype_name"] = "Type " + seg["archetype"].astype(int).astype(str) if "archetype" in seg else "Not available"

    def paths(geom):
        if geom is None or geom.is_empty:
            return []
        parts = [geom] if geom.geom_type == "LineString" else list(getattr(geom, "geoms", []))
        return [[[round(x, 6), round(y, 6)] for x, y, *_ in p.coords] for p in parts]

    seg["paths"] = seg.geometry.apply(paths)
    df = pd.DataFrame(seg.drop(columns="geometry"))
    df = df.explode("paths").dropna(subset=["paths"]).reset_index(drop=True)

    summary = json.load(open(SUMMARY_FILE)) if os.path.exists(SUMMARY_FILE) else {}
    gsep = pd.read_csv(GSEP_FILE).sort_values("year") if os.path.exists(GSEP_FILE) else None
    return df, summary, gsep


def hp_lifetime_factor(life_years, horizon=60, discount=0.03):
    """Present value of buying a heat pump now plus replacements over the pipe's horizon."""
    n = math.ceil(horizon / life_years)
    return sum(1 / (1 + discount) ** (k * life_years) for k in range(n))


def triage(df, cost_per_mile, hp_cost, mult, basis, hp_life, budget):
    seg = df.drop_duplicates("seg_id").set_index("seg_id")
    s = seg[seg["likely_leak_prone"]].copy()
    s["replace_capital"] = s["miles"] * cost_per_mile
    s["electrify_capital"] = s["res_units"] * hp_cost
    if basis == "Lifetime cost to customers":
        replace_cmp = s["replace_capital"] * mult
        electrify_cmp = s["electrify_capital"] * hp_lifetime_factor(hp_life)
    else:
        replace_cmp, electrify_cmp = s["replace_capital"], s["electrify_capital"]

    eligible = (s["n_large_bldgs"] == 0) & (s["res_units"] > 0)
    s["electrify"] = eligible & (electrify_cmp < replace_cmp)
    s["action"] = np.where(s["electrify"], "Electrify homes (non-pipe alternative)", "Replace or repair pipe")
    s["compared_cost"] = np.where(s["electrify"], electrify_cmp, replace_cmp)
    s["replace_compared"] = replace_cmp
    s["capital_cost"] = np.where(s["electrify"], s["electrify_capital"], s["replace_capital"])

    s["priority"] = s["risk_score"] * s["miles"] / s["capital_cost"].clip(lower=1)
    s = s.sort_values("priority", ascending=False)
    s["plan_year"] = (s["capital_cost"].cumsum() // budget).astype(int) + 1

    actions = pd.Series("Monitor (lower risk)", index=seg.index)
    actions.loc[s.index] = s["action"]
    years = pd.Series(np.nan, index=seg.index)
    years.loc[s.index] = s["plan_year"]

    per_home_cmp = hp_cost * (hp_lifetime_factor(hp_life) if basis == "Lifetime cost to customers" else 1)
    per_mile_cmp = cost_per_mile * (mult if basis == "Lifetime cost to customers" else 1)
    stats = {
        "flagged_miles": s["miles"].sum(),
        "electrify_miles": s.loc[s["electrify"], "miles"].sum(),
        "electrify_homes": s.loc[s["electrify"], "res_units"].sum(),
        "all_replace": s["replace_compared"].sum(),
        "mixed": s["compared_cost"].sum(),
        "years": int(s["plan_year"].max()) if len(s) else 0,
        "breakeven_per_100m": per_mile_cmp / per_home_cmp / 16.0934,
        "ej_share": (s.loc[s["electrify"], "miles"] * s.loc[s["electrify"], "is_ej"]).sum()
                    / max(s.loc[s["electrify"], "miles"].sum(), 1e-9),
    }
    return actions, years, s, stats


def path_layer(frame, color_col, width_col):
    return pdk.Layer("PathLayer", frame, get_path="paths", get_color=color_col, get_width=width_col,
                     width_units="pixels", width_min_pixels=1, pickable=True, auto_highlight=True)


def boston_view():
    return pdk.ViewState(latitude=42.315, longitude=-71.075, zoom=11.2, pitch=0)


def money(x):
    return f"${x / 1e9:,.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.0f}M"


# ---------------------------------------------------------------- page
if not os.path.exists(SEG_FILE):
    st.error("data_processed/segments_scored.geojson was not found. Run Part 3, Cell 4 first, "
             "then start the app from the project folder.")
    st.stop()

df, summary, gsep = load_data()
default_cpm = summary.get("cost_per_mile_usd", 3.06e6)
default_mult = summary.get("ratepayer_multiplier", 1.0)
mult_source = summary.get("ratepayer_multiplier_source", "not available")
risk_source = summary.get("risk_source", df["risk_source"].iloc[0] if "risk_source" in df else "")

st.title("Repair, replace, or retire?")
st.markdown("Which of Boston's old gas streets should get new pipe, and which would cost less to switch to "
            "electric heat? A street-by-street look built from public data.")
st.caption(f"Created by **{CREATOR}**")

# Program figures for the description, computed from DPU's report (gsep_annual.csv)
if gsep is not None and len(gsep):
    _spent = gsep["spend_musd"].sum() / 1000
    _miles = gsep["main_miles"].sum()
    _cpm0 = gsep["spend_musd"].iloc[0] / gsep["main_miles"].iloc[0]
    _cpm1 = gsep["spend_musd"].iloc[-1] / gsep["main_miles"].iloc[-1]
    _y0, _y1 = int(gsep["year"].iloc[0]), int(gsep["year"].iloc[-1])
    program_line = (f"Between {_y0} and {_y1}, Massachusetts gas utilities spent about **${_spent:.2f} billion** "
                    f"replacing **{_miles:,.0f} miles** of leak-prone gas mains under the Gas System Enhancement "
                    f"Program (GSEP). Over the same period, spending per mile rose from **${_cpm0:.2f}M** to "
                    f"**${_cpm1:.2f}M**, while the number of miles replaced each year stayed roughly flat.")
else:
    program_line = ("Massachusetts gas utilities have spent billions replacing leak-prone gas mains under the "
                    "Gas System Enhancement Program (GSEP), at a cost per mile that has risen steeply.")

with st.expander("About this project", expanded=True):
    st.markdown(f"""
{program_line} In 2025, the Department of Public Utilities cut the program's spending cap and directed utilities to
consider cheaper repairs and non-pipe alternatives, such as electrifying the homes on a street instead of laying new pipe.

**The question.** For each of Boston's roughly 17,800 street segments: how likely is it to sit on old, leak-prone pipe,
and would it cost less to replace that pipe or to switch the homes on that street to electric heat?

**How it was built.**
1. **Data extraction and cleaning:** Boston street segments, the FY2026 property assessment, BERDO building energy
   reports, Census housing and heating data, the state's environmental-justice map, and DPU's own reports, all
   pulled from public sources and joined street by street.
2. **Exploratory analysis:** building age as a proxy for pipe age, gas dependence, equity patterns, large-building
   gas demand, and the GSEP spending and mileage record.
3. **Modeling:** k-means clustering into six street types, a leak-prone risk score calibrated to DPU's reported
   leak-prone share for Boston Gas, and a cost model that compares new pipe with electrification and schedules
   work within an annual budget.
4. **This app:** change the assumptions in the sidebar and see the plan, the map, and the costs update.

**Main finding.** House-by-house electrification beats new pipe only on streets with low housing density.
Most of Boston's oldest streets are dense triple-decker and rental blocks, so for them the realistic alternative
to new pipe is a shared system such as networked geothermal.

**Who it's for.** Regulators, utilities, city planners, and advocates weighing where pipe replacement money
does the most good, and where alternatives deserve a closer look.

*Built with public data only. Not an official DPU or utility analysis.*
""")

with st.sidebar:
    st.header("Assumptions")
    basis = st.radio("Compare costs by", ["Capital cost", "Lifetime cost to customers"],
                     help="Lifetime cost adds the utility's return and financing on new pipe, and heat pump "
                          "replacements over 60 years.")
    hp_cost = st.slider("Heat pump cost per home ($)", 10_000, 45_000, int(summary.get("hp_cost_per_unit_usd", 22_000)),
                        step=1_000, help="Typical whole-home air-source install is about $22,000 (Mass Save program average).")
    budget_m = st.slider("Annual budget for Boston streets ($M)", 25, 300, int(summary.get("annual_budget_usd", 100e6) / 1e6), step=25)
    with st.expander("Advanced"):
        cost_per_mile = st.number_input("Pipe replacement cost per mile ($)", 1_000_000, 8_000_000,
                                        int(default_cpm), step=100_000,
                                        help=f"Default: {summary.get('cost_year', 'latest')} statewide actual from DPU's report.")
        mult = st.number_input("Customer cost per $1 of pipe capital", 1.0, 4.0, float(default_mult), step=0.05,
                               help=f"Default source: {mult_source}")
        hp_life = st.slider("Heat pump lifespan (years)", 10, 30, 18,
                            help="Used only for lifetime cost: replacements over a 60-year pipe life, discounted at 3%.")
    if basis == "Lifetime cost to customers" and default_mult == 1.0:
        st.warning("The DPU customer-cost ratio was not loaded in Part 3, so pipe lifetime cost equals capital cost. "
                   "Set it under Advanced, or rerun Part 3, Cell 4 with the working group minutes saved.")
    st.divider()
    st.caption(f"Created by {CREATOR}")

actions, years, plan, stats = triage(df, cost_per_mile, hp_cost, mult, basis, hp_life, budget_m * 1e6)
df["action"] = df["seg_id"].map(actions)
df["plan_year"] = df["seg_id"].map(years)
df["color"] = df["action"].map(ACTION_COLORS)
df["width"] = np.where(df["action"].str.startswith("Monitor"), 1, 3)
df["plan_year_txt"] = df["plan_year"].map(lambda y: "" if pd.isna(y) else f"Year {int(y)}")

tab_overview, tab_map, tab_types, tab_methods = st.tabs(["Findings", "Triage map", "Street types", "Methods and sources"])

# ---------------- Findings
with tab_overview:
    saved = 1 - stats["mixed"] / stats["all_replace"] if stats["all_replace"] else 0
    c = st.columns(4)
    c[0].metric("Likely leak-prone streets", f"{stats['flagged_miles']:,.0f} mi")
    c[1].metric("Tipping point", f"{stats['breakeven_per_100m']:.1f} homes / 100 m",
                help="Below this density, electrifying every home costs less than new pipe.")
    c[2].metric("Cheaper to electrify", f"{stats['electrify_miles']:,.0f} mi",
                f"{stats['electrify_homes']:,.0f} homes", delta_color="off")
    c[3].metric("Saved vs replacing everything", f"{saved:.0%}")

    st.subheader("What the analysis shows")
    st.markdown(
        f"- With the current assumptions, house-by-house electrification beats new pipe only on streets with fewer than "
        f"**{stats['breakeven_per_100m']:.1f} homes per 100 m**.\n"
        f"- Most of Boston's oldest streets are dense triple-decker and rental blocks, well above that line. "
        f"As a result, house-by-house electrification is generally not the lower-cost option on these streets; "
        f"across the full system, the modeled strategy saves **{saved:.0%}** compared with replacing all flagged pipe.\n"
        f"- **{stats['ej_share']:.0%}** of the streets where electrification wins are in environmental-justice "
        f"neighborhoods, so income-qualified support and renter protections would matter.\n"
        f"- At **${budget_m}M a year**, all flagged streets are addressed in about **{stats['years']} years**.\n"
        f"- For dense streets, the realistic alternative is shared systems such as networked geothermal, "
        f"which this version does not yet model."
    )

    if gsep is not None:
        st.subheader("Massachusetts pipe replacement program (GSEP), statewide")
        g = gsep.copy()
        g["Spending per mile ($M)"] = g["spend_musd"] / g["main_miles"]
        k = st.columns(3)
        k[0].metric(f"Spent {int(g.year.min())}-{int(g.year.max())}", money(g["spend_musd"].sum() * 1e6))
        k[1].metric("Miles of main replaced", f"{g['main_miles'].sum():,.0f}")
        k[2].metric("Spending per mile", f"${g['Spending per mile ($M)'].iloc[-1]:.2f}M",
                    f"from ${g['Spending per mile ($M)'].iloc[0]:.2f}M in {int(g.year.min())}", delta_color="inverse")
        left, right = st.columns(2)
        base = alt.Chart(g).encode(x=alt.X("year:O", title=None))
        left.altair_chart(base.mark_bar(color="#c0392b").encode(
            y=alt.Y("spend_musd:Q", title="Spending ($ millions)"),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("spend_musd:Q", title="$M", format=",.1f")]
        ).properties(title="Money spent per year", height=280), width="stretch")
        right.altair_chart(base.mark_bar(color="#3b6e8f").encode(
            y=alt.Y("main_miles:Q", title="Miles of main"),
            tooltip=[alt.Tooltip("year:O"), alt.Tooltip("main_miles:Q", title="Miles", format=",.1f")]
        ).properties(title="Miles replaced per year", height=280), width="stretch")
        st.caption("Source: DPU Report to the Legislature on Natural Gas Leaks, D.P.U. 25-GLR-01 (Dec 31, 2025). "
                   "Spending includes mains, services, and related work, in nominal dollars.")

# ---------------- Triage map
with tab_map:
    m = st.columns(4)
    m[0].metric("Replace or repair", f"{stats['flagged_miles'] - stats['electrify_miles']:,.0f} mi")
    m[1].metric("Electrify homes", f"{stats['electrify_miles']:,.0f} mi")
    m[2].metric("Replace everything", money(stats["all_replace"]))
    m[3].metric("Mixed plan", money(stats["mixed"]), f"{-saved:.0%}", delta_color="inverse")

    nbhds = ["All neighborhoods"] + sorted(df["NBHD_L"].dropna().unique())
    pick = st.selectbox("Neighborhood", nbhds)
    view_df = df if pick == "All neighborhoods" else df[df["NBHD_L"] == pick]

    st.pydeck_chart(pdk.Deck(
        layers=[path_layer(view_df, "color", "width")], initial_view_state=boston_view(), map_style=BASEMAP,
        tooltip={"html": "<b>{street}</b><br/>{NBHD_L}<br/>{action} {plan_year_txt}<br/>"
                         "Homes on block: {res_units}<br/>Median building year: {med_yr_built}"}),
        width="stretch", height=560)
    legend = " &nbsp; ".join(f"<span style='color:rgb{tuple(c)};font-size:20px'>■</span> {a}"
                             for a, c in ACTION_COLORS.items())
    st.markdown(legend, unsafe_allow_html=True)

    st.subheader("Priority list")
    table = (plan.reset_index()
                 .assign(street=lambda d: d["street"], homes=lambda d: d["res_units"].round(0),
                         cost=lambda d: d["capital_cost"].round(-3))
                 [["plan_year", "street", "NBHD_L", "action", "miles", "homes", "med_yr_built", "cost", "is_ej"]]
                 .rename(columns={"plan_year": "Year", "street": "Street", "NBHD_L": "Neighborhood",
                                  "action": "Action", "miles": "Miles", "homes": "Homes",
                                  "med_yr_built": "Building year", "cost": "Capital cost ($)", "is_ej": "EJ area"}))
    if pick != "All neighborhoods":
        table = table[table["Neighborhood"] == pick]
    st.dataframe(table.round({"Miles": 3}), width="stretch", height=320, hide_index=True)
    st.download_button("Download plan as CSV", table.to_csv(index=False), "boston_gas_triage_plan.csv", "text/csv")

# ---------------- Street types
with tab_types:
    st.markdown("Streets grouped by k-means clustering on building age, housing density, large-building gas use, "
                "gas heat share, renter share, income, and heat pump share.")
    tdf = df.copy()
    tdf["tcolor"] = tdf["archetype_name"].map(TYPE_COLORS).apply(lambda c: c if isinstance(c, list) else [120, 120, 120])
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("PathLayer", tdf, get_path="paths", get_color="tcolor", get_width=2,
                          width_units="pixels", pickable=True)],
        initial_view_state=boston_view(), map_style=BASEMAP,
        tooltip={"html": "<b>{street}</b><br/>{archetype_name}"}), width="stretch", height=500)
    one = df.drop_duplicates("seg_id")
    prof = (one.groupby("archetype_name")
               .agg(Miles=("miles", "sum"), **{"Building year": ("med_yr_built", "median")},
                    **{"Homes per 100 m": ("units_per_100m", "median")},
                    **{"Likely leak-prone": ("likely_leak_prone", "mean")}, **{"EJ share": ("is_ej", "mean")})
               .sort_values("Miles", ascending=False))
    st.dataframe(prof.style.format({"Miles": "{:,.0f}", "Building year": "{:.0f}", "Homes per 100 m": "{:.1f}",
                                    "Likely leak-prone": "{:.0%}", "EJ share": "{:.0%}"}), width="stretch")

# ---------------- Methods and sources
with tab_methods:
    st.markdown(f"""
**Risk score.** {risk_source}. Older streets are more likely to have cast-iron or unprotected steel mains;
the share of street miles flagged matches the leak-prone share of Boston Gas's mains reported by DPU.
When leak records are added, a gradient-boosting model replaces this score and is tested on a held-out year.

**Cost comparison.** For each flagged street: new pipe = miles × cost per mile; electrification = homes × heat pump cost.
Streets with large BERDO-reporting buildings, or no homes, are not considered for electrification.
The schedule ranks streets by risk-weighted miles per dollar and fills each year's budget.

**Data sources**
- DPU, Report to the Legislature on the Prevalence of Natural Gas Leaks, D.P.U. 25-GLR-01 (Dec 31, 2025): GSEP spending,
  miles replaced, leak counts, Boston Gas leak-prone share.
- DPU GSEP Working Group minutes (Oct 20, 2023): customer cost per $1 of GSEP capital. Status: {mult_source}.
- City of Boston open data: street segments (SAM), FY2026 property assessment, BERDO building energy reporting.
- U.S. Census Bureau: ACS 5-year estimates (summary file) and block group boundaries.
- MassGIS: 2020 Environmental Justice populations.

**Limits**
- The risk score is provisional until leak records are added.
- Statewide cost per mile; Boston's urban streets may cost more.
- Removing gas from one street can affect neighboring streets; the network is not modeled.
- Heat pump cost and annual budget are assumptions you can change in the sidebar.

Created by {CREATOR}. Built with public data only. Not an official DPU or utility analysis.
""")
