"""
Repair, Replace, or Retire? Boston gas pipe triage
Part 4: Streamlit app. Run from the project folder:  streamlit run app.py
Reads the files saved in Parts 2 and 3 (data_processed/), plus the network columns added by network_units.py.
"""
import inspect
import json
import math
import os
from collections import defaultdict

import altair as alt
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Boston gas pipe triage", page_icon="🔥", layout="wide")
NO_ARROW = {"delta_arrow": "off"} if "delta_arrow" in inspect.signature(st.metric).parameters else {}

CREATOR = "Cathy Kam"   # app creator, shown in the header, sidebar, and Methods tab
DATA = "data_processed"
SEG_FILE = os.path.join(DATA, "segments_scored.geojson")
SUMMARY_FILE = os.path.join(DATA, "plan_summary.json")
GSEP_FILE = os.path.join(DATA, "gsep_annual.csv")
LIFETIME = "Lifetime cost to customers"

# Options, in the order they're shown
REPLACE = "Replace pipe"
SEAL = "Seal joints, replace later"
ELECTRIFY = "Electrify homes (if every owner agrees)"
GEO = "Networked geothermal (if every owner agrees)"
DEFER = "Repair and monitor (deferred)"
ACTION_COLORS = {REPLACE: "#3B6E8F", SEAL: "#7FA7C9", ELECTRIFY: "#27AE60", GEO: "#8E44AD", DEFER: "#B9C2CA"}

TYPE_COLORS = {
    "Older owner-occupied neighborhoods": [230, 126, 34],
    "Dense old rental neighborhoods": [192, 57, 43],
    "Through streets, few homes": [149, 165, 166],
    "Newer low-density streets": [59, 110, 143],
    "Large-building streets": [142, 68, 173],
    "Heat-pump streets": [39, 174, 96],
}

# ---------------------------------------------------------------- sourced figures (see the Methods tab)
PIPE_LIFE_YEARS = 60
DPU_WG_CAPEX_B, DPU_WG_RATEPAYER_B = 15.9, 34.4            # DPU GSEP Working Group minutes, Oct 20, 2023
BOSTON_GAS_PLANNED_MILES_2025 = 120                         # D.P.U. 25-GLR-01: Boston Gas planned 2025 replacement
BOSTON_GAS_LEAK_PRONE_MILES = 958 + 1351 + 177              # D.P.U. 25-GLR-01: unprotected steel + cast/wrought iron + Aldyl-A
LEAK_REPAIR_BACKLOG_USD, LEAKS_OPEN_END_2024 = 18_536_658, 9_077   # D.P.U. 25-GLR-01
LEAKS_REPORTED_2024, LEAKS_OPEN_END_2023 = 20_564, 10_652         # D.P.U. 25-GLR-01
LEAK_PRONE_MILES_REMAINING_MA = 4_049                       # DPU GSEP orders, April 30, 2025
COST_PER_LEAK = LEAK_REPAIR_BACKLOG_USD / LEAKS_OPEN_END_2024                   # about $2,040
NEW_LEAKS_PER_MILE = (LEAKS_REPORTED_2024 - LEAKS_OPEN_END_2023) / LEAK_PRONE_MILES_REMAINING_MA   # about 2.4
REPAIR_PER_MILE_YEAR = COST_PER_LEAK * NEW_LEAKS_PER_MILE                       # about $5,000
FRAMINGHAM_COST_USD, FRAMINGHAM_CUSTOMERS = 15_000_000, 135                    # Eversource Framingham pilot
GEO_COST_PER_CUSTOMER = FRAMINGHAM_COST_USD / FRAMINGHAM_CUSTOMERS              # about $111,000
SEAL_CREW_SHARE = 0.25            # joint sealing: 9 weeks instead of 9 months for the same main (ULC/Cadent)
LOWELL_COST_USD, LOWELL_CUSTOMERS = 15_600_000, 31                             # National Grid Lowell pilot (canceled)

# Readiness of each option: how proven it is, and how well its cost is known. Review this table as pilots report
# results and update READINESS_REVIEWED.
READINESS_REVIEWED = "October 2026"
READINESS = [
    {"Option": REPLACE, "Readiness": "Widely deployed",
     "Track record": "Standard utility practice; GSEP replaced 2,411 miles statewide from 2015 to 2024",
     "How well the cost is known": "Well documented in DPU reports, but a statewide average that rose from "
                                   "$1.32M to $3.06M per mile from 2015 to 2024"},
    {"Option": SEAL, "Readiness": "Commercially deployed elsewhere",
     "Track record": "Robotic joint sealing used by Con Edison (New York) and Cadent (UK); little public data "
                     "for Massachusetts",
     "How well the cost is known": "A few public figures, about 10% to 25% of replacement"},
    {"Option": ELECTRIFY, "Readiness": "Heat pumps widely deployed; retiring a main street-wide is rare",
     "Track record": "Heat pumps are a mature product (Mass Save); few completed projects have converted a whole "
                     "street and retired its main",
     "How well the cost is known": "Heat pump installs well documented; panel, grid, and main-retirement costs "
                                   "not included"},
    {"Option": GEO, "Readiness": "Pilot stage",
     "Track record": "Framingham pilot connected its first customers in 2024; National Grid canceled its Lowell pilot",
     "How well the cost is known": f"Two public data points: about ${FRAMINGHAM_COST_USD / FRAMINGHAM_CUSTOMERS:,.0f} "
                                   f"and ${LOWELL_COST_USD / LOWELL_CUSTOMERS:,.0f} per customer"},
    {"Option": DEFER, "Readiness": "Standard practice",
     "Track record": "Leak survey and repair are required by state law",
     "How well the cost is known": "Repair costs documented by DPU; how fast leaks grow on old pipe is uncertain"},
]
# Cost uncertainty (plus or minus) for each option: an assumption, adjustable in the sidebar.
DEFAULT_UNCERTAINTY = {REPLACE: 20, SEAL: 40, ELECTRIFY: 30, GEO: 50}
CENTRAL, CAUTIOUS = "Central estimate", "High estimate for every option"


def name_street_types(seg):
    """Give each k-means cluster a plain-language name based on its average traits."""
    if "archetype_name" in seg and not seg["archetype_name"].astype(str).str.startswith("Type").all():
        return seg["archetype_name"]
    if "archetype" not in seg:
        return pd.Series("Not available", index=seg.index)
    get = lambda c: seg[c] if c in seg else pd.Series(0.0, index=seg.index)
    prof = pd.DataFrame({"a": seg["archetype"], "year": get("med_yr_built"), "dens": get("units_per_100m"),
                         "gas": get("large_bldg_gas_therms"), "hp": get("pct_heat_pump")}).groupby("a").mean()

    def name(r):
        if r.hp > 0.3:       return "Heat-pump streets"
        if r.gas > 1000:     return "Large-building streets"
        if r.dens < 1:       return "Through streets, few homes"
        if r.year >= 1960:   return "Newer low-density streets"
        if r.dens >= 10:     return "Dense old rental neighborhoods"
        return "Older owner-occupied neighborhoods"

    return seg["archetype"].map(prof.apply(name, axis=1))


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading street data...")
def load_data(data_version):
    """data_version changes whenever the data files change, so the cache reloads new data automatically."""
    seg = gpd.read_file(SEG_FILE)
    df = pd.DataFrame(seg.drop(columns="geometry")).drop_duplicates("seg_id").reset_index(drop=True)
    df["has_network"] = all(c in df.columns for c in ["branch_id", "branch_isolatable", "project_id"])
    defaults = {"dead_end": False, "branch_id": -1, "possible_loop": False, "edge_of_city": False,
                "branch_isolatable": False, "n_large_bldgs": 0, "n_addresses": 0, "pct_renter": np.nan,
                "CFCC": "", "is_ej": False, "risk_score": 0.5}
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
    for c in ["likely_leak_prone", "is_ej", "dead_end", "possible_loop", "edge_of_city", "branch_isolatable"]:
        df[c] = df[c].fillna(False).astype(bool)
    df["branch_id"] = df["branch_id"].fillna(-1).astype(int)
    if "miles" not in df:
        df["miles"] = df["length_m"] / 1609.34
    df["res_units"] = df["res_units"].fillna(0)
    df["n_large_bldgs"] = df["n_large_bldgs"].fillna(0)
    df["n_addresses"] = df["n_addresses"].fillna(0)
    df["pct_renter"] = df["pct_renter"].fillna(df["pct_renter"].median() if df["pct_renter"].notna().any() else 0.6)
    df["CFCC"] = df["CFCC"].fillna("").astype(str)
    df["units_per_100m"] = df["res_units"] / (df["length_m"] / 100)
    df["street"] = (df["ST_NAME"].fillna("").str.title() + " " + df["ST_TYPE"].fillna("").str.title()).str.strip()
    if "project_id" not in df.columns:          # fallback: same street in the same neighborhood
        key = df.get("st_key", df["street"]).fillna(df["street"]).astype(str) + "|" + df["NBHD_L"].fillna("").astype(str)
        df["project_id"] = pd.factorize(key)[0]
    df["archetype_name"] = name_street_types(df)
    df["network_position"] = np.select(
        [df["branch_id"] < 0, df["edge_of_city"], df["possible_loop"]],
        ["Through street", "Dead end at the city line", "Dead end that may connect (easement or path)"],
        default="Dead-end branch that could come off gas")

    summary = json.load(open(SUMMARY_FILE)) if os.path.exists(SUMMARY_FILE) else {}
    gsep = pd.read_csv(GSEP_FILE).sort_values("year") if os.path.exists(GSEP_FILE) else None
    return df, summary, gsep


@st.cache_data(show_spinner="Loading street map...")
def load_geometry(data_version):
    """Street lines for the map: one row per segment with its longitudes and latitudes (lightly simplified)."""
    seg = gpd.read_file(SEG_FILE, columns=["seg_id"]).drop_duplicates("seg_id")
    if seg.crs is not None and seg.crs.to_epsg() != 4326:
        seg = seg.to_crs(4326)
    seg = seg[seg.geometry.notna() & ~seg.geometry.is_empty]
    geom = seg.geometry.simplify(0.00002)
    lons, lats = [], []
    for g in geom:
        parts = list(g.geoms) if g.geom_type.startswith("Multi") else [g]
        xs, ys = [], []
        for part in parts:
            cx, cy = part.xy
            xs += list(cx) + [None]
            ys += list(cy) + [None]
        lons.append(xs)
        lats.append(ys)
    return pd.DataFrame({"seg_id": seg["seg_id"].values, "lon": lons, "lat": lats}).set_index("seg_id")


def map_traces(rows, geo, name, color, width, hover=True, opacity=1.0):
    """One map line trace for a group of segments, joined with gaps so it draws fast."""
    rows = rows[rows["seg_id"].isin(geo.index)]
    lon, lat, text = [], [], []
    for sid, label in zip(rows["seg_id"], rows["hover"] if hover else [""] * len(rows)):
        xs, ys = geo.at[sid, "lon"], geo.at[sid, "lat"]
        lon += xs
        lat += ys
        text += [label] * len(xs)
    return go.Scattermap(lon=lon, lat=lat, mode="lines", name=f"{name} ({rows['miles'].sum():,.1f} mi)",
                         line=dict(color=color, width=width), opacity=opacity,
                         text=text if hover else None, hoverinfo="text" if hover else "skip",
                         connectgaps=False)


def hp_lifetime_factor(life_years, horizon=PIPE_LIFE_YEARS):
    """Heat pump purchases over the life of new pipe, pro-rated (60 years / 18-year life = 3.33 purchases).
    Undiscounted, to match DPU's lifetime ratio for pipe, which sums customer payments without discounting."""
    return horizon / life_years


def budget_factor(year, glide):
    """DPU's GSEP cap falls from 2.5% to 2% (2026) to 1.5% (2027 on) of revenue; budget scaled in proportion."""
    if not glide:
        return 1.0
    return 1.0 if year == 1 else (2.0 / 2.5 if year == 2 else 1.5 / 2.5)


def schedule(units, budget, crew_cap, glide):
    """Place work units in years, highest priority first, within each year's budget and crew capacity.
    A unit too big for a whole year gets a year to itself."""
    used_cost, used_crew = defaultdict(float), defaultdict(float)
    years = {}
    order = units.sort_values(["immediate", "score"], ascending=[False, False])
    for uid, u in order.iterrows():
        y = int(u["earliest"])
        while True:
            fits = (used_cost[y] + u["cost_now"] <= budget * budget_factor(y, glide) + 1e-6
                    and used_crew[y] + u["crew"] <= crew_cap + 1e-9)
            if fits or (used_cost[y] == 0 and used_crew[y] == 0):
                break
            y += 1
        used_cost[y] += u["cost_now"]
        used_crew[y] += u["crew"]
        years[uid] = y
    return pd.Series(years, dtype=float)


def run_model(df, p):
    """Compare options for every flagged street, decide whole dead-end branches together, and schedule the work."""
    lifetime = p["basis"] == LIFETIME
    H, credit = p["horizon"], p["credit_residual"]
    if lifetime:
        # Costs counted over the analysis horizon. With credit for value left at the end, each asset is counted
        # for the share of its life inside the horizon; without it, every purchase counts in full.
        k_pipe = p["mult"] * (min(H, PIPE_LIFE_YEARS) / PIPE_LIFE_YEARS if credit else 1.0)
        n_hp = H / p["hp_life"] if credit else math.ceil(H / p["hp_life"])
        k_hp = n_hp * (p["mult"] if p["hp_in_rates"] else 1.0)
    else:
        k_pipe = k_hp = 1.0
    cpm, r = p["cost_per_mile"], p["discount"]
    unc = {a: p["unc"][a] / 100 for a in DEFAULT_UNCERTAINTY}
    # Decision rule: central estimates, or every option at the top of its own uncertainty range
    hi = {a: (1 + unc[a]) if p["rule"] == CAUTIOUS else 1.0 for a in unc}

    f = df[df["likely_leak_prone"]].copy()
    total_miles = f["miles"].sum()

    # 1) Risk tiers: the highest-risk share must be fixed first, with pipe work (no time to arrange alternatives)
    f = f.sort_values("risk_score", ascending=False)
    cum = f["miles"].cumsum() / total_miles
    f["immediate"] = (cum - f["miles"] / total_miles) < p["immediate_share"] - 1e-12
    f["low_risk_tier"] = (cum > 1 - p["monitor_share"] + 1e-12) & ~f["immediate"]

    # 2) Pipe options on every flagged segment: replace now, or seal joints now and replace later
    f["replace_capital"] = f["miles"] * cpm
    f["R"] = f["replace_capital"] * k_pipe * hi[REPLACE]
    f["seal_ok"] = p["seal_on"] & f["CFCC"].str.match(r"^A[123]")
    f["S"] = (p["seal_frac"] * f["replace_capital"] * hi[SEAL]
              + f["replace_capital"] * hi[REPLACE] / (1 + r) ** p["seal_life"]) * k_pipe
    use_seal = f["seal_ok"] & (f["S"] < f["R"])
    f["gas_action"] = np.where(use_seal, SEAL, REPLACE)
    f["gas_cost"] = np.where(use_seal, f["S"], f["R"])                       # compared, present value
    f["cost_now"] = np.where(use_seal, p["seal_frac"] * f["replace_capital"], f["replace_capital"])  # spent in plan
    f["crew"] = np.where(use_seal, SEAL_CREW_SHARE * f["miles"], f["miles"])
    # Cost range of the chosen pipe option, for checking how firm each choice is. Sealing has two parts: the
    # sealing itself (sealing range) and the later replacement (replacement range).
    # (Ranges are always around central estimates, whichever decision rule is used.)
    seal_part = p["seal_frac"] * f["replace_capital"] * k_pipe
    later_part = f["replace_capital"] * k_pipe / (1 + r) ** p["seal_life"]
    s_lo = seal_part * (1 - unc[SEAL]) + later_part * (1 - unc[REPLACE])
    s_hi = seal_part * (1 + unc[SEAL]) + later_part * (1 + unc[REPLACE])
    r_c = f["replace_capital"] * k_pipe
    f["gas_lo"] = np.where(use_seal, s_lo, r_c * (1 - unc[REPLACE]))
    f["gas_hi"] = np.where(use_seal, s_hi, r_c * (1 + unc[REPLACE]))
    # Seal vs. replace: both include replacement, so only sealing now vs. the replacement it postpones can swap
    a = p["seal_frac"]                                              # sealing, per $1 of replacement
    b = 1 - 1 / (1 + r) ** p["seal_life"]                           # value of postponing replacement
    seal_flip = (a * (1 - unc[SEAL]) < b * (1 + unc[REPLACE])) and (a * (1 + unc[SEAL]) > b * (1 - unc[REPLACE]))
    f["seal_close"] = f["seal_ok"] & seal_flip

    # Cost alone, segment by segment, ignoring the network and consent (the old view, kept for comparison)
    can_convert = (f["res_units"] > 0) & (f["n_large_bldgs"] == 0)
    stats = {
        "cost_alone_e": f.loc[can_convert & (f["res_units"] * p["hp_cost"] * k_hp * hi[ELECTRIFY] < f["gas_cost"]),
                              "miles"].sum(),
        "cost_alone_g": f.loc[can_convert & (f["res_units"] * p["geo_cost"] * k_pipe * hi[GEO] < f["gas_cost"]),
                              "miles"].sum()
                        if p["geo_on"] else 0.0,
    }

    # 3) Whole dead-end branches: only a branch that could come off gas as a unit can retire its pipe
    f["action"] = f["gas_action"]
    f["branch_P"] = np.nan
    f["branch_owners"] = np.nan
    f["expected_cost"] = f["gas_cost"]
    branches = pd.DataFrame()
    if p["has_network"]:
        allb = df[df["branch_id"] >= 0].copy()
        owners = np.where(allb["n_addresses"] > 0, allb["n_addresses"], (allb["res_units"] > 0).astype(int))
        renter = allb["pct_renter"].clip(0, 1)
        p_eff = p["p_owner"] * (1 - renter) + p["p_landlord"] * renter      # chance one property owner agrees
        allb["owners"] = owners
        allb["logp"] = owners * np.log(np.clip(p_eff, 1e-9, 1.0))
        B = allb.groupby("branch_id").agg(homes=("res_units", "sum"), owners=("owners", "sum"),
                                          logp=("logp", "sum"), large=("n_large_bldgs", "sum"),
                                          iso=("branch_isolatable", "all"), branch_miles=("miles", "sum"))
        fb = f[f["branch_id"] >= 0].groupby("branch_id").agg(
            mf=("miles", "sum"), gas=("gas_cost", "sum"), gas_lo=("gas_lo", "sum"), gas_hi=("gas_hi", "sum"),
            imm=("immediate", "any"))
        B = B.join(fb, how="inner")
        B["P"] = np.exp(B["logp"])                       # chance every property owner on the branch agrees
        B["eligible"] = B["iso"] & (B["large"] == 0) & (B["homes"] > 0) & ~B["imm"]
        B["E"] = B["homes"] * p["hp_cost"] * k_hp * hi[ELECTRIFY]
        B["G"] = B["homes"] * p["geo_cost"] * k_pipe * hi[GEO]
        B["E_exp"] = B["P"] * B["E"] + (1 - B["P"]) * B["gas"] + p["repair_cost"] * B["mf"] * p["lead_e"]
        B["G_exp"] = B["P"] * B["G"] + (1 - B["P"]) * B["gas"] + p["repair_cost"] * B["mf"] * p["lead_g"]
        if not p["geo_on"]:
            B["G_exp"] = np.inf
        B.loc[~B["eligible"], ["E_exp", "G_exp"]] = np.inf
        best = B[["gas", "E_exp", "G_exp"]].idxmin(axis=1)
        B["choice"] = best.map({"gas": "gas", "E_exp": ELECTRIFY, "G_exp": GEO})
        B["chosen_cost"] = B[["gas", "E_exp", "G_exp"]].min(axis=1)
        B["lead"] = np.where(B["choice"] == GEO, p["lead_g"], p["lead_e"])
        # How firm is the choice? Within each option's cost range, could the alternative and pipe work swap places?
        # (Only the uncertain parts move: the alternative's own cost, and the pipe work it would replace.)
        # Compare pipe work with the better of the two alternatives on each branch.
        e_best = B["E_exp"] <= B["G_exp"]
        B["best_alt"] = np.where(e_best, ELECTRIFY, GEO)
        A = np.where(e_best, B["E"] / hi[ELECTRIFY], B["G"] / hi[GEO])     # central estimate
        u_a = np.where(e_best, unc[ELECTRIFY], unc[GEO])
        rep_cost = p["repair_cost"] * B["mf"] * np.where(e_best, p["lead_e"], p["lead_g"])
        alt_chosen = B["choice"] != "gas"
        alt_could_win = B["P"] * A * (1 - u_a) + rep_cost < B["P"] * B["gas_hi"]
        pipe_could_win = B["P"] * A * (1 + u_a) + rep_cost > B["P"] * B["gas_lo"]
        B["close"] = B["eligible"] & np.isfinite(A) & np.where(alt_chosen, pipe_could_win, alt_could_win)
        stats["network_e"] = B.loc[B["eligible"] & (B["E"] < B["gas"]), "mf"].sum()
        stats["network_g"] = B.loc[B["eligible"] & (B["G"] < B["gas"]), "mf"].sum() if p["geo_on"] else 0.0
        branches = B

        alt_b = B[B["choice"] != "gas"]
        in_alt = f["branch_id"].isin(alt_b.index)
        f.loc[in_alt, "action"] = f.loc[in_alt, "branch_id"].map(alt_b["choice"])
        f.loc[f["branch_id"] >= 0, "branch_P"] = f.loc[f["branch_id"] >= 0, "branch_id"].map(B["P"])
        f.loc[f["branch_id"] >= 0, "branch_owners"] = f.loc[f["branch_id"] >= 0, "branch_id"].map(B["owners"])
        share_in_branch = f.loc[in_alt, "miles"] / f.loc[in_alt, "branch_id"].map(alt_b["mf"])
        f.loc[in_alt, "expected_cost"] = share_in_branch * f.loc[in_alt, "branch_id"].map(alt_b["chosen_cost"])
    else:
        stats["network_e"] = stats["network_g"] = np.nan

    # 4) Lowest-risk tier: repair and monitor instead of fixing within the plan (if that setting is on)
    defer = f["low_risk_tier"] & f["action"].isin([REPLACE, SEAL])
    f.loc[defer, "action"] = DEFER

    # 5) Work units: pipe work by street project; alternatives by whole branch (fallback pipe work if refused)
    gas = f[f["action"].isin([REPLACE, SEAL])].copy()
    gas["risk_miles"] = gas["risk_score"] * gas["miles"]
    units_g = gas.groupby(["project_id", "action", "immediate"]).agg(
        cost_now=("cost_now", "sum"), crew=("crew", "sum"),
        risk_miles=("risk_miles", "sum"), compare=("gas_cost", "sum"), miles=("miles", "sum"))
    units_g["immediate"] = [i for _, _, i in units_g.index]       # highest-risk work is its own project
    units_g["earliest"] = 1
    units_g.index = ["P|" + str(a) + "|" + b + "|" + str(i) for a, b, i in units_g.index]
    units = [units_g]
    if len(branches) and (branches["choice"] != "gas").any():
        alt_b = branches[branches["choice"] != "gas"]
        fa = f[f["action"].isin([ELECTRIFY, GEO])].copy()
        fa["risk_miles"] = fa["risk_score"] * fa["miles"]
        agg = fa.groupby("branch_id").agg(cost_now=("cost_now", "sum"), crew=("crew", "sum"),
                                          risk_miles=("risk_miles", "sum"), miles=("miles", "sum"))
        refuse = 1 - alt_b["P"].reindex(agg.index)
        units_b = pd.DataFrame({"cost_now": agg["cost_now"] * refuse, "crew": agg["crew"] * refuse,
                                "immediate": False, "risk_miles": agg["risk_miles"],
                                "compare": alt_b["chosen_cost"].reindex(agg.index), "miles": agg["miles"],
                                "earliest": alt_b["lead"].reindex(agg.index).astype(int) + 1})
        units_b.index = ["B|" + str(i) for i in units_b.index]
        units.append(units_b)
    units = pd.concat(units)
    units["score"] = units["risk_miles"] / units["compare"].clip(lower=1)
    units["year"] = schedule(units, p["budget"], p["crew_miles"], p["glide"]) if len(units) else pd.Series(dtype=float)

    unit_of = np.where(f["action"].isin([REPLACE, SEAL]),
                       "P|" + f["project_id"].astype(str) + "|" + f["action"] + "|" + f["immediate"].astype(str),
                       "B|" + f["branch_id"].astype(str))
    f["unit"] = unit_of
    f["plan_year"] = f["unit"].map(units["year"])
    f.loc[f["action"] == DEFER, "plan_year"] = np.nan

    # 6) Summary numbers
    waiting = f[f["action"].isin([REPLACE, SEAL])]
    interim_wait = (waiting["miles"] * p["repair_cost"] * (waiting["plan_year"] - 1).clip(lower=0)).sum()
    alt_rows = f[f["action"].isin([ELECTRIFY, GEO])]
    if len(branches):
        att = branches[branches["choice"] != "gas"]
        interim_lead = (p["repair_cost"] * att["mf"] * att["lead"]).sum()
        expected_retired = (att["P"] * att["mf"]).sum()
        attempted_owners = att["owners"].sum()
        mean_P = (att["P"] * att["mf"]).sum() / max(att["mf"].sum(), 1e-9)
    else:
        interim_lead = expected_retired = attempted_owners = mean_P = 0.0
    plan_cost = f.loc[f["action"] != DEFER, "expected_cost"].sum() + f.loc[f["action"] == DEFER, "R"].sum()
    by_action = f.groupby("action")["miles"].sum()
    stats.update({
        "flagged_miles": total_miles,
        "breakeven_per_100m": cpm * k_pipe * hi[REPLACE] / (p["hp_cost"] * k_hp * hi[ELECTRIFY]) / 16.0934,
        "miles": {a: by_action.get(a, 0.0) for a in ACTION_COLORS},
        "homes_alt": alt_rows["res_units"].sum(),
        "expected_retired": expected_retired, "attempted_owners": attempted_owners, "mean_P": mean_P,
        "all_replace": f["R"].sum(), "plan_cost": plan_cost,
        "years": int(f["plan_year"].max()) if f["plan_year"].notna().any() else 0,
        "interim_wait": interim_wait, "interim_lead": interim_lead,
        "deferred_repairs_per_year": f.loc[f["action"] == DEFER, "miles"].sum() * p["repair_cost"],
        "immediate_miles": f.loc[f["immediate"], "miles"].sum(),
        "ej_share_alt": (alt_rows["miles"] * alt_rows["is_ej"]).sum() / max(alt_rows["miles"].sum(), 1e-9),
        "seal_eligible_miles": f.loc[f["seal_ok"], "miles"].sum(),
        "seal_close_miles": f.loc[f["seal_close"] & f["action"].isin([REPLACE, SEAL]), "miles"].sum(),
        "alt_close_miles": branches.loc[branches["close"], "mf"].sum() if len(branches) else 0.0,
        "alt_close_branches": int(branches["close"].sum()) if len(branches) else 0,
    })
    return f, units, branches, stats


def tipping_point_visual(breakeven, typical_dense=19, max_homes=20):
    """A row of homes on 100 m of street, with a line at the tipping point (HTML, no dollar signs)."""
    shown = min(breakeven, max_homes)
    cells = []
    for i in range(1, max_homes + 1):
        bg = "#FBE9DA" if i <= shown else "#E3EEF6"
        cells.append(f"<span style='font-size:22px;line-height:1;padding:5px 2px;background:{bg};"
                     f"border-radius:6px'>&#127968;</span>")
        if i == int(shown):
            cells.append("<span style='display:inline-flex;flex-direction:column;align-items:center;margin:0 4px'>"
                         f"<span style='font-size:13px;font-weight:700;color:#17263A'>{breakeven:.1f}</span>"
                         "<span style='width:3px;height:34px;background:#17263A;border-radius:2px'></span></span>")
    beyond = f" (the tipping point is beyond {max_homes} homes)" if breakeven > max_homes else ""
    return f"""
<div style="border:1px solid #D3DCD8;border-radius:12px;padding:18px 22px;background:#FAFBFA;margin:6px 0 18px">
  <div style="font-weight:700;font-size:20px;color:#17263A">The key idea: the tipping point</div>
  <div style="color:#3A4757;margin:4px 0 14px">Picture 100 meters of street, about one city block.
    Each house is one home on it.</div>
  <div style="display:flex;flex-wrap:wrap;align-items:center;gap:3px">{"".join(cells)}</div>
  <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;margin-top:12px;font-weight:600">
    <span style="color:#A3501A">&larr; Fewer homes: electrifying tends to cost less</span>
    <span style="color:#2C6E9E">More homes: new pipe tends to cost less &rarr;</span>
  </div>
  <div style="font-size:14px;color:#5D6B78;margin-top:10px;line-height:1.5">
    The line sits at about <b>{breakeven:.1f} homes per 100 m</b>{beyond}. New pipe costs the same however many
    homes are on the street, while heat pumps are paid for home by home, so the more homes share a street, the more
    new pipe pays off. A typical dense old Boston street has about {typical_dense} homes per 100 m, well past the line.
    This is cost alone: the plan also requires a whole dead-end branch to come off gas, and every property owner on
    it to agree.
  </div>
</div>"""


def money(x):
    if not np.isfinite(x):
        return "n/a"
    return f"${x / 1e9:,.2f}B" if abs(x) >= 1e9 else f"${x / 1e6:,.1f}M"


def md_money(x):
    """money() for Markdown text, where a pair of dollar signs would otherwise switch to math font."""
    return money(x).replace("$", "\\$")


# ---------------------------------------------------------------- page
if not os.path.exists(SEG_FILE):
    st.error("data_processed/segments_scored.geojson was not found. Run Part 3, Cell 4 first, "
             "then start the app from the project folder.")
    st.stop()

data_version = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (SEG_FILE, SUMMARY_FILE, GSEP_FILE))
df, summary, gsep = load_data(data_version)
has_network = bool(df["has_network"].iloc[0])
default_cpm = summary.get("cost_per_mile_usd", 3.06e6)
if summary.get("ratepayer_multiplier", 1.0) > 1.0:
    default_mult = summary["ratepayer_multiplier"]
    mult_source = summary.get("ratepayer_multiplier_source", "DPU GSEP Working Group minutes, Oct 20, 2023")
else:
    default_mult = DPU_WG_RATEPAYER_B / DPU_WG_CAPEX_B
    mult_source = (f"DPU GSEP Working Group minutes, Oct 20, 2023: \\${DPU_WG_CAPEX_B}B of GSEP capital (2022-2039) "
                   f"costs ratepayers \\${DPU_WG_RATEPAYER_B}B")
risk_source = summary.get("risk_source", df["risk_source"].iloc[0] if "risk_source" in df else "")
flagged_total = df.loc[df["likely_leak_prone"], "miles"].sum()
default_crew = max(5, round(BOSTON_GAS_PLANNED_MILES_2025 * flagged_total / BOSTON_GAS_LEAK_PRONE_MILES))
default_budget = int(max(10, round(default_crew * default_cpm / 1e6 / 5) * 5))

st.title("Repair, replace, or retire?")
st.markdown("Which of Boston's old gas streets should get new pipe, which could be sealed, and where could the pipe "
            "be retired instead? A street-by-street look built from public data.")
st.caption(f"Created by **{CREATOR}**")

with st.sidebar:
    st.header("Assumptions")
    basis = st.radio("Compare costs by", ["Capital cost", LIFETIME],
                     help="Lifetime cost adds the utility's return and financing on pipe and geothermal, and heat "
                          "pump replacements, over the analysis horizon (60 years unless changed under Advanced).")
    hp_cost = st.slider("Heat pump cost per home (\\$)", 10_000, 45_000, int(summary.get("hp_cost_per_unit_usd", 22_000)),
                        step=1_000, help="Typical whole-home air-source install is about \\$22,000, including "
                                         "installation labor (published summaries of Mass Save program data).")

    with st.expander("Pipe options"):
        seal_on = st.toggle("Consider joint sealing", value=True,
                            help="Robots seal the joints of large cast-iron mains (12 inches and up) from inside. "
                                 "Only major roads are treated as likely to have mains that large.")
        seal_frac = st.slider("Sealing cost (% of replacement)", 5, 60, 25, step=5, disabled=not seal_on,
                              help="Con Edison example: \\$400,000 instead of \\$1.5-1.8 million (about 25%). "
                                   "Industry reports cite about 10%.") / 100
        seal_life = st.slider("Years sealing adds before replacement", 10, 50, 20, disabled=not seal_on,
                              help="Massachusetts law counts advanced leak repair that adds at least 10 years; "
                                   "the sealant is lab-tested to 50 years.")
        discount = st.slider("Discount rate (%)", 0.0, 7.0, 3.0, step=0.5,
                             help="Values money spent later at less than money spent now. Federal guidance "
                                  "(OMB Circular A-4) uses 2% to 7%. At 0%, sealing never pays off, because "
                                  "the replacement still comes later.") / 100
        repair_cost = st.slider("Leak repair cost per mile per year (\\$)", 1_000, 50_000,
                                int(round(REPAIR_PER_MILE_YEAR, -2)), step=500,
                                help=f"Used while streets wait for their turn, during lead times, and for deferred "
                                     f"streets. Default: about {NEW_LEAKS_PER_MILE:.1f} new leaks per mile of "
                                     f"leak-prone pipe a year × about \\${COST_PER_LEAK:,.0f} per repair (DPU figures). "
                                     f"A low estimate: it leaves out leak surveys and emergency repairs.")
        monitor_share = st.slider("Repair and monitor the lowest-risk share instead (%)", 0, 50, 0, step=5,
                                  help="These streets keep their old pipe, with leaks repaired as found, instead "
                                       "of being fixed within the plan.") / 100

    with st.expander("Alternatives to new pipe"):
        geo_on = st.toggle("Consider networked geothermal", value=True)
        geo_cost = st.slider("Geothermal cost per customer (\\$)", 30_000, 250_000,
                             int(round(GEO_COST_PER_CUSTOMER, -3)), step=5_000, disabled=not geo_on,
                             help="Framingham pilot: about \\$15 million for about 135 customers, including the "
                                  "heat pumps in each building. A second phase is expected to cost less.")
        p_owner = st.slider("Chance an owner-occupant agrees (%)", 30, 100, 80, step=5,
                            help="About 80% of homeowners along the Framingham pilot route joined, with the "
                                 "conversion paid for and a \\$10 monthly charge, so this is optimistic.") / 100
        p_landlord = st.slider("Chance a landlord agrees (%)", 10, 100, 60, step=5,
                               help="An assumption. Landlords decide for their tenants, and tenants usually pay "
                                    "the heating bills, so landlords gain less from converting.") / 100
        lead_e = st.slider("Years to arrange electrification", 1, 8, 3,
                           help="An assumption: outreach, agreements, installs, and retiring the main.")
        lead_g = st.slider("Years to arrange geothermal", 2, 10, 4,
                           help="Framingham: approved by DPU in October 2020, first customers connected in 2024.")

    with st.expander("Uncertainty and readiness"):
        rule = st.radio("Compare options at", [CENTRAL, CAUTIOUS],
                        help="Central estimate uses each option's most likely cost. High estimate puts every option "
                             "at the top of its own uncertainty range, so options whose costs are less well known "
                             "are held to a higher bar. The same rule applies to every option.")
        st.caption("Cost uncertainty, plus or minus. These are assumptions, set wider where there is less "
                   "real-world cost data. See the readiness table on the Findings tab.")
        unc_vals = {}
        for a, label in [(REPLACE, "Replace pipe"), (SEAL, "Seal joints"), (ELECTRIFY, "Electrify homes"),
                         (GEO, "Networked geothermal")]:
            unc_vals[a] = st.slider(f"{label} (±%)", 0, 100, DEFAULT_UNCERTAINTY[a], step=5,
                                    key=f"unc_{a}")

    with st.expander("Schedule"):
        immediate_share = st.slider("Highest-risk share fixed first, with pipe work (%)", 0, 30, 10, step=5,
                                    help="These streets go to the front of the schedule and aren't held up "
                                         "for alternatives that take years to arrange.") / 100
        budget_m = st.slider("Annual budget for Boston streets (\\$M)", 10, 300, default_budget, step=5,
                             help=f"Default: crew capacity × cost per mile (about \\${default_budget}M).")
        crew_miles = st.slider("Crew capacity (miles of new pipe a year)", 5, 60, int(default_crew),
                               help=f"Default: Boston Gas planned {BOSTON_GAS_PLANNED_MILES_2025} miles in 2025 "
                                    f"across its territory; Boston's flagged streets are about "
                                    f"{flagged_total / BOSTON_GAS_LEAK_PRONE_MILES:.0%} of its leak-prone main. "
                                    f"Sealing a mile uses about a quarter of the crew time.")
        glide = st.toggle("Budget falls with DPU's spending cap", value=True,
                          help="DPU lowered the GSEP cap from 2.5% to 2% (2026) and 1.5% (2027 on) of revenue. "
                               "Year 2 gets 80% of year 1's budget, and later years 60%.")

    with st.expander("Advanced"):
        cost_per_mile = st.number_input("All-in replacement cost per mile of main (\\$)", 1_000_000, 8_000_000,
                                        int(default_cpm), step=100_000,
                                        help=f"Total GSEP spending divided by miles of main replaced, statewide, "
                                             f"{summary.get('cost_year', 'latest')} actual from DPU's report. "
                                             f"Includes service lines, excavation, and paving restoration.")
        if basis != LIFETIME:
            st.caption("The two settings below apply only when **Compare costs by** is set to "
                       "**Lifetime cost to customers**.")
        mult = st.number_input("Lifetime customer cost per \\$1 of pipe capital", 1.0, 4.0, round(float(default_mult), 2),
                               step=0.05, disabled=(basis != LIFETIME),
                               help=f"How much gas customers pay over the life of new pipe for every \\$1 the utility "
                                    f"spends building it, including the utility's allowed return, financing, and "
                                    f"depreciation. Default \\${default_mult:.2f}. Source: {mult_source}.")
        hp_life = st.slider("Heat pump lifespan (years)", 10, 30, 18, disabled=(basis != LIFETIME),
                            help="The number of heat pumps bought over a 60-year pipe life (60 ÷ lifespan), counted "
                                 "without discounting so it matches how DPU's pipe ratio is calculated.")
        horizon = st.slider("Analysis horizon (years)", 10, 60, PIPE_LIFE_YEARS, step=5,
                            disabled=(basis != LIFETIME),
                            help="How many years of costs are counted. Short horizons can miss the later costs "
                                 "and benefits of long-lived assets.")
        credit_residual = st.toggle("Credit value left at the end of the horizon", value=True,
                                    disabled=(basis != LIFETIME),
                                    help="On: each asset is counted for the share of its life inside the horizon "
                                         "(pipe and geothermal last about 60 years, heat pumps as set above). "
                                         "Off: every purchase counts in full, even if most of its life falls "
                                         "after the horizon.")
        hp_in_rates = st.toggle("Pay for heat pumps through rates, like pipe", value=False,
                                disabled=(basis != LIFETIME),
                                help="Utilities earn a return on pipe and geothermal, which customers pay over time. "
                                     "Heat pumps are usually paid upfront by owners and rebates. On: heat pumps get "
                                     "the same lifetime multiplier as pipe, so all options are financed the same way.")
    st.divider()
    st.caption(f"Created by {CREATOR}")

params = dict(basis=basis, hp_cost=hp_cost, seal_on=seal_on, seal_frac=seal_frac, seal_life=seal_life,
              discount=discount, repair_cost=repair_cost, monitor_share=monitor_share, geo_on=geo_on,
              geo_cost=geo_cost, p_owner=p_owner, p_landlord=p_landlord, lead_e=lead_e, lead_g=lead_g,
              immediate_share=immediate_share, budget=budget_m * 1e6, crew_miles=crew_miles, glide=glide,
              cost_per_mile=cost_per_mile, mult=mult, hp_life=hp_life, has_network=has_network,
              rule=rule, unc=unc_vals, horizon=horizon, credit_residual=credit_residual, hp_in_rates=hp_in_rates)
plan, units, branches, stats = run_model(df, params)
basis_short = "lifetime" if basis == LIFETIME else "capital"
action_label = f"Lower-cost option ({basis_short} cost)"
saved = 1 - stats["plan_cost"] / stats["all_replace"] if stats["all_replace"] else 0

if not has_network:
    st.warning("Network data not found, so only pipe options are compared. Run network_units.py from your home "
               "folder, copy data_processed/segments_scored.geojson into the project, and redeploy to compare "
               "electrification and geothermal on whole dead-end branches.")

st.info("**Disclaimer:** this is a simplified, independent analysis built on public data, not an official DPU "
        "or utility analysis. Results show where the balance lies under the stated assumptions, not a final answer.")
with st.expander("How the comparison works, and what it still leaves out"):
    st.markdown(f"""
Every flagged street needs the same outcome: **the leak-prone pipe stops being a risk, and the homes still get heat.**
The model compares five ways to get there.

| Option | What happens | Where it's allowed |
|---|---|---|
| **Replace pipe** | New main and service lines | Any flagged street |
| **Seal joints, replace later** | Robots seal the joints of large cast-iron mains from inside; replacement is pushed back {seal_life} years | Major roads, where mains are likely 12 inches or larger |
| **Electrify homes** | Every home switches to heat pumps; the old main is retired | Whole dead-end branches only, and only if every property owner agrees |
| **Networked geothermal** | A shared ground loop heats every building; the old main is retired | Same as electrification |
| **Repair and monitor** | Old pipe stays; leaks are repaired as found | Bridges the wait for every option; can replace fixing the lowest-risk streets (off by default) |

**How the real-world constraints are handled**

- **The gas network is connected.** A main can only be retired if no customer further along still uses gas. So the
  decision is made for a **whole dead-end branch** (a cul-de-sac, or a tree of streets that joins the network at
  one point), never one segment, which also removes odd results from short street fragments. Branches whose dead
  end comes within 30 m of another street (a possible easement or path) or sits at the city line (where the street
  may continue into a neighboring town) are treated as connected. Through streets can only get pipe work.
- **Consent and the obligation to serve.** Gas utilities generally must keep serving customers who want gas, so a branch comes
  off gas only if **every property owner agrees**. The chance of that is computed from the number of owners and the
  renter share; landlords decide for tenants. If anyone declines, the pipe is replaced after all, and that cost is
  included.
- **Timing.** Alternatives take {lead_e} years (electrification) or {lead_g} years (geothermal) to arrange, with leaks
  repaired in the meantime. The highest-risk {immediate_share:.0%} of flagged miles goes first, with pipe work only.
  Hazardous (Grade 1) leaks are repaired as soon as possible under state law, whatever the plan says.
- **Readiness and uncertainty.** Each option has a cost range, set wider where there is less real-world data (see
  the readiness table). The same rule applies to every option: compare central estimates, or put every option at the
  top of its own range. The app shows where choices are close calls, which is where pilots would add the most.
- **Planning horizon and financing.** Under lifetime cost, the horizon sets how many years of costs are counted, and
  long-lived assets can be credited for life left at the end. Heat pumps can be financed through rates like pipe, so
  that differences in how each option is paid for don't decide the comparison.
- **A schedule, not just a ranking.** Pipe work is bundled into street projects (connected segments of the same
  street in one neighborhood) and placed year by year within both the budget and crew capacity, with the budget
  falling in line with DPU's cap if that setting is on. Leaks on streets waiting their turn are repaired and costed.

**What it still leaves out**

| | Pipe options | Alternatives |
|---|---|---|
| **Costs left out** | Future leak repairs on new pipe | Electrical panel and grid upgrades, weatherization, retiring the old main |
| **Lifespan** | New pipe lasts decades | Heat pumps need replacing every 15 to 20 years (counted only under lifetime cost) |
| **Monthly bills** | Not counted | Not counted |
| **Who pays** | Gas customers through rates, with utility return | Homeowners and rebates; geothermal through rates |

Also not modeled: actual pipe material, diameter, and age (estimated from building age and road type); the utility's
real network maps (dead ends come from the street map); paving schedules, which aren't in the city's open data;
lining small-diameter mains, for lack of public cost data; and regulatory approvals. Because items are left out on
every side, the net direction of these omissions is uncertain.
""")

tab_overview, tab_map, tab_types, tab_methods = st.tabs(["Findings", "Triage plan", "Street types", "Methods and sources"])

# ---------------- Findings
with tab_overview:
    m = stats["miles"]
    pipe_miles = m[REPLACE] + m[SEAL]
    c = st.columns(4)
    c[0].metric("Likely leak-prone streets", f"{stats['flagged_miles']:,.0f} mi")
    c[1].metric("Tipping point (homes per 100 m)", f"{stats['breakeven_per_100m']:.1f}",
                help="Cost alone: below this density, electrifying every home costs less than new pipe.")
    c[2].metric("Pipe expected to be retired", f"{stats['expected_retired']:,.1f} mi",
                f"attempted on {m[ELECTRIFY] + m[GEO]:,.1f} mi", delta_color="off", **NO_ARROW,
                help="Miles where an alternative is attempted, times the chance every property owner agrees.")
    c[3].metric("Years to finish", f"{stats['years']}",
                help="Years until every flagged street (except any deferred) is addressed, within the budget "
                     "and crew capacity.")

    st.markdown(tipping_point_visual(stats["breakeven_per_100m"]), unsafe_allow_html=True)

    st.subheader("From cost alone to what's realistic")
    if len(branches):
        att = branches[branches["choice"] != "gas"]
        exp_e = (att.loc[att["choice"] == ELECTRIFY, "P"] * att.loc[att["choice"] == ELECTRIFY, "mf"]).sum()
        exp_g = (att.loc[att["choice"] == GEO, "P"] * att.loc[att["choice"] == GEO, "mf"]).sum()
    else:
        exp_e = exp_g = np.nan
    funnel = pd.DataFrame({
        "Step": ["1. Cheaper on cost alone, street by street",
                 "2. ...and the whole dead-end branch could come off gas",
                 "3. ...chosen after allowing for consent and lead time",
                 "4. ...expected to be retired (times the chance every owner agrees)"],
        "Electrify homes (miles)": [stats["cost_alone_e"], stats["network_e"],
                                    m[ELECTRIFY] if has_network else np.nan, exp_e],
        "Networked geothermal (miles)": [stats["cost_alone_g"], stats["network_g"],
                                         m[GEO] if has_network else np.nan, exp_g],
    })
    st.dataframe(funnel.style.format({"Electrify homes (miles)": "{:,.1f}", "Networked geothermal (miles)": "{:,.1f}"},
                                     na_rep="n/a"), hide_index=True, width="stretch")
    if not has_network:
        st.caption("Steps 2 to 4 need the network data (network_units.py).")

    st.subheader("How firm are these choices?")
    rule_note = ("every option at its high estimate" if rule == CAUTIOUS else "central estimates")
    f1 = st.columns(3)
    f1[0].metric("Seal vs. replace: close calls", f"{stats['seal_close_miles']:,.1f} mi",
                 help="Major-road miles where sealing and replacement could swap places within their cost ranges.")
    f1[1].metric("Pipe vs. alternative: close calls", f"{stats['alt_close_miles']:,.1f} mi",
                 f"{stats['alt_close_branches']} dead-end branches", delta_color="off", **NO_ARROW,
                 help="Dead-end branches that could come off gas, where pipe work and electrification or "
                      "geothermal could swap places within their cost ranges.")
    f1[2].metric("Firm choices", f"{stats['flagged_miles'] - stats['seal_close_miles'] - stats['alt_close_miles']:,.0f} mi",
                 help="Flagged miles where the choice holds anywhere within the cost ranges set in the sidebar.")
    st.caption(f"Choices use {rule_note}. Cost ranges: replace ±{unc_vals[REPLACE]}%, seal ±{unc_vals[SEAL]}%, "
               f"electrify ±{unc_vals[ELECTRIFY]}%, geothermal ±{unc_vals[GEO]}%, set under Uncertainty and "
               f"readiness. Close calls are where better cost data, for example from pilots, would most change the plan. "
               f"The Triage plan tab lists them as pilot candidates.")
    with st.expander(f"How proven is each option? (reviewed {READINESS_REVIEWED})"):
        st.table(pd.DataFrame(READINESS).set_index("Option"))
        st.caption("Readiness changes as pilots report results, so this table should be reviewed regularly. Options "
                   "that are less proven get wider cost ranges by default, and the same rules apply to all of them.")

    st.subheader("What the analysis shows")
    geo_line = (f"- **Networked geothermal** costs about \\${geo_cost:,.0f} per customer at pilot prices, so it's "
                f"cheaper than new pipe on only {stats['cost_alone_g']:,.1f} miles even on cost alone"
                + ((f", and it's chosen on {m[GEO]:,.1f} miles.\n" if m[GEO] > 0 else ", and it isn't chosen anywhere.\n")
                   if has_network else ".\n")
                if geo_on else "- Networked geothermal is switched off in the sidebar.\n")
    defer_line = (f"- The lowest-risk {monitor_share:.0%} of flagged miles ({m[DEFER]:,.0f} miles) is repaired and "
                  f"monitored instead, at about {md_money(stats['deferred_repairs_per_year'])} a year in leak repairs.\n"
                  if monitor_share > 0 else "")
    st.markdown(
        f"- **Pipe work is the lower-cost option on about {pipe_miles / stats['flagged_miles']:.1%} of flagged "
        f"miles**: replacement on {m[REPLACE]:,.0f} miles and joint sealing on {m[SEAL]:,.0f} miles of major "
        f"roads, where large cast-iron mains are likely.\n"
        f"- **Electrification** is cheaper on cost alone on {stats['cost_alone_e']:,.1f} miles, but a main can only be "
        f"retired if no customer further along still uses gas, and only if every property owner agrees. "
        + (f"After those checks, it's attempted on {m[ELECTRIFY]:,.1f} miles, and the pipe is expected to come out "
           f"on about {stats['expected_retired']:,.1f} miles (the average chance that every owner on a chosen "
           f"branch agrees is {stats['mean_P']:.0%}).\n" if has_network else "Run the network step to apply those checks.\n")
        + geo_line + defer_line +
        f"- **Schedule:** the highest-risk {immediate_share:.0%} of flagged miles goes first. With "
        f"\\${budget_m}M a year" + (" (falling with DPU's cap)" if glide else "") + f" and crews for {crew_miles} "
        f"miles of new pipe a year, every flagged street is addressed in about **{stats['years']} years**. Leaks on "
        f"streets waiting their turn cost about {md_money(stats['interim_wait'])} to repair along the way.\n"
        f"- On cost alone, this plan costs {saved:.1%} less than replacing every flagged street "
        f"({md_money(stats['plan_cost'])} vs {md_money(stats['all_replace'])}, {basis_short} cost, present value).\n"
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

# ---------------- Triage plan
with tab_map:
    m = stats["miles"]
    t = st.columns(5)
    for col, a in zip(t, ACTION_COLORS):
        short = {REPLACE: "Replace", SEAL: "Seal joints", ELECTRIFY: "Electrify (attempt)",
                 GEO: "Geothermal (attempt)", DEFER: "Repair and monitor"}[a]
        col.metric(short, f"{m[a]:,.1f} mi")

    nbhds = ["All neighborhoods"] + sorted(df["NBHD_L"].dropna().unique())
    pick = st.selectbox("Neighborhood", nbhds)
    view = plan if pick == "All neighborhoods" else plan[plan["NBHD_L"] == pick]

    st.subheader("Map")
    color_by = st.radio("Color streets by", ["Lower-cost option", "Network position", "Plan year",
                                             "Pilot candidates"], horizontal=True, key="map_color")
    geo = load_geometry(data_version)
    mv = view.copy()
    yr = mv["plan_year"].map(lambda y: "deferred" if pd.isna(y) else f"year {int(y)}")
    mv["hover"] = (mv["street"].fillna("") + ", " + mv["NBHD_L"].fillna("").str.title() + "<br>"
                   + mv["action"] + " · " + yr + "<br>" + mv["network_position"]
                   + "<br>" + mv["res_units"].fillna(0).round(0).astype(int).astype(str) + " homes")
    traces = []
    context = df[~df["likely_leak_prone"]]
    if pick != "All neighborhoods":
        context = context[context["NBHD_L"] == pick]
    traces.append(map_traces(context.assign(hover=""), geo, "Not flagged", "#C9CFD4", 1, hover=False,
                             opacity=0.7))
    if color_by == "Lower-cost option":
        short = {REPLACE: "Replace pipe", SEAL: "Seal joints", ELECTRIFY: "Electrify (attempt)",
                 GEO: "Geothermal (attempt)", DEFER: "Repair and monitor"}
        groups = [(short[a], mv[mv["action"] == a], ACTION_COLORS[a], 4 if a in (ELECTRIFY, GEO) else 2.5)
                  for a in ACTION_COLORS]
    elif color_by == "Network position":
        pos_colors = {"Through street": ("Through street", "#3B6E8F"),
                      "Dead-end branch that could come off gas": ("Dead end, could come off gas", "#27AE60"),
                      "Dead end that may connect (easement or path)": ("Dead end, may connect", "#E67E22"),
                      "Dead end at the city line": ("Dead end at city line", "#8E44AD")}
        groups = [(n, mv[mv["network_position"] == k], c, 2.5 if k == "Through street" else 4)
                  for k, (n, c) in pos_colors.items()]
    elif color_by == "Plan year":
        bins = [("Years 1-2", 1, 2, "#7B1F1F"), ("Years 3-5", 3, 5, "#C0392B"), ("Years 6-10", 6, 10, "#E67E22"),
                ("Years 11-20", 11, 20, "#F1C40F"), ("Year 21 on", 21, 999, "#7FA7C9")]
        groups = [(n, mv[mv["plan_year"].between(a, b)], c, 2.5) for n, a, b, c in bins]
        groups.append(("Deferred", mv[mv["plan_year"].isna()], "#B9C2CA", 2.5))
    else:
        close_ids = branches.index[branches["close"]] if len(branches) else []
        is_close = mv["branch_id"].isin(close_ids)
        groups = [("Other flagged streets", mv[~is_close], "#9AA7B1", 2),
                  ("Pilot candidates", mv[is_close], "#D35400", 5)]
    for name, rows, color, width in groups:
        if len(rows):
            traces.append(map_traces(rows, geo, name, color, width))
    if len(mv):
        pts = geo.loc[geo.index.intersection(mv["seg_id"])]
        all_lon = [x for xs in pts["lon"] for x in xs if x is not None]
        all_lat = [y for ys in pts["lat"] for y in ys if y is not None]
        center = dict(lon=float(np.mean(all_lon)), lat=float(np.mean(all_lat)))
        zoom = 10.6 if pick == "All neighborhoods" else 12.6
    else:
        center, zoom = dict(lon=-71.07, lat=42.32), 10.6
    fig = go.Figure(traces)
    # The base style is built in, so the streets always draw; OpenStreetMap tiles (no key needed) load behind
    # them, faded so the colored streets stand out, and the streets still show if the tiles can't load.
    fig.update_layout(map=dict(style="white-bg", center=center, zoom=zoom, layers=[dict(
                          below="traces", sourcetype="raster", opacity=0.55,
                          sourceattribution="© OpenStreetMap contributors",
                          source=["https://tile.openstreetmap.org/{z}/{x}/{y}.png"])]),
                      height=600, margin=dict(l=0, r=0, t=70, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0))
    st.plotly_chart(fig, key="triage_map", config={"scrollZoom": True, "displaylogo": False})
    st.caption("Drag to pan, scroll or pinch to zoom, and hover over a street for details. Click a legend entry to "
               "hide or show it. Grey streets aren't flagged as likely leak-prone. Pick a neighborhood above to "
               "zoom in. Map data © OpenStreetMap contributors.")

    by_nbhd = view.groupby(["NBHD_L", "action"], as_index=False)["miles"].sum()
    if len(by_nbhd):
        present = [a for a in ACTION_COLORS if a in set(by_nbhd["action"])]
        order = by_nbhd.groupby("NBHD_L")["miles"].sum().sort_values(ascending=False).index.tolist()
        st.altair_chart(alt.Chart(by_nbhd).mark_bar().encode(
            y=alt.Y("NBHD_L:N", sort=order, title=None),
            x=alt.X("miles:Q", title="Flagged street miles"),
            color=alt.Color("action:N", title=None, sort=present,
                            scale=alt.Scale(domain=present, range=[ACTION_COLORS[a] for a in present]),
                            legend=alt.Legend(orient="bottom", columns=3, labelLimit=320)),
            tooltip=[alt.Tooltip("NBHD_L:N", title="Neighborhood"), alt.Tooltip("action:N", title=action_label),
                     alt.Tooltip("miles:Q", title="Miles", format=",.2f")]
        ).properties(title="Lower-cost option on flagged streets, by neighborhood",
                     height=alt.Step(26)), width="stretch")
    else:
        st.info("No flagged streets in this neighborhood.")

    st.subheader("Overview of flagged streets")
    if len(view):
        overview = (view.groupby(["action", "network_position"])
                        .agg(Segments=("miles", "size"), Miles=("miles", "sum"), Homes=("res_units", "sum"),
                             **{"Expected cost ($M)": ("expected_cost", lambda x: x.sum() / 1e6)})
                        .reset_index()
                        .rename(columns={"action": action_label, "network_position": "Network position"}))
        overview[action_label] = pd.Categorical(overview[action_label], categories=list(ACTION_COLORS), ordered=True)
        overview = overview.sort_values([action_label, "Network position"])
        total = pd.DataFrame([{action_label: "All flagged streets", "Network position": "",
                               "Segments": overview["Segments"].sum(), "Miles": overview["Miles"].sum(),
                               "Homes": overview["Homes"].sum(),
                               "Expected cost ($M)": overview["Expected cost ($M)"].sum()}])
        overview = pd.concat([overview.astype({action_label: str}), total], ignore_index=True)
        st.dataframe(overview.style.format({"Segments": "{:,.0f}", "Miles": "{:,.1f}", "Homes": "{:,.0f}",
                                            "Expected cost ($M)": "{:,.1f}"}),
                     hide_index=True, width="stretch")
        st.caption("Expected cost is the present value of the chosen option. For alternatives, it includes the chance "
                   "that someone declines and the pipe is replaced after all, plus leak repairs during the lead time. "
                   "Deferred streets are shown at their replacement cost.")

    st.subheader("Year by year")
    sched = plan[plan["plan_year"].notna()].copy()
    if len(view) and len(sched):
        sched_v = sched if pick == "All neighborhoods" else sched[sched["NBHD_L"] == pick]
        per_year = sched_v.groupby(["plan_year", "action"], as_index=False)["miles"].sum()
        present = [a for a in ACTION_COLORS if a in set(per_year["action"])]
        st.altair_chart(alt.Chart(per_year).mark_bar().encode(
            x=alt.X("plan_year:O", title="Plan year"),
            y=alt.Y("miles:Q", title="Miles"),
            color=alt.Color("action:N", title=None, sort=present,
                            scale=alt.Scale(domain=present, range=[ACTION_COLORS[a] for a in present]),
                            legend=alt.Legend(orient="bottom", columns=3, labelLimit=320)),
            tooltip=[alt.Tooltip("plan_year:O", title="Year"), alt.Tooltip("action:N", title=action_label),
                     alt.Tooltip("miles:Q", title="Miles", format=",.2f")]
        ).properties(title="Miles addressed each year", height=260), width="stretch")
        st.caption("Alternatives are placed in the year their lead time ends; their pipe work happens only if "
                   "someone declines.")

    st.subheader("Pilot candidates")
    if len(branches) and branches["close"].any():
        cand = branches[branches["close"]].copy()
        names = plan[plan["branch_id"].isin(cand.index)].groupby("branch_id").agg(
            Streets=("street", lambda x: ", ".join(sorted(set(x)))), Neighborhood=("NBHD_L", "first"),
            ej=("is_ej", "max"))
        cand = cand.join(names)
        cand["Lower-cost choice"] = cand["choice"].map({"gas": "Pipe work", ELECTRIFY: "Electrify", GEO: "Geothermal"})
        cand["Could swap with"] = np.where(cand["choice"] != "gas", "Pipe work",
                                           cand["best_alt"].map({ELECTRIFY: "Electrify", GEO: "Geothermal"}))
        cand_tbl = pd.DataFrame({
            "Streets": cand["Streets"], "Neighborhood": cand["Neighborhood"].str.title(),
            "Homes": cand["homes"].round(0), "Owners": cand["owners"],
            "Chance all agree (%)": (cand["P"] * 100).round(0),
            "Pipe work ($)": cand["gas"].round(-3), "Electrify ($)": cand["E"].round(-3),
            "Geothermal ($)": cand["G"].round(-3) if geo_on else np.nan,
            "Lower-cost choice": cand["Lower-cost choice"], "Could swap with": cand["Could swap with"],
            "EJ area": np.where(cand["ej"].astype(bool), "Yes", "No")}).sort_values(["Chance all agree (%)", "Homes"], ascending=[False, True])
        st.dataframe(cand_tbl, hide_index=True, width="stretch", height=260)
        st.caption(f"{len(cand_tbl)} dead-end branches where the choice could go either way within the cost ranges. "
                   "A pilot here, run jointly by the utility and technology providers, with results shared openly, "
                   "would narrow the cost ranges and test consent. Costs are present value before the chance of "
                   "consent and lead-time repairs.")
    else:
        st.caption("No dead-end branch is a close call at the current settings.")

    st.subheader("Priority list")
    table = view.assign(
        homes=view["res_units"].round(0),
        owners=view["branch_owners"],
        chance=(view["branch_P"] * 100).round(0),
        first=np.where(view["immediate"], "Yes", ""),
        cost=view["expected_cost"].round(-3),
    )[["plan_year", "street", "NBHD_L", "action", "miles", "homes", "owners", "chance", "network_position",
       "first", "med_yr_built", "is_ej", "cost"]].rename(columns={
        "plan_year": "Year", "street": "Street", "NBHD_L": "Neighborhood", "action": action_label,
        "miles": "Miles", "homes": "Homes", "owners": "Owners on branch", "chance": "Chance all agree (%)",
        "network_position": "Network position", "first": "Highest risk", "med_yr_built": "Building year",
        "is_ej": "EJ area", "cost": "Expected cost ($)"})
    show = st.radio("Show", ["All flagged streets", "Through streets", "Dead-end branches"],
                    horizontal=True, label_visibility="collapsed")
    if show == "Through streets":
        table = table[table["Network position"] == "Through street"]
    elif show == "Dead-end branches":
        table = table[table["Network position"] != "Through street"]
    table = table.sort_values(["Year", "Street"], na_position="last")
    st.dataframe(table.round({"Miles": 3}), width="stretch", height=340, hide_index=True)
    st.caption("Owners on branch and chance all agree apply to the whole dead-end branch the street belongs to.")
    st.download_button("Download plan as CSV", table.to_csv(index=False), "boston_gas_triage_plan.csv", "text/csv")

# ---------------- Street types
with tab_types:
    st.markdown("Streets grouped by k-means clustering on building age, housing density, large-building gas use, "
                "gas heat share, renter share, income, and heat pump share.")
    miles_by_type = df.groupby("archetype_name", as_index=False)["miles"].sum()
    known = [n for n in TYPE_COLORS if n in set(miles_by_type["archetype_name"])]
    st.altair_chart(alt.Chart(miles_by_type).mark_bar().encode(
        y=alt.Y("archetype_name:N", sort="-x", title=None),
        x=alt.X("miles:Q", title="Street miles"),
        color=alt.Color("archetype_name:N", legend=None,
                        scale=alt.Scale(domain=known, range=["#%02x%02x%02x" % tuple(TYPE_COLORS[n]) for n in known])),
        tooltip=[alt.Tooltip("archetype_name:N", title="Street type"), alt.Tooltip("miles:Q", title="Miles", format=",.0f")]
    ).properties(title="Boston street miles by type", height=260), width="stretch")
    prof = (df.groupby("archetype_name")
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

**Options and costs** (present value, {basis_short} cost)
- **Replace:** miles × all-in cost per mile (\\${cost_per_mile / 1e6:.2f}M).
- **Seal joints:** {seal_frac:.0%} of replacement now, plus replacement {seal_life} years later, discounted at
  {discount:.1%}. Only on major roads (road classes A1 to A3), as a stand-in for mains 12 inches and larger.
- **Electrify:** homes × \\${hp_cost:,.0f}; **geothermal:** homes × \\${geo_cost:,.0f}. For a whole dead-end
  branch: expected cost = chance all agree × alternative + chance someone declines × pipe work + leak repairs during
  the lead time. A branch is eligible only if it could come off gas as a unit, has no large BERDO buildings, and isn't
  in the highest-risk tier.
- **Chance all agree:** each property owner (one per address) agrees with probability
  {p_owner:.0%} × (1 − renter share) + {p_landlord:.0%} × renter share, from the block group's Census renter share.
  Every owner must agree.
- **Schedule:** street projects and branches, highest-risk tier first, then by risk-weighted miles per dollar, placed
  in the first year with room in both the budget and crew capacity.

**Data sources**
- [DPU, Report to the Legislature on the Prevalence of Natural Gas Leaks, D.P.U. 25-GLR-01 (Dec 31, 2025)](https://malegislature.gov/Bills/194/SD3561.pdf):
  GSEP spending and miles (cost per mile), Boston Gas leak-prone miles (2,486) and 2025 plan (120 miles), leak counts
  (9,912 newly found in 2024), repair backlog cost (\\$18.5M for 9,077 leaks), the 10-year standard for advanced
  leak repair, and the Grade 1 repair requirement.
- DPU GSEP orders (April 30, 2025), as reported by State House News: 4,049 miles of leak-prone pipe remaining; cap
  of 2.5%, then 2% (2026) and 1.5% (2027).
- DPU GSEP Working Group minutes (Oct 20, 2023): lifetime customer cost per \\$1 of GSEP capital (\\${default_mult:.2f}).
  GSEP Working Group minutes (April 12, 2023): Framingham geothermal pilot approved October 2020.
- Joint sealing: [ENR, "Joint-Filling Robot is Game-Changer for Gas Utilities"](https://www.enr.com/articles/43945-joint-filling-robot-is-game-changer-for-gas-utilities)
  (about one-tenth of replacement cost; sealant lab-tested to 50 years);
  [ULC Technologies](https://ulctechnologies.com/services/cisbot-robotic-cast-iron-joint-sealing/) (12-48 inch cast
  iron; Con Edison job \\$400,000 instead of \\$1.5-1.8 million; Cadent: 9 weeks instead of 9 months).
- Networked geothermal: [Framingham pilot, about \\$15 million for about 135 customers](https://gcn.com/massachusetts-pilot-connects-135-homes-shared/22333);
  [Lowell pilot canceled, \\$15.6 million for 31 customers](https://bankerandtradesman.com/national-grid-pulls-plug-on-residential-geothermal-pilot-in-lowell/);
  [about 80% of homeowners along the Framingham route joined](https://www.eversource.com/residential/save-money-energy/clean-energy-options/geothermal-energy/geothermal-pilot-reference-guide).
- Discount rate: OMB Circular A-4 (2% to 7%).
- City of Boston open data: street segments (SAM), FY2026 property assessment, BERDO building energy reporting.
- U.S. Census Bureau: ACS 5-year estimates and block group boundaries. MassGIS: 2020 Environmental Justice populations.

**Assumptions without a direct source:** cost uncertainty for each option (replace ±{unc_vals[REPLACE]}%, seal
±{unc_vals[SEAL]}%, electrify ±{unc_vals[ELECTRIFY]}%, geothermal ±{unc_vals[GEO]}%), the 60-year life used for
geothermal, chance a landlord agrees ({p_landlord:.0%}), years to arrange
electrification ({lead_e}), the highest-risk share ({immediate_share:.0%}), the road classes used for sealing, and the
30 m and 60 m distances in the network check. All are adjustable or stated.

**What would reduce the uncertainty, for every option**
- **Pilots with shared results.** Joint pilots by utilities and technology providers on close-call streets, with cost,
  consent, and performance data kept in a shared public repository.
- **Tracking readiness.** Reviewing the readiness table as pilots report, and narrowing cost ranges as data comes in.
- **Regulatory clarity.** A regulatory sandbox for pilots would shorten lead times and reduce approval risk.
- **Longer planning horizons.** Comparing options over the life of the assets, with credit for life left at the end.
- **Aligned incentives.** Financing all options on the same basis, so that how an option is paid for doesn't drive
  the choice.

**Limits that remain**
- The risk score is provisional until leak records are added.
- Pipe material, diameter, and the utility's network maps are estimated, not known.
- Grid upgrades, panel upgrades, weatherization, monthly bills, and lining small mains aren't costed.
- Paving schedules aren't in the city's open data.

Created by {CREATOR}. Built with public data only. Not an official DPU or utility analysis.
""")
