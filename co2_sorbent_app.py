"""
CO2 Sorbent Capacity & Kinetics Analyzer
-----------------------------------------
For calcium-looping TGA data: computes cyclic CO2 capture capacity from
calcination/carbonation weight steps, and fits carbonation kinetic models.

Run locally:
    pip install streamlit pandas numpy plotly
    streamlit run co2_sorbent_app.py

Deploy online (free):
    - Streamlit Community Cloud: push this file + a requirements.txt to a
      GitHub repo, then connect it at https://share.streamlit.io
    - Hugging Face Spaces: create a Space with the "Streamlit" SDK and
      upload this file as app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------- constants
M_CO2 = 44.01
M_CAO = 56.08
THEORETICAL_GG_PER_UNIT_ACTIVE = M_CO2 / M_CAO  # 0.7847 g CO2 / g CaO, full conversion

st.set_page_config(page_title="CO2 Sorbent Analyzer", layout="wide")

# ---------------------------------------------------------------- math helpers
def linreg(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"slope": slope, "intercept": intercept, "r2": r2}


def diffusion_f(X):
    return 1 - (2 / 3) * X - (1 - X) ** (2 / 3)


def diffusion_inv(y):
    lo, hi = 0.0, 0.999
    for _ in range(40):
        mid = (lo + hi) / 2
        if diffusion_f(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


MODELS = {
    "first": {
        "name": "Pseudo first-order",
        "eq": "-ln(1-X) = k*t",
        "transform": lambda X, t: (t, -np.log(1 - X)),
        "invert": lambda k, n, t: 1 - np.exp(-k * t),
        "k_units": "1/time",
        "has_n": False,
    },
    "avrami": {
        "name": "Avrami-Erofeev (JMAK)",
        "eq": "ln(-ln(1-X)) = ln k + n*ln t",
        "transform": lambda X, t: (np.log(t), np.log(-np.log(1 - X))),
        "invert": lambda k, n, t: 1 - np.exp(-k * t ** n),
        "k_units": "1/time^n",
        "has_n": True,
    },
    "scm_reaction": {
        "name": "Shrinking core - reaction control",
        "eq": "1-(1-X)^(1/3) = k*t",
        "transform": lambda X, t: (t, 1 - (1 - X) ** (1 / 3)),
        "invert": lambda k, n, t: 1 - (1 - min(k * t, 1)) ** 3,
        "k_units": "1/time",
        "has_n": False,
    },
    "scm_diffusion": {
        "name": "Shrinking core - product-layer diffusion",
        "eq": "1-(2/3)X-(1-X)^(2/3) = k*t",
        "transform": lambda X, t: (t, diffusion_f(X)),
        "invert": lambda k, n, t: diffusion_inv(k * t),
        "k_units": "1/time",
        "has_n": False,
    },
}


def fit_model(model_key, t_arr, X_arr):
    m = MODELS[model_key]
    x_list, y_list = [], []
    for t, X in zip(t_arr, X_arr):
        x, y = m["transform"](X, t)
        if np.isfinite(x) and np.isfinite(y):
            x_list.append(x)
            y_list.append(y)
    fit = linreg(x_list, y_list)
    if fit is None:
        return None
    if m["has_n"]:
        k, n = np.exp(fit["intercept"]), fit["slope"]
    else:
        k, n = fit["slope"], None
    return {"k": k, "n": n, "r2": fit["r2"], "model": model_key}


# ---------------------------------------------------------------- UI: header
st.markdown(
    "<div style='font-size:12px;letter-spacing:.03em;color:#5B665F;text-transform:uppercase;'>"
    "Calcium looping &middot; TGA analysis</div>",
    unsafe_allow_html=True,
)
st.title("CO\u2082 Sorbent Capacity & Kinetics")
st.caption(
    "Upload thermogravimetric cycling data for a CaO sorbent, mark the calcination baseline "
    "and carbonation windows for each cycle, and fit a carbonation kinetic model to any cycle."
)

# ---------------------------------------------------------------- step 1: upload
st.header("1. Upload TGA data")
uploaded = st.file_uploader("CSV with a time column and a weight column (mg or %)", type="csv")

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

raw = pd.read_csv(uploaded)
st.success(f"{uploaded.name} \u2014 {len(raw):,} rows parsed")

# ---------------------------------------------------------------- step 2: column mapping
st.header("2. Map columns & sample")
cols = list(raw.columns)


def guess(patterns):
    for p in patterns:
        for c in cols:
            if p in c.lower():
                return c
    return cols[0]


c1, c2, c3, c4 = st.columns(4)
time_col = c1.selectbox("Time column", cols, index=cols.index(guess(["time"])))
weight_col = c2.selectbox("Weight column", cols, index=cols.index(guess(["weight", "mass"])))
temp_options = ["\u2014 none \u2014"] + cols
temp_guess = guess(["temp"])
temp_default = cols.index(temp_guess) + 1 if temp_guess in cols and any("temp" in c.lower() for c in cols) else 0
temp_col = c3.selectbox("Temperature column (optional)", temp_options, index=temp_default)
active_cao = c4.number_input("Active CaO content (wt%)", min_value=1.0, max_value=100.0, value=100.0)

st.caption(
    "Capacities are computed as mass ratios, so the weight column can be in mg or in % \u2014 the units cancel out. "
    "Active CaO content lets the theoretical maximum uptake account for inert support in a DFM; "
    "leave at 100% for unsupported CaO."
)

df = raw[[time_col, weight_col] + ([temp_col] if temp_col != "\u2014 none \u2014" else [])].copy()
df.columns = ["t", "w"] + (["temp"] if temp_col != "\u2014 none \u2014" else [])
df = df.dropna(subset=["t", "w"]).sort_values("t").reset_index(drop=True)
t_min, t_max = float(df["t"].min()), float(df["t"].max())
theoretical_max_gg = (active_cao / 100) * THEORETICAL_GG_PER_UNIT_ACTIVE

# ---------------------------------------------------------------- step 3: chart + cycles
st.header("3. Define cycles")
st.caption(
    "For each cycle, set the time window of the flat calcined baseline and the carbonation stage "
    "that follows it. Carbonation mass is averaged over the last 20% of its window, to approximate the plateau."
)

if "cycles" not in st.session_state:
    st.session_state.cycles = pd.DataFrame(
        [{"label": "Cycle 1", "baseline_start": t_min, "baseline_end": t_min, "carb_start": t_min, "carb_end": t_min}]
    )

cycles_df = st.data_editor(
    st.session_state.cycles,
    num_rows="dynamic",
    use_container_width=True,
    key="cycle_editor",
)
st.session_state.cycles = cycles_df


def window_mean(lo, hi, last_fraction=1.0):
    lo, hi = min(lo, hi), max(lo, hi)
    cut = lo if last_fraction >= 1 else hi - (hi - lo) * last_fraction
    sel = df[(df["t"] >= lo) & (df["t"] <= hi) & (df["t"] >= cut)]
    return sel["w"].mean() if len(sel) else None


results = []
for _, row in cycles_df.iterrows():
    try:
        bs, be, cs, ce = float(row.baseline_start), float(row.baseline_end), float(row.carb_start), float(row.carb_end)
        valid = be > bs and ce > cs
    except (TypeError, ValueError):
        valid = False
    rec = {"label": row.label, "valid": False}
    if valid:
        baseline = window_mean(bs, be, 1.0)
        carb = window_mean(cs, ce, 0.2)
        if baseline and carb and baseline > 0:
            delta_m = carb - baseline
            uptake_gg = delta_m / baseline
            uptake_mmol_g = uptake_gg * 1000 / M_CO2
            X = uptake_gg / theoretical_max_gg if theoretical_max_gg > 0 else None
            rec.update(
                valid=True, baseline_start=bs, baseline_end=be, carb_start=cs, carb_end=ce,
                baseline=baseline, carb=carb, delta_m=delta_m, uptake_gg=uptake_gg,
                uptake_mmol_g=uptake_mmol_g, X=X,
            )
    results.append(rec)

results_df = pd.DataFrame(results)

# main TGA chart with shaded windows
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["t"], y=df["w"], mode="lines", name="weight", line=dict(color="#2F6F5E", width=1.6)))
if "temp" in df.columns:
    fig.add_trace(go.Scatter(x=df["t"], y=df["temp"], mode="lines", name="temperature",
                              line=dict(color="#B5482F", width=1, dash="dot"), yaxis="y2"))
for rec in results:
    if rec.get("valid"):
        fig.add_vrect(x0=rec["baseline_start"], x1=rec["baseline_end"], fillcolor="#5B665F", opacity=0.12, line_width=0)
        fig.add_vrect(x0=rec["carb_start"], x1=rec["carb_end"], fillcolor="#2F6F5E", opacity=0.14, line_width=0)
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="time", yaxis_title="weight",
    yaxis2=dict(title="temperature", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.08),
    plot_bgcolor="white",
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- step 4: capacity results
st.header("4. Capacity results")
valid_results = [r for r in results if r.get("valid")]

if not valid_results:
    st.warning("Define at least one valid cycle window above (baseline_end > baseline_start and carb_end > carb_start).")
    st.stop()

table = pd.DataFrame(
    [
        {
            "Cycle": r["label"],
            "\u0394m / baseline (%)": round(r["uptake_gg"] * 100, 2),
            "mmol CO2 / g": round(r["uptake_mmol_g"], 3),
            "Conversion X": round(r["X"], 3) if r["X"] is not None else None,
        }
        for r in valid_results
    ]
)
st.dataframe(table, use_container_width=True, hide_index=True)

if any(r["X"] and r["X"] > 1 for r in valid_results):
    st.caption("\u26a0\ufe0f One or more cycles show X > 1 \u2014 check the active CaO content or window placement.")

if len(valid_results) > 1:
    cap_fig = go.Figure()
    cap_fig.add_trace(go.Scatter(
        x=[r["label"] for r in valid_results],
        y=[r["uptake_mmol_g"] for r in valid_results],
        mode="lines+markers", line=dict(color="#2F6F5E", width=2),
    ))
    cap_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="mmol CO2 / g", plot_bgcolor="white")
    st.plotly_chart(cap_fig, use_container_width=True)

# ---------------------------------------------------------------- step 5: kinetics
st.header("5. Carbonation kinetics")
st.caption(
    "Fit a kinetic model to the conversion curve of one cycle. Conversion X can be normalized to the "
    "theoretical maximum (from active CaO content) or to the experimental plateau reached in that cycle."
)

k1, k2, k3, k4 = st.columns(4)
cycle_labels = [r["label"] for r in valid_results]
sel_label = k1.selectbox("Cycle", cycle_labels)
norm_mode = k2.selectbox("Normalize X to", ["theoretical maximum", "experimental plateau"])
two_stage = k4.checkbox("Two-stage (fast + diffusion)")
model_key = k3.selectbox(
    "Model", list(MODELS.keys()), format_func=lambda k: MODELS[k]["name"], disabled=two_stage
)

transition_X = 0.75
if two_stage:
    transition_X = st.slider("Transition at X =", 0.1, 0.95, 0.75, 0.05)

sel = next(r for r in valid_results if r["label"] == sel_label)
cs, ce, baseline = sel["carb_start"], sel["carb_end"], sel["baseline"]
window = df[(df["t"] >= cs) & (df["t"] <= ce)].copy()
window["t_rel"] = window["t"] - cs
exp_max_gg = (sel["carb"] - baseline) / baseline
x_max_gg = theoretical_max_gg if norm_mode == "theoretical maximum" else exp_max_gg
window["X"] = ((window["w"] - baseline) / baseline / x_max_gg).clip(0.0005, 0.999)
window = window[window["t_rel"] > 0]

min_pts = 6 if two_stage else 3
if len(window) < min_pts:
    st.warning("Not enough points in this cycle's carbonation window to fit a model.")
    st.stop()

t_arr, X_arr = window["t_rel"].values, window["X"].values

kin_fig = go.Figure()
kin_fig.add_trace(go.Scatter(x=t_arr, y=X_arr, mode="markers", marker=dict(color="#5B665F", opacity=0.5), name="data"))

single_fit, two_fit = None, None
t_curve = np.linspace(t_arr.min(), t_arr.max(), 60)

if not two_stage:
    single_fit = fit_model(model_key, t_arr, X_arr)
    if single_fit:
        m = MODELS[model_key]
        X_curve = np.clip([m["invert"](single_fit["k"], single_fit["n"], t) for t in t_curve], 0, 1)
        kin_fig.add_trace(go.Scatter(x=t_curve, y=X_curve, mode="lines", line=dict(color="#2F6F5E", width=2), name="fit"))
else:
    mask1 = X_arr <= transition_X
    mask2 = ~mask1
    if mask1.sum() >= 3 and mask2.sum() >= 3:
        f1 = fit_model("first", t_arr[mask1], X_arr[mask1])
        f2 = fit_model("scm_diffusion", t_arr[mask2], X_arr[mask2])
        t_split = t_arr[mask2].min()
        two_fit = {"f1": f1, "f2": f2, "t_split": t_split}
        c1 = t_curve[t_curve <= t_split]
        c2 = t_curve[t_curve > t_split]
        if f1 is not None and len(c1):
            y1 = np.clip([MODELS["first"]["invert"](f1["k"], None, t) for t in c1], 0, 1)
            kin_fig.add_trace(go.Scatter(x=c1, y=y1, mode="lines", line=dict(color="#2F6F5E", width=2), name="fast-stage fit"))
        if f2 is not None and len(c2):
            y2 = np.clip([MODELS["scm_diffusion"]["invert"](f2["k"], None, t) for t in c2], 0, 1)
            kin_fig.add_trace(go.Scatter(x=c2, y=y2, mode="lines", line=dict(color="#B5482F", width=2), name="diffusion-stage fit"))
        kin_fig.add_vline(x=t_split, line_dash="dot", line_color="#B5482F")
    else:
        st.warning("Not enough points on one side of the transition to fit both stages.")

kin_fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="time since carbonation start", yaxis_title="X",
                       yaxis_range=[0, 1], plot_bgcolor="white")
st.plotly_chart(kin_fig, use_container_width=True)

if not two_stage and single_fit:
    m = MODELS[model_key]
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric(f"k ({m['k_units']})", f"{single_fit['k']:.4g}")
    if m["has_n"]:
        mc2.metric("n", f"{single_fit['n']:.3g}")
    mc3.metric("R\u00b2 (linearized)", f"{single_fit['r2']:.4f}")
    st.caption(f"*{m['eq']}*")
elif two_stage and two_fit and two_fit["f1"] and two_fit["f2"]:
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("k\u2081 fast stage (1/time)", f"{two_fit['f1']['k']:.4g}")
    mc2.metric("R\u00b2 fast stage", f"{two_fit['f1']['r2']:.4f}")
    mc3.metric("k\u2082 diffusion stage (1/time)", f"{two_fit['f2']['k']:.4g}")
    mc4.metric("R\u00b2 diffusion stage", f"{two_fit['f2']['r2']:.4f}")
    st.caption(
        f"Fast stage: *{MODELS['first']['eq']}* (kinetically controlled, surface reaction). "
        f"Diffusion stage: *{MODELS['scm_diffusion']['eq']}* (product-layer / CaCO3 shell diffusion control), "
        f"fit separately on either side of X = {transition_X:.2f}."
    )
