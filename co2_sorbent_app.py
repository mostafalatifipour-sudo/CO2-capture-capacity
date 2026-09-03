"""
CO2 Sorbent Capacity & Kinetics Analyzer
-----------------------------------------
For calcium-looping TGA data: drag-select the initial (post-calcination) and
final (carbonation plateau) mass of each cycle directly on the curve, get
mmol CO2/g for each cycle, and fit any of eight kinetic models to each
selected section independently.

Run locally:
    pip install "streamlit>=1.35" pandas numpy scipy plotly
    streamlit run co2_sorbent_app.py

Deploy online (free): push this file + requirements.txt to a GitHub repo,
then connect it at https://share.streamlit.io (Streamlit Community Cloud).

Note: click-drag chart selection requires Streamlit 1.35 or newer.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit

# ---------------------------------------------------------------- constants
M_CO2 = 44.01

st.set_page_config(page_title="CO2 Sorbent Analyzer", layout="wide")

# ---------------------------------------------------------------- generic helpers
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


def nearest_row(df, t):
    return df.iloc[(df["t"] - t).abs().idxmin()]


PARAM_LABELS = {"k": "k", "n": "n", "psi": "\u03c8", "f": "f", "k1": "k\u2081", "k2": "k\u2082", "kr": "k_r", "kd": "k_d"}

# ---------------------------------------------------------------- model 1-4: existing (linearizable)
def diffusion_f(X):
    return 1 - (2 / 3) * X - (1 - X) ** (2 / 3)


def diffusion_inv_scalar(y):
    lo, hi = 0.0, 0.999
    for _ in range(40):
        mid = (lo + hi) / 2
        if diffusion_f(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fit_pfo(t, X):
    r = linreg(t, -np.log(1 - X))
    return None if r is None else {"k": r["slope"]}


def invert_pfo(p, t):
    return 1 - np.exp(-p["k"] * t)


def fit_avrami(t, X):
    r = linreg(np.log(t), np.log(-np.log(1 - X)))
    return None if r is None else {"k": np.exp(r["intercept"]), "n": r["slope"]}


def invert_avrami(p, t):
    return 1 - np.exp(-p["k"] * t ** p["n"])


def fit_scm_r(t, X):
    r = linreg(t, 1 - (1 - X) ** (1 / 3))
    return None if r is None else {"k": r["slope"]}


def invert_scm_r(p, t):
    y = min(p["k"] * t, 1)
    return 1 - (1 - y) ** 3


def fit_scm_d(t, X):
    r = linreg(t, diffusion_f(X))
    return None if r is None else {"k": r["slope"]}


def invert_scm_d(p, t):
    return diffusion_inv_scalar(p["k"] * t)


# ---------------------------------------------------------------- model 5: Random Pore Model
def _rpm_curve(t, k, psi):
    val = (1 - (1 + psi * k * t) ** 2) / psi
    return 1 - np.exp(np.clip(val, -50, 0))


def fit_rpm(t, X):
    t_med = np.median(t) if len(t) else 1.0
    p0 = [0.5 / t_med if t_med > 0 else 0.1, 1.0]
    try:
        popt, _ = curve_fit(_rpm_curve, t, X, p0=p0, bounds=([1e-8, 1e-6], [np.inf, 50]), maxfev=8000)
        return {"k": popt[0], "psi": popt[1]}
    except Exception:
        return None


def invert_rpm(p, t):
    return _rpm_curve(np.array([t]), p["k"], p["psi"])[0]


# ---------------------------------------------------------------- model 6: Double Exponential Model
def _dem_curve(t, f, k1, k2):
    return f * (1 - np.exp(-k1 * t)) + (1 - f) * (1 - np.exp(-k2 * t))


def fit_dem(t, X):
    t_med = np.median(t) if len(t) else 1.0
    p0 = [0.7, 1.0 / t_med if t_med > 0 else 0.1, 0.1 / t_med if t_med > 0 else 0.01]
    try:
        popt, _ = curve_fit(_dem_curve, t, X, p0=p0, bounds=([0, 1e-8, 1e-8], [1, np.inf, np.inf]), maxfev=8000)
        return {"f": popt[0], "k1": popt[1], "k2": popt[2]}
    except Exception:
        return None


def invert_dem(p, t):
    return _dem_curve(np.array([t]), p["f"], p["k1"], p["k2"])[0]


# ---------------------------------------------------------------- model 7: Changing Grain Size Model
def _cgsm_g1(X):
    return 1 - (1 - X) ** (1 / 3)


def _cgsm_g2(X):
    return 1 - 3 * (1 - X) ** (2 / 3) + 2 * (1 - X)


def fit_cgsm(t, X):
    A = np.column_stack([_cgsm_g1(X), _cgsm_g2(X)])
    try:
        coef, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        a, b = coef
        if a <= 0 or b <= 0:
            return None
        return {"kr": 1 / a, "kd": 1 / b, "_a": a, "_b": b}
    except Exception:
        return None


def invert_cgsm(p, t):
    a, b = p["_a"], p["_b"]
    lo, hi = 0.0, 0.999
    for _ in range(40):
        mid = (lo + hi) / 2
        val = a * _cgsm_g1(mid) + b * _cgsm_g2(mid)
        if val < t:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- model 8: generalized nth-order
def _nth_curve(t, k, n):
    out = np.empty_like(t, dtype=float)
    for i, ti in enumerate(t):
        if abs(n - 1) < 1e-6:
            out[i] = 1 - np.exp(-k * ti)
        else:
            base = max(1 - k * (1 - n) * ti, 1e-12)
            out[i] = min(max(1 - base ** (1 / (1 - n)), 0), 1)
    return out


def fit_nth(t, X):
    t_med = np.median(t) if len(t) else 1.0
    p0 = [0.5 / t_med if t_med > 0 else 0.1, 1.2]
    try:
        popt, _ = curve_fit(_nth_curve, t, X, p0=p0, bounds=([1e-8, 0.1], [np.inf, 4]), maxfev=8000)
        return {"k": popt[0], "n": popt[1]}
    except Exception:
        return None


def invert_nth(p, t):
    return _nth_curve(np.array([t]), p["k"], p["n"])[0]


# ---------------------------------------------------------------- model registry
MODEL_DEFS = {
    "Pseudo first-order": {
        "eq": "-ln(1-X) = k\u00b7t", "fit": fit_pfo, "invert": invert_pfo, "params": ["k"],
    },
    "Avrami-Erofeev (JMAK)": {
        "eq": "ln(-ln(1-X)) = ln k + n\u00b7ln t", "fit": fit_avrami, "invert": invert_avrami, "params": ["k", "n"],
    },
    "Shrinking core - reaction control": {
        "eq": "1-(1-X)^(1/3) = k\u00b7t", "fit": fit_scm_r, "invert": invert_scm_r, "params": ["k"],
    },
    "Shrinking core - product-layer diffusion": {
        "eq": "1-(2/3)X-(1-X)^(2/3) = k\u00b7t", "fit": fit_scm_d, "invert": invert_scm_d, "params": ["k"],
    },
    "Random Pore Model (RPM)": {
        "eq": "k\u00b7t = (1/\u03c8)[\u221a(1-\u03c8\u00b7ln(1-X)) - 1]  (Bhatia & Perlmutter)",
        "fit": fit_rpm, "invert": invert_rpm, "params": ["k", "psi"],
    },
    "Double Exponential Model (DEM)": {
        "eq": "X = f(1-e^(-k\u2081t)) + (1-f)(1-e^(-k\u2082t))  (Sun et al.)",
        "fit": fit_dem, "invert": invert_dem, "params": ["f", "k1", "k2"],
    },
    "Changing Grain Size Model (CGSM)": {
        "eq": "t = (1/k_r)[1-(1-X)^(1/3)] + (1/k_d)[1-3(1-X)^(2/3)+2(1-X)]  (Szekely & Evans)",
        "fit": fit_cgsm, "invert": invert_cgsm, "params": ["kr", "kd"],
    },
    "Generalized nth-order model": {
        "eq": "dX/dt = k(1-X)^n", "fit": fit_nth, "invert": invert_nth, "params": ["k", "n"],
    },
}
MODEL_NAMES = list(MODEL_DEFS.keys())


def fit_model(model_name, t_arr, X_arr):
    d = MODEL_DEFS[model_name]
    t_arr = np.asarray(t_arr, dtype=float)
    X_arr = np.asarray(X_arr, dtype=float)
    params = d["fit"](t_arr, X_arr)
    if params is None:
        return None
    X_pred = np.array([d["invert"](params, t) for t in t_arr])
    ss_res = np.sum((X_arr - X_pred) ** 2)
    ss_tot = np.sum((X_arr - X_arr.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {"params": params, "r2": r2}


def show_params(cols, fit, prefix=""):
    d = MODEL_DEFS[prefix] if prefix in MODEL_DEFS else None
    for i, (key, val) in enumerate(fit["params"].items()):
        if key.startswith("_"):
            continue
        label = PARAM_LABELS.get(key, key)
        cols[i % len(cols)].metric(label, f"{val:.4g}")


# ---------------------------------------------------------------- header
st.markdown(
    "<div style='font-size:12px;letter-spacing:.03em;color:#5B665F;text-transform:uppercase;'>"
    "Calcium looping &middot; TGA analysis</div>",
    unsafe_allow_html=True,
)
st.title("CO\u2082 Sorbent Capacity & Kinetics")
st.caption(
    "Drag-select a cycle's window directly on the curve \u2014 from just after calcination to the "
    "carbonation plateau \u2014 to get its capacity in mmol CO2/g, then fit a kinetic model to each "
    "selected section independently."
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
st.header("2. Map columns")
cols = list(raw.columns)


def guess(patterns):
    for p in patterns:
        for c in cols:
            if p in c.lower():
                return c
    return cols[0]


c1, c2, c3 = st.columns(3)
time_col = c1.selectbox("Time column", cols, index=cols.index(guess(["time"])))
weight_col = c2.selectbox("Weight column", cols, index=cols.index(guess(["weight", "mass"])))
temp_options = ["\u2014 none \u2014"] + cols
temp_guess = guess(["temp"])
temp_default = cols.index(temp_guess) + 1 if any("temp" in c.lower() for c in cols) else 0
temp_col = c3.selectbox("Temperature column (optional)", temp_options, index=temp_default)

st.caption(
    "mmol CO2/g is computed as a mass ratio, so the weight column can be in mg or in % \u2014 units cancel out: "
    "mmol/g = ((final \u2212 initial) / initial) \u00d7 1000 / 44.01."
)

df = raw[[time_col, weight_col] + ([temp_col] if temp_col != "\u2014 none \u2014" else [])].copy()
df.columns = ["t", "w"] + (["temp"] if temp_col != "\u2014 none \u2014" else [])
df = df.dropna(subset=["t", "w"]).sort_values("t").reset_index(drop=True)

# ---------------------------------------------------------------- step 3: drag-select chart
st.header("3. Select each cycle")
st.caption(
    "Use the box-select tool (top-right of the chart) and drag across a cycle, starting just after "
    "calcination and ending once carbonation has plateaued. Then click 'Add cycle from selection' below."
)

if "cycles" not in st.session_state:
    st.session_state.cycles = pd.DataFrame(
        columns=["label", "t_start", "t_end", "initial_mass", "final_mass", "model", "two_stage", "transition_X"]
    )

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["t"], y=df["w"], mode="lines+markers",
                          marker=dict(size=3, opacity=0), line=dict(color="#2F6F5E", width=1.6), name="weight"))
if "temp" in df.columns:
    fig.add_trace(go.Scatter(x=df["t"], y=df["temp"], mode="lines", name="temperature",
                              line=dict(color="#B5482F", width=1, dash="dot"), yaxis="y2"))
for _, row in st.session_state.cycles.iterrows():
    fig.add_vrect(x0=row["t_start"], x1=row["t_end"], fillcolor="#2F6F5E", opacity=0.12, line_width=0,
                  annotation_text=row["label"], annotation_position="top left", annotation_font_size=10)
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10), dragmode="select",
    xaxis_title="time", yaxis_title="weight",
    yaxis2=dict(title="temperature", overlaying="y", side="right"),
    legend=dict(orientation="h", y=1.06), plot_bgcolor="white",
)

event = st.plotly_chart(
    fig, use_container_width=True, on_select="rerun", selection_mode=("box",), key="tga_chart"
)

t0 = t1 = None
sel = event.get("selection", {}) if event else {}
box = sel.get("box", [])
if box:
    x0, x1 = box[0]["x"]
    t0, t1 = min(x0, x1), max(x0, x1)
elif sel.get("points"):
    xs = [p["x"] for p in sel["points"] if p.get("curve_number") == 0]
    if xs:
        t0, t1 = min(xs), max(xs)

colA, colB = st.columns([3, 1])
if t0 is not None and t1 is not None and t1 > t0:
    r0, r1 = nearest_row(df, t0), nearest_row(df, t1)
    colA.write(f"Current selection: t = {t0:.2f} \u2192 {t1:.2f}  |  mass = {r0['w']:.4f} \u2192 {r1['w']:.4f}")
    if colB.button("Add cycle from selection"):
        n = len(st.session_state.cycles) + 1
        new_row = pd.DataFrame([{
            "label": f"Cycle {n}", "t_start": r0["t"], "t_end": r1["t"],
            "initial_mass": r0["w"], "final_mass": r1["w"],
            "model": "Pseudo first-order", "two_stage": False, "transition_X": 0.75,
        }])
        st.session_state.cycles = pd.concat([st.session_state.cycles, new_row], ignore_index=True)
        st.rerun()
else:
    colA.write("No active box selection \u2014 drag a rectangle on the chart above.")

# ---------------------------------------------------------------- step 4: cycle table
st.header("4. Cycles")
st.caption("Fine-tune values here if needed, choose a kinetic model per cycle, and remove rows with the trash icon.")

cycles_df = st.data_editor(
    st.session_state.cycles,
    num_rows="dynamic",
    use_container_width=True,
    key="cycle_editor",
    column_config={
        "label": st.column_config.TextColumn("Label"),
        "t_start": st.column_config.NumberColumn("t start", format="%.3f"),
        "t_end": st.column_config.NumberColumn("t end", format="%.3f"),
        "initial_mass": st.column_config.NumberColumn("Initial mass", format="%.5f"),
        "final_mass": st.column_config.NumberColumn("Final mass", format="%.5f"),
        "model": st.column_config.SelectboxColumn("Kinetic model", options=MODEL_NAMES, width="medium"),
        "two_stage": st.column_config.CheckboxColumn("Two-stage fit"),
        "transition_X": st.column_config.NumberColumn("Transition X", min_value=0.1, max_value=0.95, step=0.05, format="%.2f"),
    },
)
st.session_state.cycles = cycles_df

valid = cycles_df.dropna(subset=["t_start", "t_end", "initial_mass", "final_mass"])
valid = valid[valid["t_end"] > valid["t_start"]]

if valid.empty:
    st.info("Add at least one cycle above to see capacity and kinetics.")
    st.stop()

# ---------------------------------------------------------------- step 5: capacity results
st.header("5. Capacity results")

cap_table = valid.copy()
cap_table["mmol CO2 / g"] = ((cap_table["final_mass"] - cap_table["initial_mass"]) / cap_table["initial_mass"]) * 1000 / M_CO2
st.dataframe(
    cap_table[["label", "t_start", "t_end", "initial_mass", "final_mass", "mmol CO2 / g"]]
    .rename(columns={"label": "Cycle", "t_start": "t start", "t_end": "t end"})
    .round(4),
    use_container_width=True, hide_index=True,
)

if len(cap_table) > 1:
    cap_fig = go.Figure()
    cap_fig.add_trace(go.Scatter(x=cap_table["label"], y=cap_table["mmol CO2 / g"],
                                  mode="lines+markers", line=dict(color="#2F6F5E", width=2)))
    cap_fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="mmol CO2 / g", plot_bgcolor="white")
    st.plotly_chart(cap_fig, use_container_width=True)

# ---------------------------------------------------------------- step 6: per-cycle kinetics
st.header("6. Kinetics per cycle")
st.caption(
    "Each cycle is fit with the model chosen for it in the table above. X is normalized to that cycle's own "
    "selected initial and final mass (0 at selection start, 1 at selection end). R\u00b2 is computed on actual vs. "
    "predicted conversion, so it's comparable across every model. Models with more parameters (DEM, CGSM, RPM, "
    "nth-order) need more data points in the window to fit reliably."
)

for _, row in valid.iterrows():
    with st.expander(f"{row['label']}  \u2014  {row['model']}" + ("  (two-stage)" if row["two_stage"] else "")):
        window = df[(df["t"] >= row["t_start"]) & (df["t"] <= row["t_end"])].copy()
        window["t_rel"] = window["t"] - row["t_start"]
        span = row["final_mass"] - row["initial_mass"]
        if span == 0 or len(window) < 3:
            st.warning("Not enough points, or zero mass change, in this selection.")
            continue
        window["X"] = ((window["w"] - row["initial_mass"]) / span).clip(0.0005, 0.999)
        window = window[window["t_rel"] > 0]

        t_arr, X_arr = window["t_rel"].values, window["X"].values
        kin_fig = go.Figure()
        kin_fig.add_trace(go.Scatter(x=t_arr, y=X_arr, mode="markers",
                                      marker=dict(color="#5B665F", opacity=0.5), name="data"))
        t_curve = np.linspace(t_arr.min(), t_arr.max(), 60) if len(t_arr) else np.array([])

        compare_mode = st.checkbox("Compare all models", key=f"compare_{row.name}_{row['label']}")

        if compare_mode:
            palette = ["#2F6F5E", "#B5482F", "#4A6FA5", "#B08A2E", "#7A4E9E", "#3C8C8C", "#8E6C3A", "#6B7B4F"]
            results_list = []
            for i, name in enumerate(MODEL_NAMES):
                fit = fit_model(name, t_arr, X_arr) if len(t_arr) >= 3 else None
                if fit:
                    d = MODEL_DEFS[name]
                    y_curve = np.clip([d["invert"](fit["params"], t) for t in t_curve], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=t_curve, y=y_curve, mode="lines",
                                                  line=dict(color=palette[i % len(palette)], width=2), name=name))
                    params_str = ", ".join(
                        f"{PARAM_LABELS.get(k, k)}={v:.4g}" for k, v in fit["params"].items() if not k.startswith("_")
                    )
                    results_list.append((name, fit["r2"], params_str))
                else:
                    results_list.append((name, None, "fit failed"))
            st.plotly_chart(kin_fig, use_container_width=True)
            results_list.sort(key=lambda r: (r[1] is None, -(r[1] if r[1] is not None else 0)))
            comp_df = pd.DataFrame([
                {"Model": n, "R\u00b2": (f"{r2:.4f}" if r2 is not None else "\u2014"), "Parameters": p}
                for n, r2, p in results_list
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            best = results_list[0]
            if best[1] is not None:
                st.caption(f"Best fit for this cycle: **{best[0]}** (R\u00b2 = {best[1]:.4f}).")
        elif not row["two_stage"]:
            fit = fit_model(row["model"], t_arr, X_arr) if len(t_arr) >= 3 else None
            if fit:
                d = MODEL_DEFS[row["model"]]
                y_curve = np.clip([d["invert"](fit["params"], t) for t in t_curve], 0, 1)
                kin_fig.add_trace(go.Scatter(x=t_curve, y=y_curve, mode="lines",
                                              line=dict(color="#2F6F5E", width=2), name="fit"))
                st.plotly_chart(kin_fig, use_container_width=True)
                display_params = {k: v for k, v in fit["params"].items() if not k.startswith("_")}
                metric_cols = st.columns(len(display_params) + 1)
                for i, (key, val) in enumerate(display_params.items()):
                    metric_cols[i].metric(PARAM_LABELS.get(key, key), f"{val:.4g}")
                metric_cols[-1].metric("R\u00b2", f"{fit['r2']:.4f}")
                st.caption(f"*{d['eq']}*")
            else:
                st.warning("Couldn't fit this model to this cycle \u2014 try a simpler model or a wider selection.")
        else:
            tx = row["transition_X"]
            mask1, mask2 = X_arr <= tx, X_arr > tx
            if mask1.sum() >= 3 and mask2.sum() >= 3:
                f1 = fit_model("Pseudo first-order", t_arr[mask1], X_arr[mask1])
                f2 = fit_model("Shrinking core - product-layer diffusion", t_arr[mask2], X_arr[mask2])
                t_split = t_arr[mask2].min()
                c1, c2 = t_curve[t_curve <= t_split], t_curve[t_curve > t_split]
                if f1 and len(c1):
                    y1 = np.clip([MODEL_DEFS["Pseudo first-order"]["invert"](f1["params"], t) for t in c1], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c1, y=y1, mode="lines", line=dict(color="#2F6F5E", width=2), name="fast stage"))
                if f2 and len(c2):
                    y2 = np.clip([MODEL_DEFS["Shrinking core - product-layer diffusion"]["invert"](f2["params"], t) for t in c2], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c2, y=y2, mode="lines", line=dict(color="#B5482F", width=2), name="diffusion stage"))
                kin_fig.add_vline(x=t_split, line_dash="dot", line_color="#B5482F")
                st.plotly_chart(kin_fig, use_container_width=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("k\u2081 fast (1/time)", f"{f1['params']['k']:.4g}" if f1 else "\u2014")
                mc2.metric("R\u00b2 fast", f"{f1['r2']:.4f}" if f1 else "\u2014")
                mc3.metric("k\u2082 diffusion (1/time)", f"{f2['params']['k']:.4g}" if f2 else "\u2014")
                mc4.metric("R\u00b2 diffusion", f"{f2['r2']:.4f}" if f2 else "\u2014")
                st.caption(
                    f"Fast stage: *{MODEL_DEFS['Pseudo first-order']['eq']}*. "
                    f"Diffusion stage: *{MODEL_DEFS['Shrinking core - product-layer diffusion']['eq']}*, "
                    f"split at X = {tx:.2f}."
                )
            else:
                st.warning("Not enough points on one side of the transition to fit both stages.")
