"""
CO2 Sorbent Capacity & Kinetics Analyzer
-----------------------------------------
For calcium-looping TGA data: drag-select cycles, get mmol CO2/g, fit any of
eight kinetic models per cycle (with model comparison via R2/SSE/AIC/BIC),
fit the Grasa-Abanades cyclic decay model, run an Arrhenius analysis across
cycles at different temperatures, and export tables and publication figures.

Run locally:
    pip install "streamlit>=1.35" pandas numpy scipy plotly matplotlib openpyxl
    streamlit run co2_sorbent_app.py

Deploy online (free): push this file + requirements.txt to a GitHub repo,
then connect it at https://share.streamlit.io (Streamlit Community Cloud).

Note: click-drag chart selection requires Streamlit 1.35 or newer.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
M_CO2 = 44.01
R_GAS = 8.314  # J / (mol K)

st.set_page_config(page_title="CO2 Sorbent Analyzer", layout="wide")

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.linewidth": 0.8,
    "figure.dpi": 150, "axes.edgecolor": "#333333", "text.color": "#1B2420",
    "axes.labelcolor": "#1B2420", "xtick.color": "#1B2420", "ytick.color": "#1B2420",
})

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


def df_download_buttons(df_export, basename, key_prefix):
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_export.to_excel(writer, index=False, sheet_name="data")
    buf.seek(0)
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", csv_bytes, file_name=f"{basename}.csv", mime="text/csv", key=f"{key_prefix}_csv")
    c2.download_button("Download Excel", buf, file_name=f"{basename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key_prefix}_xlsx")


def fig_download_buttons(fig, basename, key_prefix):
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300, bbox_inches="tight")
    buf_png.seek(0)
    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
    buf_pdf.seek(0)
    c1, c2 = st.columns(2)
    c1.download_button("Download PNG (300 dpi)", buf_png, file_name=f"{basename}.png", mime="image/png", key=f"{key_prefix}_png")
    c2.download_button("Download PDF (vector)", buf_pdf, file_name=f"{basename}.pdf", mime="application/pdf", key=f"{key_prefix}_pdf")


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
    resid = X_arr - X_pred
    sse = float(np.sum(resid ** 2))
    ss_tot = np.sum((X_arr - X_arr.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - sse / ss_tot
    n_pts = len(X_arr)
    k_params = len([k for k in params if not k.startswith("_")])
    sse_safe = max(sse, 1e-12)
    aic = n_pts * np.log(sse_safe / n_pts) + 2 * k_params
    bic = n_pts * np.log(sse_safe / n_pts) + k_params * np.log(n_pts)
    return {"params": params, "r2": r2, "sse": sse, "aic": aic, "bic": bic, "n": n_pts, "k_params": k_params}


def params_str(params):
    return ", ".join(f"{PARAM_LABELS.get(k, k)}={v:.4g}" for k, v in params.items() if not k.startswith("_"))


# ---------------------------------------------------------------- two-stage fitting (fast + diffusion)
def _pack_stage_fit(params, X_arr, X_pred, count_keys):
    resid = X_arr - X_pred
    sse = float(np.sum(resid ** 2))
    ss_tot = np.sum((X_arr - X_arr.mean()) ** 2)
    r2 = 1.0 if ss_tot == 0 else 1 - sse / ss_tot
    n_pts = len(X_arr)
    k_params = len(count_keys)
    sse_safe = max(sse, 1e-12)
    aic = n_pts * np.log(sse_safe / n_pts) + 2 * k_params
    bic = n_pts * np.log(sse_safe / n_pts) + k_params * np.log(n_pts)
    return {"params": params, "r2": r2, "sse": sse, "aic": aic, "bic": bic, "n": n_pts, "k_params": k_params}


def fit_stage1_fast(t_arr, X_arr):
    """Fast, reaction-controlled stage. Forced through the true origin (t=0, X=0),
    since this stage starts at the actual reaction onset."""
    r = linreg(t_arr, -np.log(1 - X_arr))
    if r is None:
        return None
    params = {"k": r["slope"]}
    X_pred = 1 - np.exp(-params["k"] * t_arr)
    fit = _pack_stage_fit(params, X_arr, X_pred, ["k"])
    fit["invert"] = lambda t: 1 - np.exp(-params["k"] * t)
    return fit


def fit_stage2_diffusion(t_arr, X_arr):
    """Slow, diffusion-controlled stage. This segment does NOT start at X=0 (it picks
    up mid-curve, at the transition conversion), so it needs its own local intercept
    rather than being forced through the origin -- forcing it through zero was the bug
    causing wildly negative R2 on this stage."""
    t0 = t_arr.min()
    t_local = t_arr - t0
    r = linreg(t_local, diffusion_f(X_arr))
    if r is None:
        return None
    params = {"k": r["slope"], "c": r["intercept"], "t0": t0}

    def invert(t):
        target = params["c"] + params["k"] * (t - params["t0"])
        return diffusion_inv_scalar(target)

    X_pred = np.array([invert(t) for t in t_arr])
    fit = _pack_stage_fit(params, X_arr, X_pred, ["k", "c"])
    fit["invert"] = invert
    return fit


def find_best_split(t_arr, X_arr, lo=0.2, hi=0.9, step=0.02):
    """Search candidate transition points and keep the one minimizing total SSE across
    both stages. Since both stages always use the same two sub-models regardless of
    where the split falls, and the combined point count doesn't change, minimizing
    total SSE here is equivalent to minimizing total AIC/BIC."""
    best = None
    for tx in np.arange(lo, hi + 1e-9, step):
        mask1, mask2 = X_arr <= tx, X_arr > tx
        if mask1.sum() < 3 or mask2.sum() < 3:
            continue
        f1 = fit_stage1_fast(t_arr[mask1], X_arr[mask1])
        f2 = fit_stage2_diffusion(t_arr[mask2], X_arr[mask2])
        if f1 is None or f2 is None:
            continue
        total_sse = f1["sse"] + f2["sse"]
        if best is None or total_sse < best["total_sse"]:
            best = {"tx": round(float(tx), 3), "f1": f1, "f2": f2, "total_sse": total_sse}
    return best


# ---------------------------------------------------------------- Grasa-Abanades cyclic decay model
def _decay_curve(N, cap0, capr, kd):
    return capr + (cap0 - capr) / (1 + kd * (N - 1))


def fit_decay(N_arr, cap_arr):
    N_arr = np.asarray(N_arr, dtype=float)
    cap_arr = np.asarray(cap_arr, dtype=float)
    p0 = [cap_arr[0], max(cap_arr.min(), 1e-3), 0.2]
    try:
        popt, _ = curve_fit(_decay_curve, N_arr, cap_arr, p0=p0,
                             bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=8000)
        pred = _decay_curve(N_arr, *popt)
        ss_res = np.sum((cap_arr - pred) ** 2)
        ss_tot = np.sum((cap_arr - cap_arr.mean()) ** 2)
        r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        return {"cap0": popt[0], "capr": popt[1], "kd": popt[2], "r2": r2}
    except Exception:
        return None


def prepare_kinetics(df, row, onset_pct):
    """Extract onset-trimmed (t, X) data for a cycle row. Returns None if not fittable."""
    window = df[(df["t"] >= row["t_start"]) & (df["t"] <= row["t_end"])].copy()
    window["t_rel"] = window["t"] - row["t_start"]
    span = row["final_mass"] - row["initial_mass"]
    if span == 0 or len(window) < 3:
        return None
    window["X"] = ((window["w"] - row["initial_mass"]) / span).clip(0.0005, 0.999)
    window = window[window["t_rel"] > 0]
    onset_mask = window["X"] >= onset_pct
    if onset_mask.sum() < 3:
        return None
    t_onset = window.loc[onset_mask, "t_rel"].iloc[0]
    excluded = window[window["t_rel"] < t_onset]
    kept = window[window["t_rel"] >= t_onset].copy()
    kept["t_fit"] = kept["t_rel"] - t_onset
    return {"kept": kept, "excluded": excluded, "t_onset": t_onset}


# ================================================================== HEADER
st.markdown(
    "<div style='font-size:12px;letter-spacing:.03em;color:#5B665F;text-transform:uppercase;'>"
    "Calcium looping &middot; TGA analysis</div>",
    unsafe_allow_html=True,
)
st.title("CO\u2082 Sorbent Capacity & Kinetics")
st.caption(
    "Drag-select a cycle's window directly on the curve, get its capacity in mmol CO2/g, fit and compare "
    "kinetic models, track cyclic decay, and export publication-ready tables and figures."
)

# ================================================================== STEP 1: upload
st.header("1. Upload TGA data")
uploaded = st.file_uploader("CSV with a time column and a weight column (mg or %)", type="csv")

if uploaded is None:
    st.info("Upload a CSV to get started.")
    st.stop()

raw = pd.read_csv(uploaded)
st.success(f"{uploaded.name} \u2014 {len(raw):,} rows parsed")

# ================================================================== STEP 2: column mapping
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
    "mmol CO2/g is computed as a mass ratio: mmol/g = ((final \u2212 initial) / initial) \u00d7 1000 / 44.01."
)

df = raw[[time_col, weight_col] + ([temp_col] if temp_col != "\u2014 none \u2014" else [])].copy()
df.columns = ["t", "w"] + (["temp"] if temp_col != "\u2014 none \u2014" else [])
df = df.dropna(subset=["t", "w"]).sort_values("t").reset_index(drop=True)

# ================================================================== STEP 3: drag-select chart
st.header("3. Select each cycle")
st.caption(
    "Use the box-select tool (top-right of the chart) and drag across a cycle, starting just after "
    "calcination and ending once carbonation has plateaued. Then click 'Add cycle from selection' below."
)

if "cycles" not in st.session_state:
    st.session_state.cycles = pd.DataFrame(
        columns=["label", "t_start", "t_end", "initial_mass", "final_mass", "model", "two_stage", "transition_X", "temperature"]
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
        temp_default_val = None
        if "temp" in df.columns:
            seg = df[(df["t"] >= r0["t"]) & (df["t"] <= r1["t"])]
            if len(seg):
                temp_default_val = float(seg["temp"].mean())
        new_row = pd.DataFrame([{
            "label": f"Cycle {n}", "t_start": r0["t"], "t_end": r1["t"],
            "initial_mass": r0["w"], "final_mass": r1["w"],
            "model": "Pseudo first-order", "two_stage": False, "transition_X": 0.75,
            "temperature": temp_default_val,
        }])
        st.session_state.cycles = pd.concat([st.session_state.cycles, new_row], ignore_index=True)
        st.rerun()
else:
    colA.write("No active box selection \u2014 drag a rectangle on the chart above.")

# ================================================================== STEP 4: cycle table
st.header("4. Cycles")
st.caption(
    "Fine-tune values here if needed, choose a kinetic model per cycle, remove rows with the trash icon. "
    "Temperature is optional and only needed for the Arrhenius analysis below."
)

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
        "temperature": st.column_config.NumberColumn("Carb. temp", format="%.1f"),
    },
)
st.session_state.cycles = cycles_df

valid = cycles_df.dropna(subset=["t_start", "t_end", "initial_mass", "final_mass"])
valid = valid[valid["t_end"] > valid["t_start"]]

if valid.empty:
    st.info("Add at least one cycle above to see capacity and kinetics.")
    st.stop()

# ================================================================== STEP 5: capacity results
st.header("5. Capacity results")

cap_table = valid.copy()
cap_table["mmol CO2 / g"] = ((cap_table["final_mass"] - cap_table["initial_mass"]) / cap_table["initial_mass"]) * 1000 / M_CO2
display_cap = (
    cap_table[["label", "t_start", "t_end", "initial_mass", "final_mass", "mmol CO2 / g"]]
    .rename(columns={"label": "Cycle", "t_start": "t start", "t_end": "t end"})
    .round(4)
)
st.dataframe(display_cap, use_container_width=True, hide_index=True)
df_download_buttons(display_cap, "capacity_results", "cap")

decay_fit = None
cap_fig = None
if len(cap_table) > 1:
    N_arr = np.arange(1, len(cap_table) + 1)
    cap_arr = cap_table["mmol CO2 / g"].values
    cap_fig = go.Figure()
    cap_fig.add_trace(go.Scatter(x=N_arr, y=cap_arr, mode="markers",
                                  marker=dict(color="#2F6F5E", size=8), name="observed"))
    fit_decay_toggle = st.checkbox("Fit cyclic decay model (Grasa & Abanades)", value=(len(cap_table) >= 3))
    if fit_decay_toggle:
        if len(cap_table) < 3:
            st.warning("Need at least 3 cycles to fit the decay model.")
        else:
            decay_fit = fit_decay(N_arr, cap_arr)
            if decay_fit:
                N_curve = np.linspace(1, len(cap_table), 100)
                cap_curve = _decay_curve(N_curve, decay_fit["cap0"], decay_fit["capr"], decay_fit["kd"])
                cap_fig.add_trace(go.Scatter(x=N_curve, y=cap_curve, mode="lines",
                                              line=dict(color="#B5482F", width=2), name="Grasa-Abanades fit"))
            else:
                st.warning("Decay model didn't converge on this data.")
    cap_fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis_title="mmol CO2 / g", plot_bgcolor="white",
                           xaxis=dict(tickmode="array", tickvals=list(N_arr), ticktext=list(cap_table["label"]), title="Cycle"))
    st.plotly_chart(cap_fig, use_container_width=True)
    if decay_fit:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Cap\u2080 (mmol/g)", f"{decay_fit['cap0']:.4g}")
        mc2.metric("Cap_r (mmol/g)", f"{decay_fit['capr']:.4g}")
        mc3.metric("k_d", f"{decay_fit['kd']:.4g}")
        mc4.metric("R\u00b2", f"{decay_fit['r2']:.4f}")
        st.caption("*Cap\u2099 = Cap_r + (Cap\u2080 \u2212 Cap_r) / (1 + k_d(N\u22121))*  \u2014  Grasa & Abanades (2006) cyclic decay model.")

# ================================================================== STEP 6: per-cycle kinetics
st.header("6. Kinetics per cycle")
st.caption(
    "Each cycle is fit with the model chosen for it in the table above. X is normalized to that cycle's own "
    "selected initial and final mass. R\u00b2, SSE, AIC and BIC are computed on actual vs. predicted conversion, "
    "so they're comparable across every model \u2014 AIC/BIC penalize extra parameters, so they're the fairer way "
    "to compare models with different numbers of parameters (e.g. DEM's 3 vs. first-order's 1). Lower AIC/BIC "
    "= better fit for its complexity."
)

onset_pct = st.slider(
    "Ignore pre-reaction dead time: ", 0.0, 15.0, 2.0, 0.5,
    help="Trims points before conversion crosses this percentage, and resets t=0 to that point, for every cycle below.",
    format="%.1f%%",
) / 100.0

for _, row in valid.iterrows():
    with st.expander(f"{row['label']}  \u2014  {row['model']}" + ("  (two-stage)" if row["two_stage"] else "")):
        prep = prepare_kinetics(df, row, onset_pct)
        if prep is None:
            st.warning("Not enough points, or conversion never rises above the dead-time threshold, in this selection.")
            continue
        kept, excluded, t_onset = prep["kept"], prep["excluded"], prep["t_onset"]

        t_arr, X_arr = kept["t_fit"].values, kept["X"].values
        kin_fig = go.Figure()
        if len(excluded):
            kin_fig.add_trace(go.Scatter(x=excluded["t_rel"] - t_onset, y=excluded["X"], mode="markers",
                                          marker=dict(color="#B8BDB9", size=5), name="excluded (dead time)"))
        kin_fig.add_trace(go.Scatter(x=t_arr, y=X_arr, mode="markers",
                                      marker=dict(color="#5B665F", opacity=0.6), name="data used in fit"))
        kin_fig.add_vline(x=0, line_dash="dot", line_color="#5B665F")
        t_curve = np.linspace(t_arr.min(), t_arr.max(), 60) if len(t_arr) else np.array([])

        st.caption(f"Onset detected at t = {t_onset:.3f} (selection-relative) \u2014 {len(excluded)} point(s) excluded as dead time.")
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
                    results_list.append((name, fit["r2"], fit["sse"], fit["aic"], fit["bic"], params_str(fit["params"])))
                else:
                    results_list.append((name, None, None, None, None, "fit failed"))
            st.plotly_chart(kin_fig, use_container_width=True)
            results_list.sort(key=lambda r: (r[3] is None, r[3] if r[3] is not None else 0))
            comp_df = pd.DataFrame([
                {"Model": n, "R\u00b2": (f"{r2:.4f}" if r2 is not None else "\u2014"),
                 "SSE": (f"{sse:.4g}" if sse is not None else "\u2014"),
                 "AIC": (f"{aic:.2f}" if aic is not None else "\u2014"),
                 "BIC": (f"{bic:.2f}" if bic is not None else "\u2014"),
                 "Parameters": p}
                for n, r2, sse, aic, bic, p in results_list
            ])
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            best = results_list[0]
            if best[3] is not None:
                st.caption(f"Lowest AIC (best-fitting model for its complexity): **{best[0]}** (AIC = {best[3]:.2f}, R\u00b2 = {best[1]:.4f}).")
        elif not row["two_stage"]:
            fit = fit_model(row["model"], t_arr, X_arr) if len(t_arr) >= 3 else None
            if fit:
                d = MODEL_DEFS[row["model"]]
                y_curve = np.clip([d["invert"](fit["params"], t) for t in t_curve], 0, 1)
                kin_fig.add_trace(go.Scatter(x=t_curve, y=y_curve, mode="lines",
                                              line=dict(color="#2F6F5E", width=2), name="fit"))
                st.plotly_chart(kin_fig, use_container_width=True)
                display_params = {k: v for k, v in fit["params"].items() if not k.startswith("_")}
                metric_cols = st.columns(len(display_params) + 4)
                for i, (key, val) in enumerate(display_params.items()):
                    metric_cols[i].metric(PARAM_LABELS.get(key, key), f"{val:.4g}")
                base = len(display_params)
                metric_cols[base].metric("R\u00b2", f"{fit['r2']:.4f}")
                metric_cols[base + 1].metric("SSE", f"{fit['sse']:.4g}")
                metric_cols[base + 2].metric("AIC", f"{fit['aic']:.2f}")
                metric_cols[base + 3].metric("BIC", f"{fit['bic']:.2f}")
                st.caption(f"*{d['eq']}*")
            else:
                st.warning("Couldn't fit this model to this cycle \u2014 try a simpler model or a wider selection.")
        else:
            auto_split = st.checkbox("Auto-find best split (minimize total SSE)", key=f"autosplit_{row.name}_{row['label']}", value=True)
            if auto_split:
                best = find_best_split(t_arr, X_arr)
                if best is None:
                    st.warning("Couldn't find a valid split point with enough points on both sides \u2014 falling back to the manual Transition X.")
                    tx = row["transition_X"]
                else:
                    tx = best["tx"]
                    st.caption(f"Auto-detected transition at X = {tx:.2f} (lowest combined SSE across both stages).")
            else:
                tx = row["transition_X"]

            mask1, mask2 = X_arr <= tx, X_arr > tx
            if mask1.sum() >= 3 and mask2.sum() >= 3:
                f1 = fit_stage1_fast(t_arr[mask1], X_arr[mask1])
                f2 = fit_stage2_diffusion(t_arr[mask2], X_arr[mask2])
                t_split = t_arr[mask2].min()
                c1, c2 = t_curve[t_curve <= t_split], t_curve[t_curve > t_split]
                if f1 and len(c1):
                    y1 = np.clip([f1["invert"](t) for t in c1], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c1, y=y1, mode="lines", line=dict(color="#2F6F5E", width=2), name="fast stage"))
                if f2 and len(c2):
                    y2 = np.clip([f2["invert"](t) for t in c2], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c2, y=y2, mode="lines", line=dict(color="#B5482F", width=2), name="diffusion stage"))
                kin_fig.add_vline(x=t_split, line_dash="dot", line_color="#B5482F")
                st.plotly_chart(kin_fig, use_container_width=True)

                stage_table = pd.DataFrame([
                    {"Stage": "Fast (reaction control)",
                     "k": f"{f1['params']['k']:.4g}" if f1 else "\u2014",
                     "R\u00b2": f"{f1['r2']:.4f}" if f1 else "\u2014",
                     "SSE": f"{f1['sse']:.4g}" if f1 else "\u2014",
                     "AIC": f"{f1['aic']:.2f}" if f1 else "\u2014",
                     "BIC": f"{f1['bic']:.2f}" if f1 else "\u2014"},
                    {"Stage": "Diffusion (product-layer control)",
                     "k": f"{f2['params']['k']:.4g}" if f2 else "\u2014",
                     "R\u00b2": f"{f2['r2']:.4f}" if f2 else "\u2014",
                     "SSE": f"{f2['sse']:.4g}" if f2 else "\u2014",
                     "AIC": f"{f2['aic']:.2f}" if f2 else "\u2014",
                     "BIC": f"{f2['bic']:.2f}" if f2 else "\u2014"},
                ])
                st.dataframe(stage_table, use_container_width=True, hide_index=True)
                if f1 and f2:
                    total_sse = f1["sse"] + f2["sse"]
                    total_n = f1["n"] + f2["n"]
                    total_k = f1["k_params"] + f2["k_params"]
                    total_aic = total_n * np.log(max(total_sse, 1e-12) / total_n) + 2 * total_k
                    total_bic = total_n * np.log(max(total_sse, 1e-12) / total_n) + total_k * np.log(total_n)
                    tc1, tc2, tc3 = st.columns(3)
                    tc1.metric("Total SSE", f"{total_sse:.4g}")
                    tc2.metric("Total AIC", f"{total_aic:.2f}")
                    tc3.metric("Total BIC", f"{total_bic:.2f}")
                st.caption(
                    f"Fast stage: *-ln(1-X) = k\u00b7t* (forced through the true reaction origin). "
                    f"Diffusion stage: *1-(2/3)X-(1-X)^(2/3) = k\u00b7t + c* (own local intercept, since this "
                    f"segment doesn't start at X=0), split at X = {tx:.2f}."
                )
            else:
                st.warning("Not enough points on one side of the transition to fit both stages.")

# ================================================================== STEP 7: compare cycles
st.header("7. Compare cycles")
st.caption("Pick two or more cycles to overlay their conversion curves and fitted kinetics side by side.")

cycle_labels_all = list(valid["label"])
picked = st.multiselect("Cycles to compare", cycle_labels_all, default=cycle_labels_all[: min(2, len(cycle_labels_all))])
compare_model = st.selectbox("Model to fit for comparison", MODEL_NAMES, key="compare_cycles_model")

if len(picked) < 2:
    st.info("Pick at least two cycles above to compare them.")
else:
    palette = ["#2F6F5E", "#B5482F", "#4A6FA5", "#B08A2E", "#7A4E9E", "#3C8C8C", "#8E6C3A", "#6B7B4F"]
    cmp_fig = go.Figure()
    cmp_rows = []
    for i, label in enumerate(picked):
        row = valid[valid["label"] == label].iloc[0]
        prep = prepare_kinetics(df, row, onset_pct)
        color = palette[i % len(palette)]
        if prep is None:
            cmp_rows.append({"Cycle": label, "R\u00b2": "\u2014", "SSE": "\u2014", "AIC": "\u2014", "BIC": "\u2014", "Parameters": "not enough data"})
            continue
        kept = prep["kept"]
        t_arr, X_arr = kept["t_fit"].values, kept["X"].values
        cmp_fig.add_trace(go.Scatter(x=t_arr, y=X_arr, mode="markers",
                                      marker=dict(color=color, opacity=0.5), name=f"{label} data", showlegend=False))
        fit = fit_model(compare_model, t_arr, X_arr) if len(t_arr) >= 3 else None
        if fit:
            d = MODEL_DEFS[compare_model]
            t_curve = np.linspace(t_arr.min(), t_arr.max(), 60)
            y_curve = np.clip([d["invert"](fit["params"], t) for t in t_curve], 0, 1)
            cmp_fig.add_trace(go.Scatter(x=t_curve, y=y_curve, mode="lines",
                                          line=dict(color=color, width=2), name=label))
            cmp_rows.append({"Cycle": label, "R\u00b2": f"{fit['r2']:.4f}", "SSE": f"{fit['sse']:.4g}",
                              "AIC": f"{fit['aic']:.2f}", "BIC": f"{fit['bic']:.2f}", "Parameters": params_str(fit["params"])})
        else:
            cmp_rows.append({"Cycle": label, "R\u00b2": "\u2014", "SSE": "\u2014", "AIC": "\u2014", "BIC": "\u2014", "Parameters": "fit failed"})

    cmp_fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="time since reaction onset", yaxis_title="X",
                           yaxis_range=[0, 1], plot_bgcolor="white")
    st.plotly_chart(cmp_fig, use_container_width=True)
    st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)
    st.caption(f"All cycles fit with: *{MODEL_DEFS[compare_model]['eq']}*")

# ================================================================== STEP 8: Arrhenius analysis
st.header("8. Arrhenius analysis")
st.caption(
    "Only meaningful if different cycles were carbonated at different temperatures (fill in 'Carb. temp' in the "
    "cycle table above). Uses each cycle's single-model fit rate constant k \u2014 so this only works for models "
    "with a plain k (first-order, Avrami, both shrinking-core forms, RPM, nth-order); DEM and CGSM are skipped "
    "since they don't have one overall k. Two-stage cycles are also skipped."
)

temp_unit = st.radio("Temperature unit entered above", ["\u00b0C", "K"], horizontal=True)

arr_points = []
for _, row in valid.iterrows():
    if row.get("two_stage") or pd.isna(row.get("temperature")):
        continue
    if row["model"] in ("Double Exponential Model (DEM)", "Changing Grain Size Model (CGSM)"):
        continue
    prep = prepare_kinetics(df, row, onset_pct)
    if prep is None:
        continue
    kept = prep["kept"]
    t_arr, X_arr = kept["t_fit"].values, kept["X"].values
    if len(t_arr) < 3:
        continue
    fit = fit_model(row["model"], t_arr, X_arr)
    if fit is None or "k" not in fit["params"]:
        continue
    T_K = row["temperature"] + 273.15 if temp_unit == "\u00b0C" else row["temperature"]
    arr_points.append({"label": row["label"], "T_K": T_K, "k": fit["params"]["k"]})

if len(arr_points) < 2:
    st.info("Need at least two cycles with a temperature entered and a compatible model fit to run this.")
else:
    invT = np.array([1 / p["T_K"] for p in arr_points])
    lnk = np.array([np.log(p["k"]) for p in arr_points])
    arr_fit = linreg(invT, lnk)
    if arr_fit is None:
        st.warning("Couldn't fit an Arrhenius line to these points.")
    else:
        Ea = -arr_fit["slope"] * R_GAS / 1000  # kJ/mol
        A = np.exp(arr_fit["intercept"])
        arr_fig = go.Figure()
        arr_fig.add_trace(go.Scatter(x=invT, y=lnk, mode="markers+text",
                                      text=[p["label"] for p in arr_points], textposition="top center",
                                      marker=dict(color="#2F6F5E", size=8)))
        x_line = np.linspace(invT.min(), invT.max(), 50)
        y_line = arr_fit["slope"] * x_line + arr_fit["intercept"]
        arr_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="#B5482F", width=2)))
        arr_fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                               xaxis_title="1/T (1/K)", yaxis_title="ln(k)", plot_bgcolor="white", showlegend=False)
        st.plotly_chart(arr_fig, use_container_width=True)
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Ea (kJ/mol)", f"{Ea:.3g}")
        mc2.metric("A (pre-exponential)", f"{A:.4g}")
        mc3.metric("R\u00b2", f"{arr_fit['r2']:.4f}")
        st.caption("*ln(k) = ln(A) \u2212 Ea/(R\u00b7T)*")

# ================================================================== STEP 9: kinetics summary export
st.header("9. Export kinetics summary")

summary_rows = []
for _, row in valid.iterrows():
    prep = prepare_kinetics(df, row, onset_pct)
    if prep is None:
        continue
    kept = prep["kept"]
    t_arr, X_arr = kept["t_fit"].values, kept["X"].values
    if not row["two_stage"]:
        fit = fit_model(row["model"], t_arr, X_arr) if len(t_arr) >= 3 else None
        rec = {"Cycle": row["label"], "Model": row["model"], "Temperature": row.get("temperature")}
        if fit:
            for k, v in fit["params"].items():
                if not k.startswith("_"):
                    rec[PARAM_LABELS.get(k, k)] = v
            rec.update({"R2": fit["r2"], "SSE": fit["sse"], "AIC": fit["aic"], "BIC": fit["bic"]})
        summary_rows.append(rec)
    else:
        auto_split = st.session_state.get(f"autosplit_{row.name}_{row['label']}", True)
        if auto_split:
            best = find_best_split(t_arr, X_arr)
            tx = best["tx"] if best else row["transition_X"]
        else:
            tx = row["transition_X"]
        mask1, mask2 = X_arr <= tx, X_arr > tx
        rec = {"Cycle": row["label"], "Model": "Two-stage (first-order + diffusion)", "Temperature": row.get("temperature"), "Transition X": tx}
        if mask1.sum() >= 3 and mask2.sum() >= 3:
            f1 = fit_stage1_fast(t_arr[mask1], X_arr[mask1])
            f2 = fit_stage2_diffusion(t_arr[mask2], X_arr[mask2])
            if f1:
                rec.update({"k1": f1["params"]["k"], "R2_fast": f1["r2"], "AIC_fast": f1["aic"], "BIC_fast": f1["bic"], "SSE_fast": f1["sse"]})
            if f2:
                rec.update({"k2": f2["params"]["k"], "c2": f2["params"]["c"], "R2_diff": f2["r2"], "AIC_diff": f2["aic"], "BIC_diff": f2["bic"], "SSE_diff": f2["sse"]})
        summary_rows.append(rec)

summary_df = pd.DataFrame(summary_rows)
if not summary_df.empty:
    st.dataframe(summary_df.round(5), use_container_width=True, hide_index=True)
    df_download_buttons(summary_df, "kinetics_summary", "kin")
else:
    st.info("No fittable cycles yet.")

# ================================================================== STEP 10: publication-ready figures
st.header("10. Publication-ready figures")
st.caption("Static, journal-styled versions of the key plots, downloadable as 300 dpi PNG or vector PDF.")

st.subheader("TGA curve with cycle windows")
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(df["t"], df["w"], color="black", lw=1)
for _, row in valid.iterrows():
    ax1.axvspan(row["t_start"], row["t_end"], color="#2F6F5E", alpha=0.15)
ax1.set_xlabel("Time")
ax1.set_ylabel("Weight")
if "temp" in df.columns:
    ax1b = ax1.twinx()
    ax1b.plot(df["t"], df["temp"], color="#B5482F", lw=0.8, ls="--")
    ax1b.set_ylabel("Temperature")
fig1.tight_layout()
st.pyplot(fig1)
fig_download_buttons(fig1, "tga_curve", "fig1")

if len(cap_table) > 1:
    st.subheader("Capacity vs. cycle number")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    N_arr2 = np.arange(1, len(cap_table) + 1)
    ax2.plot(N_arr2, cap_table["mmol CO2 / g"].values, "o", color="#2F6F5E", label="observed")
    if decay_fit:
        N_curve2 = np.linspace(1, len(cap_table), 100)
        cap_curve2 = _decay_curve(N_curve2, decay_fit["cap0"], decay_fit["capr"], decay_fit["kd"])
        ax2.plot(N_curve2, cap_curve2, "-", color="#B5482F", label="Grasa-Abanades fit")
        ax2.legend(frameon=False, fontsize=8)
    ax2.set_xlabel("Cycle number")
    ax2.set_ylabel("mmol CO$_2$ g$^{-1}$")
    ax2.set_xticks(N_arr2)
    fig2.tight_layout()
    st.pyplot(fig2)
    fig_download_buttons(fig2, "capacity_vs_cycle", "fig2")

st.subheader("Kinetics fit for one cycle")
pub_cycle = st.selectbox("Cycle", cycle_labels_all, key="pub_cycle")
pub_row = valid[valid["label"] == pub_cycle].iloc[0]
pub_model = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(pub_row["model"]), key="pub_model")
pub_prep = prepare_kinetics(df, pub_row, onset_pct)
if pub_prep:
    kept = pub_prep["kept"]
    t_arr3, X_arr3 = kept["t_fit"].values, kept["X"].values
    fit3 = fit_model(pub_model, t_arr3, X_arr3) if len(t_arr3) >= 3 else None
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    ax3.scatter(t_arr3, X_arr3, color="#5B665F", s=14, alpha=0.6, label="data")
    if fit3:
        d3 = MODEL_DEFS[pub_model]
        t_curve3 = np.linspace(t_arr3.min(), t_arr3.max(), 100)
        y_curve3 = np.clip([d3["invert"](fit3["params"], t) for t in t_curve3], 0, 1)
        ax3.plot(t_curve3, y_curve3, color="#2F6F5E", lw=1.5, label=f"fit (R\u00b2={fit3['r2']:.3f})")
        ax3.legend(frameon=False, fontsize=8)
    ax3.set_xlabel("Time since reaction onset")
    ax3.set_ylabel("Conversion, X")
    ax3.set_ylim(0, 1)
    fig3.tight_layout()
    st.pyplot(fig3)
    fig_download_buttons(fig3, f"kinetics_{pub_cycle}", "fig3")
else:
    st.info("Not enough data in this cycle to plot.")
