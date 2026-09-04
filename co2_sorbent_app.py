import io
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

st.set_page_config(page_title="CaO TGA Calcium-Looping Analyzer", page_icon="🧪", layout="wide")

R_GAS = 8.314462618
MW_CO2 = 44.01  # g/mol


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def guess_column(columns, candidates):
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    for c in columns:
        s = re.sub(r"[^a-z0-9]", "", c.lower())
        if any(re.sub(r"[^a-z0-9]", "", cand.lower()) in s for cand in candidates):
            return c
    return columns[0] if columns else None


def alpha_from_weight(t, w, start_idx, end_idx):
    w0 = float(w[start_idx])
    w1 = float(w[end_idx])
    dw = w1 - w0
    if abs(dw) < 1e-15:
        return np.zeros(end_idx - start_idx + 1), 0.0
    ww = w[start_idx:end_idx + 1]
    # For a selected carbonation segment, conversion is defined from the cycle start.
    # A negative gain is retained; users can reverse endpoints if they selected the wrong direction.
    alpha = (ww - w0) / dw
    return np.asarray(alpha, dtype=float), dw


def prepare_segment(df, i0, i1):
    if i1 <= i0:
        raise ValueError("End point must be after start point.")
    seg = df.iloc[i0:i1 + 1].copy().reset_index(drop=True)
    t = pd.to_numeric(seg["__time"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(seg["__weight"], errors="coerce").to_numpy(dtype=float)
    temp = pd.to_numeric(seg["__temperature"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t) & np.isfinite(w) & np.isfinite(temp)
    t, w, temp = t[good], w[good], temp[good]
    if len(t) < 8:
        raise ValueError("A cycle needs at least 8 valid data points.")
    order = np.argsort(t)
    t, w, temp = t[order], w[order], temp[order]
    t = t - t[0]
    # Average duplicate time values to keep fitting well behaved.
    tmp = pd.DataFrame({"t": t, "w": w, "T": temp}).groupby("t", as_index=False).mean()
    t, w, temp = tmp.t.to_numpy(), tmp.w.to_numpy(), tmp.T.to_numpy()
    alpha, dw = alpha_from_weight(t, w, 0, len(t) - 1)
    return t, w, temp, alpha, dw


def clip_alpha(t, alpha):
    # Kinetic models describe 0 <= alpha < 1. Preserve endpoints for plotting,
    # but use a lightly clipped version for numerical fitting.
    a = np.clip(alpha, 0.0, 1.0)
    mask = np.isfinite(t) & np.isfinite(a)
    t, a = t[mask], a[mask]
    return t, a


def model_first(t, k):
    return 1 - np.exp(-k * np.maximum(t, 0))


def model_avrami(t, k, n):
    return 1 - np.exp(-np.maximum(k * np.maximum(t, 0), 0) ** n)


def model_scm_surface(t, k):
    z = np.clip(k * np.maximum(t, 0), 0, 1)
    return 1 - (1 - z) ** 3


def model_scm_diffusion(t, k):
    z = np.maximum(k * np.maximum(t, 0), 0)
    # 1 - 3(1-X)^(2/3) + 2(1-X) = kt. Solve for X numerically pointwise.
    # This function uses the standard integrated SCM relation by inverting it via bracketing.
    out = np.zeros_like(z, dtype=float)
    for j, zz in enumerate(z):
        if zz >= 1:
            out[j] = 1.0
            continue
        lo, hi = 0.0, 1.0 - 1e-12
        for _ in range(45):
            mid = (lo + hi) / 2
            val = 1 - 3 * (1 - mid) ** (2 / 3) + 2 * (1 - mid)
            if val < zz:
                lo = mid
            else:
                hi = mid
        out[j] = (lo + hi) / 2
    return out


def model_nth(t, k, n):
    tt = np.maximum(t, 0)
    if abs(n - 1.0) < 1e-6:
        return model_first(tt, k)
    base = 1 + (n - 1) * k * tt
    base = np.maximum(base, 1e-12)
    return 1 - base ** (-1 / (n - 1))


def model_double_exp(t, k1, k2, f):
    tt = np.maximum(t, 0)
    f = np.clip(f, 0, 1)
    return f * (1 - np.exp(-k1 * tt)) + (1 - f) * (1 - np.exp(-k2 * tt))


def rpm_rhs(t, y, k, psi):
    a = np.clip(y[0], 0, 1 - 1e-12)
    rad = max(1 - psi * np.log(max(1 - a, 1e-12)), 1e-12)
    return [k * (1 - a) * np.sqrt(rad)]


def model_rpm(t, k, psi):
    tt = np.asarray(t, dtype=float)
    if len(tt) == 0:
        return np.array([])
    order = np.argsort(tt)
    ts = np.maximum(tt[order], 0)
    sol = solve_ivp(lambda x, y: rpm_rhs(x, y, k, psi), (0, float(ts[-1]) + 1e-12), [0.0], t_eval=ts,
                    rtol=2e-6, atol=1e-8, max_step=max(float(ts[-1]) / 80, 1e-4))
    vals = np.interp(ts, sol.t, sol.y[0]) if sol.success else np.zeros_like(ts)
    out = np.empty_like(vals)
    out[order] = vals
    return out


MODELS = {
    "First-order": (model_first, [0.01], ([1e-9], [10.0]), ["k"]),
    "Avrami-Erofeev": (model_avrami, [0.01, 1.0], ([1e-9, 0.1], [10.0, 8.0]), ["k", "n"]),
    "Shrinking-core (surface)": (model_scm_surface, [0.01], ([1e-9], [10.0]), ["k"]),
    "Shrinking-core (diffusion)": (model_scm_diffusion, [0.01], ([1e-9], [10.0]), ["k"]),
    "Random pore": (model_rpm, [0.01, 1.0], ([1e-9, 0.001], [10.0, 20.0]), ["k", "psi"]),
    "Double exponential": (model_double_exp, [0.01, 0.001, 0.5], ([1e-9, 1e-9, 0.0], [10.0, 10.0, 1.0]), ["k1", "k2", "f_fast"]),
    "Grain model": (model_scm_surface, [0.01], ([1e-9], [10.0]), ["k"]),
    "nth-order": (model_nth, [0.01, 1.5], ([1e-9, 0.05], [10.0, 8.0]), ["k", "n"]),
}


def metrics(y, yhat, p):
    resid = y - yhat
    rss = float(np.sum(resid ** 2))
    n = len(y)
    if n == 0:
        return np.nan, np.nan, np.nan
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - rss / tss if tss > 1e-15 else np.nan
    sigma2 = max(rss / n, 1e-300)
    aic = n * np.log(sigma2) + 2 * p
    bic = n * np.log(sigma2) + p * np.log(n)
    return r2, aic, bic


def fit_one(model_name, t, alpha, two_stage=False):
    fn, x0, bounds, names = MODELS[model_name]
    t, a = clip_alpha(t, alpha)
    if len(t) < 6:
        return None
    # Remove exact t=0 duplicate noise and keep monotonic time.
    keep = np.r_[True, np.diff(t) > 0]
    t, a = t[keep], a[keep]

    def single_fit(tx, ax):
        try:
            res = least_squares(lambda p: fn(tx, *p) - ax, x0=np.asarray(x0, float), bounds=bounds,
                                max_nfev=3000, loss="soft_l1")
            pred = fn(tx, *res.x)
            return res, pred
        except Exception:
            return None, None

    if not two_stage:
        res, pred = single_fit(t, a)
        if res is None:
            return None
        r2, aic, bic = metrics(a, pred, len(res.x))
        return {"model": model_name, "r2": r2, "aic": aic, "bic": bic,
                "transition_min": np.nan, "transition_max": np.nan,
                "params": dict(zip(names, res.x)), "prediction": pred,
                "success": res.success}

    # Independent transition optimization for this model.
    # Each model gets its own transition candidate. The objective is the combined
    # least-squares error of two independently optimized parameter sets.
    candidates = np.unique(np.quantile(t[1:-1], np.linspace(0.10, 0.90, min(31, max(5, len(t) // 4)))))
    best = None
    for tr in candidates:
        m1 = t <= tr
        m2 = t > tr
        if m1.sum() < max(5, len(x0) + 2) or m2.sum() < max(5, len(x0) + 2):
            continue
        r1, p1 = single_fit(t[m1], a[m1])
        r2_, p2 = single_fit(t[m2] - tr, a[m2] - a[m1][-1])
        if r1 is None or r2_ is None:
            continue
        # Second stage starts at zero incremental conversion.
        pred2 = a[m1][-1] + p2
        pred = np.empty_like(a)
        pred[m1] = p1
        pred[m2] = pred2
        rss = float(np.sum((a - pred) ** 2))
        if best is None or rss < best[0]:
            best = (rss, tr, r1, r2_, pred, p1, p2, m1, m2)
    if best is None:
        return None
    rss, tr, r1, r2_, pred, p1, p2, m1, m2 = best
    # Parameter count includes both stages plus the transition point.
    pcount = len(r1.x) + len(r2_.x) + 1
    r2v, aic, bic = metrics(a, pred, pcount)
    params = {}
    for n, v in zip(names, r1.x):
        params[f"stage1_{n}"] = v
    for n, v in zip(names, r2_.x):
        params[f"stage2_{n}"] = v
    params["transition"] = tr
    return {"model": model_name, "r2": r2v, "aic": aic, "bic": bic,
            "transition_min": tr, "transition_max": tr, "params": params,
            "prediction": pred, "success": r1.success and r2_.success}


def result_table(fits):
    rows = []
    for r in fits:
        row = {"Model": r["model"], "R²": r["r2"], "AIC": r["aic"], "BIC": r["bic"]}
        if np.isfinite(r.get("transition_min", np.nan)):
            row["Optimized transition (s)"] = r["transition_min"]
        for k, v in r["params"].items():
            row[k] = v
        rows.append(row)
    return pd.DataFrame(rows).sort_values("BIC", na_position="last")


def export_excel(cycles_df, fits_by_cycle):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        cycles_df.to_excel(writer, sheet_name="Cycle summary", index=False)
        for cyc, fits in fits_by_cycle.items():
            result_table(fits).to_excel(writer, sheet_name=f"Cycle {cyc}"[:31], index=False)
    buf.seek(0)
    return buf


def make_publication_figure(df, selected_cycles, fits_by_cycle, show_fits):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["__time"], y=df["__weight"], mode="lines", name="TGA weight", line=dict(width=2)))
    for cyc in selected_cycles:
        i0, i1 = cyc["start_idx"], cyc["end_idx"]
        fig.add_vrect(x0=df["__time"].iloc[i0], x1=df["__time"].iloc[i1], fillcolor="LightGray", opacity=0.25,
                      line_width=1, annotation_text=f"Cycle {cyc['cycle']}", annotation_position="top left")
    fig.update_layout(template="simple_white", xaxis_title="Time", yaxis_title="Weight",
                      font=dict(family="Arial", size=14), width=1100, height=650,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


st.title("🧪 CaO Calcium-Looping TGA Analyzer")
st.caption("Cycle selection → CO₂ capture capacity → kinetic model fitting → cycle comparison → publication-ready export")

uploaded = st.file_uploader("Upload TGA CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV containing time, weight, and temperature columns. The app will try to identify the columns automatically.")
    st.markdown("**Expected data:** one row per TGA measurement, with time, weight, and temperature. Column names can be customized after upload.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

raw = clean_columns(raw)
cols = list(raw.columns)
if len(cols) < 3:
    st.error("The CSV needs at least three columns for time, weight, and temperature.")
    st.stop()

def_time = guess_column(cols, ["time", "timestamp", "seconds", "sec", "t"])
def_weight = guess_column(cols, ["weight", "mass", "wt", "sampleweight"])
def_temp = guess_column(cols, ["temperature", "temp", "T", "degC", "celsius"])

with st.sidebar:
    st.header("1. Data setup")
    time_col = st.selectbox("Time column", cols, index=cols.index(def_time))
    weight_col = st.selectbox("Weight column", cols, index=cols.index(def_weight))
    temp_col = st.selectbox("Temperature column", cols, index=cols.index(def_temp))
    time_factor = st.number_input("Time conversion factor", value=1.0, min_value=1e-12, format="%.8g",
                                  help="Multiply the selected time column by this factor. Example: 60 if time is in minutes and you want seconds.")
    weight_unit = st.selectbox("Weight unit in CSV", ["mg", "g", "µg"], index=0)
    sample_basis = st.selectbox("CO₂ capacity mass basis", ["Cycle-start mass", "Fixed initial sample mass"])
    if sample_basis == "Fixed initial sample mass":
        fixed_mass = st.number_input("Initial sample mass (mg)", min_value=0.000001, value=300.0, format="%.6f")
    else:
        fixed_mass = None
    two_stage = st.checkbox("Two-stage kinetic fitting", value=True,
                            help="Each kinetic model independently optimizes its own transition point. Transition points are not shared between models.")
    selected_models = st.multiselect("Kinetic models", list(MODELS.keys()), default=list(MODELS.keys()))

work = pd.DataFrame({
    "__time": pd.to_numeric(raw[time_col], errors="coerce") * time_factor,
    "__weight": pd.to_numeric(raw[weight_col], errors="coerce"),
    "__temperature": pd.to_numeric(raw[temp_col], errors="coerce"),
}).dropna().reset_index(drop=True)

st.subheader("Data overview")
c1, c2, c3 = st.columns(3)
c1.metric("Data points", f"{len(work):,}")
c2.metric("Time span", f"{work['__time'].iloc[-1] - work['__time'].iloc[0]:,.2f}")
c3.metric("Weight range", f"{work['__weight'].min():.4g} – {work['__weight'].max():.4g}")

fig_data = go.Figure()
fig_data.add_trace(go.Scatter(x=work["__time"], y=work["__weight"], mode="lines", name="Weight", line=dict(width=2)))
fig_data.update_layout(height=520, dragmode="select", xaxis_title="Time", yaxis_title=f"Weight ({weight_unit})",
                       margin=dict(l=40, r=20, t=35, b=45), legend=dict(orientation="h"))

st.subheader("2. Drag-select carbonation cycles")
st.write("Drag a box over the start/end region of one carbonation cycle. Then add that selection to the cycle list. You can also use the index fields as a precise fallback.")
event = st.plotly_chart(fig_data, use_container_width=True, key="cycle_chart", on_select="rerun", selection_mode="box")

sel_start_idx = None
sel_end_idx = None
try:
    pts = event.selection.points if event and event.selection else []
    if pts:
        xs = [float(p["x"]) for p in pts if "x" in p]
        if xs:
            lo, hi = min(xs), max(xs)
            sel_start_idx = int(np.argmin(np.abs(work["__time"].to_numpy() - lo)))
            sel_end_idx = int(np.argmin(np.abs(work["__time"].to_numpy() - hi)))
except Exception:
    pass

if "cycles" not in st.session_state:
    st.session_state.cycles = []

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    start_idx = st.number_input("Start row", min_value=0, max_value=max(0, len(work)-2),
                                value=int(sel_start_idx if sel_start_idx is not None else 0), step=1)
with colB:
    end_idx = st.number_input("End row", min_value=1, max_value=max(1, len(work)-1),
                              value=int(sel_end_idx if sel_end_idx is not None else min(len(work)-1, max(1, len(work)//10))), step=1)
with colC:
    if st.button("➕ Add selected cycle", type="primary"):
        if end_idx <= start_idx:
            st.error("End row must be greater than start row.")
        else:
            cycle_no = len(st.session_state.cycles) + 1
            st.session_state.cycles.append({"cycle": cycle_no, "start_idx": int(start_idx), "end_idx": int(end_idx)})
            st.rerun()

if st.session_state.cycles:
    cycle_rows = []
    for c in st.session_state.cycles:
        i0, i1 = c["start_idx"], c["end_idx"]
        cycle_rows.append({"Cycle": c["cycle"], "Start index": i0, "End index": i1,
                           "Start time": work["__time"].iloc[i0], "End time": work["__time"].iloc[i1],
                           "Duration": work["__time"].iloc[i1] - work["__time"].iloc[i0]})
    st.dataframe(pd.DataFrame(cycle_rows), use_container_width=True, hide_index=True)
    if st.button("🗑️ Clear all cycles"):
        st.session_state.cycles = []
        st.rerun()
else:
    st.warning("No cycles selected yet.")

if not st.session_state.cycles:
    st.stop()

st.subheader("3. Analyze selected cycles")
run = st.button("▶ Run analysis", type="primary", disabled=not selected_models)

if run:
    all_summary = []
    fits_by_cycle = {}
    curves_by_cycle = {}
    progress = st.progress(0)
    status = st.empty()
    for n, c in enumerate(st.session_state.cycles, start=1):
        status.write(f"Analyzing cycle {c['cycle']} …")
        try:
            t, w, temp, alpha, dw = prepare_segment(work, c["start_idx"], c["end_idx"])
            if sample_basis == "Fixed initial sample mass":
                mass_g = fixed_mass / 1000.0
            else:
                raw_mass = float(w[0])
                mass_g = raw_mass / (1000.0 if weight_unit == "mg" else 1.0 if weight_unit == "g" else 1e6)
            capacity = (dw / (1000.0 if weight_unit == "mg" else 1.0 if weight_unit == "g" else 1e6)) / MW_CO2 * 1000.0 / mass_g
            fits = []
            for model in selected_models:
                r = fit_one(model, t, alpha, two_stage=two_stage)
                if r is not None:
                    fits.append(r)
            fits_by_cycle[c["cycle"]] = fits
            curves_by_cycle[c["cycle"]] = (t, alpha, temp)
            best = min(fits, key=lambda x: x["bic"]) if fits else None
            all_summary.append({"Cycle": c["cycle"], "Start time": work["__time"].iloc[c["start_idx"]],
                                "End time": work["__time"].iloc[c["end_idx"]], "Duration": work["__time"].iloc[c["end_idx"]] - work["__time"].iloc[c["start_idx"]],
                                "Weight change": dw, "CO₂ capture (mmol/g)": capacity,
                                "Best model by BIC": best["model"] if best else "—",
                                "Best BIC": best["bic"] if best else np.nan})
        except Exception as e:
            st.error(f"Cycle {c['cycle']} failed: {e}")
        progress.progress(n / len(st.session_state.cycles))
    status.empty()

    summary_df = pd.DataFrame(all_summary)
    st.session_state.analysis = {"summary": summary_df, "fits": fits_by_cycle, "curves": curves_by_cycle}

if "analysis" not in st.session_state:
    st.info("Click **Run analysis** to fit the selected cycles.")
    st.stop()

summary_df = st.session_state.analysis["summary"]
fits_by_cycle = st.session_state.analysis["fits"]
curves_by_cycle = st.session_state.analysis["curves"]

st.subheader("4. CO₂ capture capacity by cycle")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

cap_fig = go.Figure(go.Scatter(x=summary_df["Cycle"], y=summary_df["CO₂ capture (mmol/g)"], mode="lines+markers", marker=dict(size=9)))
cap_fig.update_layout(template="simple_white", xaxis_title="Cycle", yaxis_title="CO₂ capture capacity (mmol CO₂/g)",
                     font=dict(family="Arial", size=14), height=450)
st.plotly_chart(cap_fig, use_container_width=True)

st.subheader("5. Kinetic model comparison")
cycle_options = [int(x) for x in summary_df["Cycle"].tolist()]
chosen_cycle = st.selectbox("Cycle to inspect", cycle_options)
fits = fits_by_cycle.get(chosen_cycle, [])
if not fits:
    st.warning("No successful model fits for this cycle.")
else:
    table = result_table(fits)
    st.dataframe(table, use_container_width=True, hide_index=True)
    best = min(fits, key=lambda x: x["bic"])
    st.success(f"Best model by BIC for Cycle {chosen_cycle}: **{best['model']}**")
    t, alpha, temp = curves_by_cycle[chosen_cycle]
    fit_fig = go.Figure()
    fit_fig.add_trace(go.Scatter(x=t, y=alpha, mode="markers", name="TGA conversion", marker=dict(size=5)))
    for r in fits:
        fit_fig.add_trace(go.Scatter(x=t, y=r["prediction"], mode="lines", name=r["model"], line=dict(width=2)))
        if two_stage and np.isfinite(r.get("transition_min", np.nan)):
            fit_fig.add_vline(x=r["transition_min"], line_dash="dash", annotation_text=f"{r['model']} transition", annotation_position="top")
    fit_fig.update_layout(template="simple_white", xaxis_title="Time from cycle start (s)", yaxis_title="Normalized carbonation conversion, α",
                          font=dict(family="Arial", size=13), height=650, legend=dict(orientation="h", y=-0.18))
    st.plotly_chart(fit_fig, use_container_width=True)

st.subheader("6. Compare cycles")
compare_fig = go.Figure()
for cyc, (t, alpha, _) in curves_by_cycle.items():
    compare_fig.add_trace(go.Scatter(x=t, y=alpha, mode="lines", name=f"Cycle {cyc}", line=dict(width=2)))
compare_fig.update_layout(template="simple_white", xaxis_title="Time from cycle start (s)", yaxis_title="Normalized conversion, α",
                          font=dict(family="Arial", size=14), height=550, legend=dict(orientation="h", y=-0.18))
st.plotly_chart(compare_fig, use_container_width=True)

st.subheader("7. Publication-quality exports")
export_fig = make_publication_figure(work, st.session_state.cycles, fits_by_cycle, two_stage)
img_png = export_fig.to_image(format="png", width=1800, height=1050, scale=2)
img_svg = export_fig.to_image(format="svg", width=1800, height=1050)
excel_buf = export_excel(summary_df, fits_by_cycle)

c1, c2, c3, c4 = st.columns(4)
c1.download_button("Download summary CSV", summary_df.to_csv(index=False).encode("utf-8"), file_name="cao_tga_cycle_summary.csv", mime="text/csv")
c2.download_button("Download Excel workbook", excel_buf, file_name="cao_tga_kinetic_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
c3.download_button("Download PNG figure", img_png, file_name="cao_tga_publication_figure.png", mime="image/png")
c4.download_button("Download SVG figure", img_svg, file_name="cao_tga_publication_figure.svg", mime="image/svg+xml")

st.caption("Model notes: capacity is calculated from the selected cycle's net mass gain and the chosen mass basis. Kinetic fits use normalized conversion α. In two-stage mode, every model independently optimizes its own transition time and separate parameters for each stage; AIC/BIC include the extra stage parameters and transition point.")
