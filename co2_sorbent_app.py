"""
CO2 Sorbent Capacity & Kinetics Analyzer
-----------------------------------------
For calcium-looping TGA data: drag-select the initial (post-calcination) and
final (carbonation plateau) mass of each cycle directly on the curve, get
mmol CO2/g for each cycle, and fit a kinetic model to each selected section
independently.

Run locally:
    pip install "streamlit>=1.35" pandas numpy plotly
    streamlit run co2_sorbent_app.py

Deploy online (free): push this file + requirements.txt to a GitHub repo,
then connect it at https://share.streamlit.io (Streamlit Community Cloud).

Note: click-drag chart selection requires Streamlit 1.35 or newer.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------- constants
M_CO2 = 44.01

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
    "Pseudo first-order": {
        "eq": "-ln(1-X) = k*t",
        "transform": lambda X, t: (t, -np.log(1 - X)),
        "invert": lambda k, n, t: 1 - np.exp(-k * t),
        "k_units": "1/time",
        "has_n": False,
    },
    "Avrami-Erofeev (JMAK)": {
        "eq": "ln(-ln(1-X)) = ln k + n*ln t",
        "transform": lambda X, t: (np.log(t), np.log(-np.log(1 - X))),
        "invert": lambda k, n, t: 1 - np.exp(-k * t ** n),
        "k_units": "1/time^n",
        "has_n": True,
    },
    "Shrinking core - reaction control": {
        "eq": "1-(1-X)^(1/3) = k*t",
        "transform": lambda X, t: (t, 1 - (1 - X) ** (1 / 3)),
        "invert": lambda k, n, t: 1 - (1 - min(k * t, 1)) ** 3,
        "k_units": "1/time",
        "has_n": False,
    },
    "Shrinking core - product-layer diffusion": {
        "eq": "1-(2/3)X-(1-X)^(2/3) = k*t",
        "transform": lambda X, t: (t, diffusion_f(X)),
        "invert": lambda k, n, t: diffusion_inv(k * t),
        "k_units": "1/time",
        "has_n": False,
    },
}
MODEL_NAMES = list(MODELS.keys())


def fit_model(model_name, t_arr, X_arr):
    m = MODELS[model_name]
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
    return {"k": k, "n": n, "r2": fit["r2"]}


def nearest_row(df, t):
    return df.iloc[(df["t"] - t).abs().idxmin()]


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
        "model": st.column_config.SelectboxColumn("Kinetic model", options=MODEL_NAMES),
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
    "Each cycle is fit with the model chosen for it in the table above. X is normalized to that "
    "cycle's own selected initial and final mass (0 at selection start, 1 at selection end)."
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

        if not row["two_stage"]:
            fit = fit_model(row["model"], t_arr, X_arr) if len(t_arr) >= 3 else None
            if fit:
                m = MODELS[row["model"]]
                y_curve = np.clip([m["invert"](fit["k"], fit["n"], t) for t in t_curve], 0, 1)
                kin_fig.add_trace(go.Scatter(x=t_curve, y=y_curve, mode="lines",
                                              line=dict(color="#2F6F5E", width=2), name="fit"))
                st.plotly_chart(kin_fig, use_container_width=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(f"k ({m['k_units']})", f"{fit['k']:.4g}")
                if m["has_n"]:
                    mc2.metric("n", f"{fit['n']:.3g}")
                mc3.metric("R\u00b2", f"{fit['r2']:.4f}")
                st.caption(f"*{m['eq']}*")
            else:
                st.warning("Not enough valid points to fit this cycle.")
        else:
            tx = row["transition_X"]
            mask1, mask2 = X_arr <= tx, X_arr > tx
            if mask1.sum() >= 3 and mask2.sum() >= 3:
                f1 = fit_model("Pseudo first-order", t_arr[mask1], X_arr[mask1])
                f2 = fit_model("Shrinking core - product-layer diffusion", t_arr[mask2], X_arr[mask2])
                t_split = t_arr[mask2].min()
                c1, c2 = t_curve[t_curve <= t_split], t_curve[t_curve > t_split]
                if f1 and len(c1):
                    y1 = np.clip([MODELS["Pseudo first-order"]["invert"](f1["k"], None, t) for t in c1], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c1, y=y1, mode="lines", line=dict(color="#2F6F5E", width=2), name="fast stage"))
                if f2 and len(c2):
                    y2 = np.clip([MODELS["Shrinking core - product-layer diffusion"]["invert"](f2["k"], None, t) for t in c2], 0, 1)
                    kin_fig.add_trace(go.Scatter(x=c2, y=y2, mode="lines", line=dict(color="#B5482F", width=2), name="diffusion stage"))
                kin_fig.add_vline(x=t_split, line_dash="dot", line_color="#B5482F")
                st.plotly_chart(kin_fig, use_container_width=True)
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("k\u2081 fast (1/time)", f"{f1['k']:.4g}" if f1 else "\u2014")
                mc2.metric("R\u00b2 fast", f"{f1['r2']:.4f}" if f1 else "\u2014")
                mc3.metric("k\u2082 diffusion (1/time)", f"{f2['k']:.4g}" if f2 else "\u2014")
                mc4.metric("R\u00b2 diffusion", f"{f2['r2']:.4f}" if f2 else "\u2014")
                st.caption(
                    f"Fast stage: *{MODELS['Pseudo first-order']['eq']}*. "
                    f"Diffusion stage: *{MODELS['Shrinking core - product-layer diffusion']['eq']}*, "
                    f"split at X = {tx:.2f}."
                )
            else:
                st.warning("Not enough points on one side of the transition to fit both stages.")
