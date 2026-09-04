import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import io

st.set_page_config(layout="wide", page_title="CaO Sorbent Kinetic Analyzer")

# -----------------------------------------------------------------------------
# 1. KINETIC MODEL EQUATIONS (Normalized Conversion X vs Time t)
# -----------------------------------------------------------------------------
def model_first_order(t, k):
    return 1 - np.exp(-k * t)

def model_avrami(t, k, n):
    return 1 - np.exp(-(k * t)**n)

def model_shrinking_core(t, k):
    val = 1 - k * t
    val = np.clip(val, 0, None)
    return 1 - val**3

def model_random_pore(t, k, psi=2.0):
    return 1 - np.exp(-k * t * (1 + (psi * k * t) / 4))

def model_double_exponential(t, k1, k2, a):
    return a * (1 - np.exp(-k1 * t)) + (1 - a) * (1 - np.exp(-k2 * t))

def model_grain(t, k):
    val = 1 - k * t
    val = np.clip(val, 0, None)
    return 1 - val**3

def model_nth_order(t, k, n):
    if abs(n - 1.0) < 1e-3:
        return model_first_order(t, k)
    inner = 1 + (n - 1) * k * t
    inner = np.clip(inner, 1e-5, None)
    return 1 - inner**(1 / (1 - n))

def evaluate_two_stage(t, X, model_func, bounds_func, p0_func):
    best_rss = float('inf')
    best_results = None
    
    n_pts = len(t)
    if n_pts < 10:
        return None
        
    start_idx = max(2, int(n_pts * 0.15))
    end_idx = min(n_pts - 2, int(n_pts * 0.85))
    
    for idx in range(start_idx, end_idx):
        t1, X1 = t[:idx], X[:idx]
        t2, X2 = t[idx:] - t[idx], (X[idx:] - X[idx]) / (1.001 - X[idx])
        
        try:
            popt1, _ = curve_fit(model_func, t1, X1, p0=p0_func(1), bounds=bounds_func(1), maxfev=2000)
            X1_pred = model_func(t1, *popt1)
            
            popt2, _ = curve_fit(model_func, t2, X2, p0=p0_func(2), bounds=bounds_func(2), maxfev=2000)
            X2_pred_raw = model_func(t2, *popt2)
            X2_pred = X[idx] + X2_pred_raw * (1.001 - X[idx])
            
            total_pred = np.concatenate([X1_pred, X2_pred])
            rss = np.sum((X - total_pred)**2)
            
            if rss < best_rss:
                best_rss = rss
                best_results = {
                    't_tr': t[idx],
                    'X_tr': X[idx],
                    'idx_tr': idx,
                    'popt1': popt1,
                    'popt2': popt2,
                    'y_pred': total_pred,
                    'rss': rss
                }
        except:
            continue
            
    return best_results

# -----------------------------------------------------------------------------
# 2. STATISTICAL METRICS COMPLIANCE
# -----------------------------------------------------------------------------
def calculate_metrics(y_true, y_pred, n_param):
    n = len(y_true)
    rss = np.sum((y_true - y_pred)**2)
    if n <= n_param + 1:
        return 0, float('inf'), float('inf')
    
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (rss / ss_tot) if ss_tot != 0 else 0
    
    aic = n * np.log(rss / n) + 2 * n_param
    bic = n * np.log(rss / n) + n_param * np.log(n)
    return r2, aic, bic

# -----------------------------------------------------------------------------
# 3. INTERFACE AND RUNTIME
# -----------------------------------------------------------------------------
st.title("🔬 CaO Sorbent Calcium-Looping Kinetic Analyzer")
st.markdown("Upload raw TGA parameters, drag/crop cycle regimes, parse capture capacity, and run dynamically optimized two-stage kinetic fits.")

uploaded_file = st.file_uploader("Upload TGA Run CSV Data", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.sidebar.header("Column Mapping")
    t_col = st.sidebar.selectbox("Time Column", df.columns, index=0 if "time" in df.columns.str.lower() else 0)
    w_col = st.sidebar.selectbox("Weight Column", df.columns, index=1 if "weight" in df.columns.str.lower() else 1)
    temp_col = st.sidebar.selectbox("Temperature Column", df.columns, index=2 if "temp" in df.columns.str.lower() else 2)
    
    df = df[[t_col, w_col, temp_col]].dropna().astype(float).sort_values(by=t_col).reset_index(drop=True)
    
    st.subheader("1. Full Profile & Cycle Bounding Range")
    st.info("Use the Plotly interactive toolbar zoom/box-select to locate your cycle coordinates.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[t_col], y=df[w_col], name="Weight (mg)", yaxis="y1", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df[t_col], y=df[temp_col], name="Temp (°C)", yaxis="y2", line=dict(color="#ff7f0e", dash="dash")))
    
    fig.update_layout(
        xaxis=dict(title=dict(text="Time (s or min)")),
        yaxis=dict(
            title=dict(text="Weight (mg / %)", font=dict(color="#1f77b4")), 
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title=dict(text="Temperature (°C)", font=dict(color="#ff7f0e")), 
            tickfont=dict(color="#ff7f0e"), 
            overlaying="y", 
            side="right"
        ),
        height=450, 
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        t_start = st.number_input("Carbonation Start Time", min_value=float(df[t_col].min()), max_value=float(df[t_col].max()), value=float(df[t_col].min()))
    with col2:
        t_end = st.number_input("Carbonation End Time", min_value=float(df[t_col].min()), max_value=float(df[t_col].max()), value=float(df[t_col].max()))
        
    cycle_df = df[(df[t_col] >= t_start) & (df[t_col] <= t_end)].copy()
    
    if len(cycle_df) > 5:
        st.subheader("2. CO₂ Capture Performance Metrics")
        
        w_initial = cycle_df[w_col].iloc[0]
        w_final = cycle_df[w_col].iloc[-1]
        
        weight_gain = w_final - w_initial
        capacity = (weight_gain / w_initial) / 44.009 * 1000 if w_initial > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Initial Sample Weight", f"{w_initial:.4f} mg")
        c2.metric("Net CO₂ Captured Weight", f"{weight_gain:.4f} mg")
        c3.metric("CO₂ Capture Capacity", f"{capacity:.3f} mmol CO₂/g")
        
        cycle_df['t_rel'] = cycle_df[t_col] - t_start
        if weight_gain != 0:
            cycle_df['X'] = (cycle_df[w_col] - w_initial) / weight_gain
        else:
            cycle_df['X'] = 0.0
            
        cycle_df['X'] = np.clip(cycle_df['X'], 0.0, 1.0)
        
        t_arr = cycle_df['t_rel'].values
        X_arr = cycle_df['X'].values
        
        st.subheader("3. Multi-Model Kinetic Fitting Framework")
        fit_type = st.radio("Select Structuring Strategy", ["Single-Stage Continuous", "Two-Stage Optimized Split (Independent per Model)"])
        
        models_config = {
            "First-Order": {
                "func": model_first_order,
                "p0": lambda stage: [0.01],
                "bounds": lambda stage: ([0.0], [np.inf]),
                "n_p": 1
            },
            "Avrami": {
                "func": model_avrami,
                "p0": lambda stage: [0.01, 1.0],
                "bounds": lambda stage: ([0.0, 0.1], [np.inf, 5.0]),
                "n_p": 2
            },
            "Shrinking-Core (SCM)": {
                "func": model_shrinking_core,
                "p0": lambda stage: [0.005],
                "bounds": lambda stage: ([0.0], [np.inf]),
                "n_p": 1
            },
            "Random Pore (RPM)": {
                "func": model_random_pore,
                "p0": lambda stage: [0.01, 2.0] if stage==1 else [0.002, 2.0],
                "bounds": lambda stage: ([0.0, 0.0], [np.inf, 50.0]),
                "n_p": 2
            },
            "Double Exponential": {
                "func": model_double_exponential,
                "p0": lambda stage: [0.05, 0.005, 0.6],
                "bounds": lambda stage: ([0.0, 0.0, 0.0], [np.inf, np.inf, 1.0]),
                "n_p": 3
            },
            "Grain Model": {
                "func": model_grain,
                "p0": lambda stage: [0.005],
                "bounds": lambda stage: ([0.0], [np.inf]),
                "n_p": 1
            },
            "nth-Order": {
                "func": model_nth_order,
                "p0": lambda stage: [0.01, 1.5],
                "bounds": lambda stage: ([0.0, 0.0], [np.inf, 10.0]),
                "n_p": 2
            }
        }
        
        results_summary = []
        plot_traces = {}
        
        for name, cfg in models_config.items():
            if fit_type == "Single-Stage Continuous":
                try:
                    popt, _ = curve_fit(cfg["func"], t_arr, X_arr, p0=cfg["p0"](1), bounds=cfg["bounds"](1), maxfev=3000)
                    y_pred = cfg["func"](t_arr, *popt)
                    r2, aic, bic = calculate_metrics(X_arr, y_pred, cfg["n_p"])
                    
                    results_summary.append({
                        "Model": name, "Fit Strategy": "Single-Stage", 
                        "R²": round(r2, 4), "AIC": round(aic, 2), "BIC": round(bic, 2),
                        "Transition Point (t_tr)": "N/A", "Transition Conv (X_tr)": "N/A"
                    })
                    plot_traces[name] = y_pred
                except:
                    continue
            else:
                fit_res = evaluate_two_stage(t_arr, X_arr, cfg["func"], cfg["bounds"], cfg["p0"])
                if fit_res is not None:
                    r2, aic, bic = calculate_metrics(X_arr, fit_res['y_pred'], cfg["n_p"] * 2)
                    
                    results_summary.append({
                        "Model": name, "Fit Strategy": "Two-Stage Split", 
                        "R²": round(r2, 4), "AIC": round(aic, 2), "BIC": round(bic, 2),
                        "Transition Point (t_tr)": f"{fit_res['t_tr']:.2f} s", 
