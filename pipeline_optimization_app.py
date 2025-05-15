import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pipeline_model import solve_pipeline

    # --------------------
    # SOLVE using Bonmin locally (fallback to NEOS remote MINLP solvers)
    opts = {'tol':1e-3, 'acceptable_tol':1e-3, 'max_cpu_time':3000, 'max_iter':100000}
    solver = SolverFactory('bonmin')
    if solver.available():
        results = solver.solve(model, tee=True, options=opts)
    else:
        # Remote NEOS: try Bonmin, then Couenne
        manager = SolverManagerFactory('neos')
        try:
            results = manager.solve(model, opt='bonmin', tee=True)
        except Exception:
            results = manager.solve(model, opt='couenne', tee=True)

    # ---------------------
st.set_page_config(
    page_title="Pipeline Optimization Dashboard",
    layout="wide"
)

# ---------------------
# Custom CSS
# ---------------------
st.markdown(
    """
    <style>
      .metric-card { background: #f0f2f6; border-radius: 8px; padding: 15px; }
      .section-title { font-size: 1.3rem; font-weight: bold; margin-top: 1rem; }
      .stTabs [role=tab] { font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------
# Header
# ---------------------
st.markdown(
    """
    <div style='background: linear-gradient(90deg,#4f81bd,#2c3e50);padding:20px;border-radius:10px;'>
      <h1 style='color:white;text-align:center;margin:0;'>Pipeline Optimization Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------
# Sidebar Inputs
# ---------------------
with st.sidebar:
    st.title("🔧 Pipeline Inputs")
    with st.expander("Adjust Parameters", expanded=True):
        FLOW      = st.number_input("Flow rate (m³/hr)",    value=1000.0, step=10.0)
        KV        = st.number_input("Viscosity (cSt)",      value=1.0,    step=0.1)
        rho       = st.number_input("Density (kg/m³)",     value=850.0,  step=10.0)
        SFC_J     = st.number_input("SFC Jamnagar (gm/bhp/hr)", value=210.0, step=1.0)
        SFC_R     = st.number_input("SFC Rajkot (gm/bhp/hr)",   value=215.0, step=1.0)
        SFC_S     = st.number_input("SFC Surendranagar (gm/bhp/hr)", value=220.0, step=1.0)
        RateDRA   = st.number_input("DRA Rate (INR/L)",      value=1.0,    step=0.1)
        Price_HSD = st.number_input("HSD Rate (INR/L)",      value=90.0,   step=0.5)
    run = st.button("🚀 Run Optimization")

if run:
    with st.spinner("Solving pipeline optimization..."):
        res = solve_pipeline(FLOW, KV, rho, SFC_J, SFC_R, SFC_S, RateDRA, Price_HSD)

    stations = ["Vadinar","Jamnagar","Rajkot","Surendranagar","Viramgam"]

    # KPI Cards
    total_cost  = res.get("total_cost",0)
    pumps_total = sum(int(res.get(f"num_pumps_{s.lower()}",0)) for s in stations)
    avg_speed   = np.mean([res.get(f"speed_{s.lower()}",0) for s in stations])
    avg_eff     = np.mean([res.get(f"efficiency_{s.lower()}",0) for s in stations])

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cost (INR)", f"₹{total_cost:,.2f}")
    c2.metric("Total Pumps", pumps_total)
    c3.metric("Avg Speed (rpm)", f"{avg_speed:.1f}")
    c4.metric("Avg Efficiency (%)", f"{avg_eff:.1f}")

    # Tabs
    tab1,tab2,tab3 = st.tabs(["📋 Summary Table","💰 Cost Charts","⚙️ Performance Charts"])

    # Summary Table
    with tab1:
        st.markdown("<div class='section-title'>Optimized Parameters Summary</div>", unsafe_allow_html=True)
        data = {"Process Particulars":[
            "Power & Fuel cost (INR/day)","DRA cost (INR/day)",
            "No. of Pumps","Pump Speed (rpm)","Pump Efficiency (%)",
            "Reynold's No.","Dynamic Head Loss (mcl)","Velocity (m/s)",
            "Residual Head (mcl)","SDH (mcl)","Drag Reduction (%)"
        ]}
        for s in stations:
            k = s.lower()
            data[s] = [
                res.get(f"power_cost_{k}",0),
                res.get(f"dra_cost_{k}",0),
                int(res.get(f"num_pumps_{k}",0)),
                res.get(f"speed_{k}",0),
                res.get(f"efficiency_{k}",0),
                res.get(f"reynolds_{k}",0),
                res.get(f"head_loss_{k}",0),
                res.get(f"velocity_{k}",0),
                res.get(f"residual_head_{k}",0),
                res.get(f"sdh_{k}",0),
                res.get(f"drag_reduction_{k}",0)
            ]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

    # Cost Charts
    with tab2:
        st.markdown("<div class='section-title'>Cost Breakdown per Station</div>", unsafe_allow_html=True)
        cost_df = pd.DataFrame({
            "Station":stations,
            "Power & Fuel (INR/day)": [res.get(f"power_cost_{s.lower()}",0) for s in stations],
            "DRA (INR/day)":         [res.get(f"dra_cost_{s.lower()}",0)   for s in stations]
        })
        fig = px.bar(cost_df.melt(id_vars="Station", value_vars=["Power & Fuel (INR/day)","DRA (INR/day)"], var_name="Type", value_name="Amount"),
                     x="Station", y="Amount", color="Type", barmode="group", title="Cost Components by Station")
        fig.update_layout(xaxis_title="Station", yaxis_title="Amount (INR)")
        st.plotly_chart(fig, use_container_width=True)

    # Performance Charts
    with tab3:
        st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
        perf_df = pd.DataFrame({
            "Station":stations,
            "Head Loss (m)":   [res.get(f"head_loss_{s.lower()}",0) for s in stations]
        })
        fig = go.Figure(go.Bar(x=perf_df["Station"], y=perf_df["Head Loss (m)"], name="Head Loss (m)"))
        fig.update_layout(xaxis_title="Station", yaxis_title="Head Loss (m)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-title'>Pump Characteristic Curves</div>", unsafe_allow_html=True)
        sel = st.multiselect("Select stations", stations, default=stations)
        fr = np.arange(0,4501,100)
        for stn in sel:
            k=stn.lower(); A,B,C=res.get(f"coef_A_{k}"),res.get(f"coef_B_{k}"),res.get(f"coef_C_{k}")
            D,Min=res.get(f"dol_{k}"),res.get(f"min_rpm_{k}")
            if None in [A,B,C,D,Min]: continue
            dfh=pd.DataFrame();
            for rpm in np.arange(Min,D+1,100):
                dfh=pd.concat([dfh,pd.DataFrame({"Flow (m³/hr)":fr,"Head (m)":(A*fr**2+B*fr+C)*(rpm/D)**2})])
            figh=px.line(dfh,x="Flow (m³/hr)",y="Head (m)")
            figh.update_traces(mode="lines"); figh.update_layout(xaxis_title="Flow (m³/hr)", yaxis_title="Head (m)")
            st.plotly_chart(figh, use_container_width=True)

    st.markdown("---")
    st.caption("© 2025 Developed by Parichay Das")
