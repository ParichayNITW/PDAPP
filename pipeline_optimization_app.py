import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pipeline_model import solve_pipeline

# ---------------------
# Page configuration
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

    # Main Tabs
    tab1,tab2,tab3,tab4,tab5 = st.tabs(["📋 Summary Table","💰 Cost Charts","⚙️ Performance Charts","🌀 System Curves","🔄 Pump-System Interaction"])

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
        fig_cost = px.bar(
            cost_df.melt(id_vars="Station", value_vars=["Power & Fuel (INR/day)","DRA (INR/day)"],
                         var_name="Cost Type", value_name="Amount"),
            x="Station", y="Amount", color="Cost Type", barmode="group", height=450,
            title="Cost Components by Station"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # Performance Charts
    with tab3:
        perf_tab1, perf_tab2 = st.tabs(["Performance Metrics","Pump Characteristic Curves"])

        # Perf Metrics
        with perf_tab1:
            st.markdown("<div class='section-title'>Performance Metrics</div>", unsafe_allow_html=True)
            perf_df = pd.DataFrame({
                "Station":stations,
                "Head Loss (m)":   [res.get(f"head_loss_{s.lower()}",0) for s in stations]
            })
            fig_perf = go.Figure()
            fig_perf.add_trace(go.Bar(
                x=perf_df["Station"],
                y=perf_df["Head Loss (m)"],
                name="Head Loss (m)"
            ))
            fig_perf.update_layout(
                xaxis_title="Station",
                yaxis_title="Head Loss (m)",
                height=500
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        # Pump Curves
        with perf_tab2:
            st.markdown("<div class='section-title'>Pump Characteristic Curves</div>", unsafe_allow_html=True)
            sel = st.multiselect("Select stations", stations, default=stations)
            flow_range = np.arange(0, 4500+1, 100)

            # Q-H curves
            st.subheader("Flow vs TDHA (Head) by RPM")
            for stn in sel:
                k = stn.lower()
                A,B,C = res.get(f"coef_A_{k}"), res.get(f"coef_B_{k}"), res.get(f"coef_C_{k}")
                DOL,Min = res.get(f"dol_{k}"), res.get(f"min_rpm_{k}")
                if None in [A,B,C,DOL,Min]: 
                    st.warning(f"Missing pump curve params for {stn}")
                    continue
                dfh = pd.DataFrame()
                for rpm in np.arange(Min, DOL+1, 100):
                    H = (A*flow_range**2 + B*flow_range + C)*(rpm/DOL)**2
                    dfh = pd.concat([dfh, pd.DataFrame({"Flow (m³/hr)": flow_range, "Head (m)": H, "RPM": rpm})])
                fig_h = px.line(
                    dfh, x="Flow (m³/hr)", y="Head (m)", color="RPM", markers=True,
                    title=f"Flow vs Head Curves ({stn})"
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # Q-Eff curves
            st.subheader("Flow vs Efficiency by RPM")
            for stn in sel:
                k = stn.lower()
                P,Q,R,S,T = [res.get(f"coef_{x}_{k}") for x in ['P','Q','R','S','T']]
                DOL,Min = res.get(f"dol_{k}"), res.get(f"min_rpm_{k}")
                if None in [P,Q,R,S,T,DOL,Min]:
                    st.warning(f"Missing efficiency curve params for {stn}")
                    continue
                dfe = pd.DataFrame()
                for rpm in np.arange(Min, DOL+1, 100):
                    eqQ = flow_range * DOL / rpm
                    E = (P*eqQ**4 + Q*eqQ**3 + R*eqQ**2 + S*eqQ + T) / 100
                    dfe = pd.concat([dfe, pd.DataFrame({"Flow (m³/hr)": flow_range, "Efficiency (%)": E, "RPM": rpm})])
                fig_e = px.line(
                    dfe, x="Flow (m³/hr)", y="Efficiency (%)", color="RPM", markers=True,
                    title=f"Flow vs Efficiency Curves ({stn})"
                )
                st.plotly_chart(fig_e, use_container_width=True)

        # System Curves of SDHR
    with tab4:
        st.markdown("<div class='section-title'>System Curves of SDHR</div>", unsafe_allow_html=True)
        all_stations = ["Vadinar","Jamnagar","Rajkot","Chotila","Surendranagar"]
        params = {
            "Vadinar": {"d":0.697, "L":46.7, "e":0.00004},
            "Jamnagar": {"d":0.697, "L":67.9, "e":0.00004},
            "Rajkot": {"d":0.697, "L":40.2, "e":0.00004},
            "Chotila": {"d":0.697, "L":60.0, "e":0.00004},
            "Surendranagar": {"d":0.697, "L":60.0, "e":0.00004}
        }
        static_heads = {
            "Vadinar": 24-8,
            "Jamnagar": 113-24,
            "Rajkot": 232-113,
            "Chotila": 80-232,
            "Surendranagar": 23-80
        }
        flow_range = np.arange(0,4501,100)
        for stn in all_stations:
            p = params[stn]
            sd0 = static_heads[stn]
            df_sys = pd.DataFrame()
            for dra in range(0,41,5):
                v = flow_range/(3.414*p["d"]**2/4)/3600
                Re = v*p["d"]/(KV*1e-6)
                f = 0.25/(np.log10((p["e"]/p["d"]/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(p["L"]*1000/p["d"])*(v**2/(2*9.81)))*(1 - dra/100)
                SDHR = sd0 + DH
                df2 = pd.DataFrame({"Flow (m³/hr)": flow_range, "SDHR (m)": SDHR, "DRA (%)": dra})
                df_sys = pd.concat([df_sys, df2], ignore_index=True)
            fig_sys = px.line(df_sys, x="Flow (m³/hr)", y="SDHR (m)", color="DRA (%)",
                              title=f"System Curve: SDHR vs Flow ({stn})")
            st.plotly_chart(fig_sys, use_container_width=True)

    # Pump-System Interaction
    with tab5:
        st.markdown("<div class='section-title'>Pump-System Interaction</div>", unsafe_allow_html=True)
        pump_stations = ["Vadinar","Jamnagar","Rajkot","Surendranagar"]
        flow_range = np.arange(0,4501,100)
        for stn in pump_stations:
            k = stn.lower()
            # system curves at 0 and 40%
            p = params[stn]
            sd0 = static_heads[stn]
            df_sys2 = pd.DataFrame()
            for dra in [0,40]:
                v = flow_range/(3.414*p["d"]**2/4)/3600
                Re = v*p["d"]/(KV*1e-6)
                f = 0.25/(np.log10((p["e"]/p["d"]/3.7)+(5.74/(Re**0.9)))**2)
                DH = (f*(p["L"]*1000/p["d"])*(v**2/(2*9.81)))*(1 - dra/100)
                SDHR = sd0 + DH
                df_sys2 = pd.concat([df_sys2, pd.DataFrame({"Flow (m³/hr)": flow_range, "Head (m)": SDHR, "Curve": f"System, DRA={dra}%"})], ignore_index=True)
            # pump curves
            df_pump = pd.DataFrame()
            for rpm in np.arange(res.get(f"min_rpm_{k}"), res.get(f"dol_{k}")+1, 100):
                A,B,C = res.get(f"coef_A_{k}"), res.get(f"coef_B_{k}"), res.get(f"coef_C_{k}")
                H = (A*flow_range**2 + B*flow_range + C)*(rpm/res.get(f"dol_{k}"))**2
                df_pump = pd.concat([df_pump, pd.DataFrame({"Flow (m³/hr)": flow_range, "Head (m)": H, "Curve": f"Pump, {rpm} rpm"})], ignore_index=True)
            df_all = pd.concat([df_sys2, df_pump], ignore_index=True)
            fig_int = px.line(df_all, x="Flow (m³/hr)", y="Head (m)", color="Curve",
                              title=f"Pump-System Interaction ({stn})")
            st.plotly_chart(fig_int, use_container_width=True)

    st.markdown("---")
    st.caption("© 2025 Developed by Parichay Das")
