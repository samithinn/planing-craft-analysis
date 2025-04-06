import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# Set page config for mobile-friendly layout
st.set_page_config(
    layout="wide",  # Use wide layout for better mobile experience
    initial_sidebar_state="collapsed"  # Start with sidebar collapsed on mobile
)

# Custom CSS for a mobile-friendly experience
st.markdown("""
    <style>
        .block-container {
            padding: 1rem;
            max-width: 100%;
            margin: 0;
        }
        .css-1y4t6uw {
            font-size: 16px;
            padding: 15px 30px;
        }
        .css-18e3th9 {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and inputs
st.title("🚢 Planing Craft Resistance and Porpoising Stability Analysis")

# === Inputs (Departure Only) ===
st.sidebar.header("🔧 Input: Departure Condition")
nabla_departure = st.sidebar.number_input("Displacement Volume (∇) [m³]", value=40.0)
LCG_departure = st.sidebar.number_input("LCG [m]", value=6.0)
Beam = st.sidebar.number_input("Beam (B) [m]", value=6.5)
beta_deg = st.sidebar.slider("Deadrise (β) [deg]", 0, 20, 6)
eta_t = st.sidebar.number_input(r"Transmission Efficiency ($\eta_t$)", value=0.95)
eta_d = st.sidebar.number_input(r"Propeller Efficiency ($\eta_d$)", value=0.50)

# Speed input
st.sidebar.header("📈 Speed Range")
speed_min = st.sidebar.number_input("Min Speed [knots]", value=10.0)
speed_max = st.sidebar.number_input("Max Speed [knots]", value=52.0)
speed_step = st.sidebar.number_input("Speed Step [knots]", value=1.0)
speeds_knots = np.arange(speed_min, speed_max + speed_step, speed_step)
speeds_ms = speeds_knots * 1852 / 3600

# Constants
g = 9.81
rho = 1027
nu = 1.356e-6

# === Functions ===
def calculate_CL0(C_L_beta, beta_deg):
    C_L0_n = C_L_beta
    for _ in range(100):
        C_L0_next = C_L_beta + 0.0065 * beta_deg * (C_L0_n ** 0.6)
        if abs(C_L0_next - C_L0_n) < 1e-6:
            break
        C_L0_n = C_L0_next
    return C_L0_n

def calculate_lambda(Lp, B, C_V):
    lambda_n = Lp / B
    for _ in range(100):
        fraction = (5.236 * C_V**2) / lambda_n**2
        denominator = 0.75 - (1 / (fraction + 2.4))
        lambda_next = (Lp / B) / denominator
        if abs(lambda_next - lambda_n) < 1e-6:
            break
        lambda_n = lambda_next
    return lambda_n

def porpoising_limit(beta_deg, C_L_beta):
    cl_half = C_L_beta / 2
    sqrt_cl = np.sqrt(cl_half)
    def limit_at(beta_val):
        if beta_val == 0:
            return 78.502 * cl_half + 15.232 * sqrt_cl - 2.3669
        elif beta_val == 10:
            return 76.916 * cl_half + 8.8667 * sqrt_cl + 0.1276
        elif beta_val == 20:
            return 68.500 * cl_half + 12.662 * sqrt_cl + 0.5128
    if beta_deg in [0, 10, 20]:
        return limit_at(beta_deg)
    elif 0 < beta_deg < 10:
        return np.interp(beta_deg, [0, 10], [limit_at(0), limit_at(10)])
    elif 10 < beta_deg < 20:
        return np.interp(beta_deg, [10, 20], [limit_at(10), limit_at(20)])
    return np.nan

# === Main Calculation Function ===
def run_analysis(nabla, LCG):
    results = []
    for V, V_kts in zip(speeds_ms, speeds_knots):
        try:
            Delta = nabla * g * rho
            C_L_beta = Delta / (0.5 * rho * V**2 * Beam**2)
            C_V = V / np.sqrt(g * Beam)
            C_L0 = calculate_CL0(C_L_beta, beta_deg)
            lambda_val = calculate_lambda(LCG, Beam, C_V)
            tau_deg = (C_L0 / ((0.012 * np.sqrt(lambda_val)) + ((0.0055 * lambda_val**2.5) / (C_V**2)))) ** (1 / 1.1)
            tau_rad = np.radians(tau_deg)

            exp_factor = (0.012 * np.sqrt(lambda_val) * tau_rad**1.1) ** (-0.4)
            exp_factor = min(exp_factor, 2.0)
            function_beta = max(0.1, (1 - (0.0065 * beta_deg * exp_factor)) * np.cos(np.radians(beta_deg)))
            inside_sqrt = (1 - ((0.012 * tau_rad**1.1) / (np.sqrt(lambda_val) * np.cos(tau_rad)))) * function_beta
            V_1 = np.sqrt(inside_sqrt) * V

            R_n = (V_1 * lambda_val * Beam) / nu
            CF_total = ((1 / (3.4 * np.log10(R_n) - 5.6)) ** 2 + 0.0004)
            D_F = ((0.5 * rho * V_1 ** 2 * lambda_val * Beam**2) / np.cos(np.radians(beta_deg))) * CF_total
            D_total = Delta * np.tan(tau_rad) + (D_F / np.cos(tau_rad))
            P_E = D_total * V
            P_B = P_E / (eta_t * eta_d)

            limit_trim = porpoising_limit(beta_deg, C_L_beta)
            stability = "Stable" if tau_deg <= limit_trim else "Unstable"

            results.append({
                'Speed [knots]': V_kts,
                'Speed [m/s]': V,
                'Lift Coeff (C_L_beta)': C_L_beta,
                '√(C_L_beta / 2)': np.sqrt(C_L_beta / 2),
                'Speed Coeff (C_V)': C_V,
                'Lift Coeff (C_L0)': C_L0,
                'Lambda (λ)': lambda_val,
                'Trim Angle [deg]': tau_deg,
                'Trim Angle [rad]': tau_rad,
                'Average Bottom Velocity (V₁) [m/s]': V_1,
                'Reynolds Number': R_n,
                'Skin Friction Coeff (C_F total)': CF_total,
                'Frictional Drag Force (D_F) [N]': D_F,
                'Total Resistance [kN]': D_total / 1e3,
                'Effective Power [kW]': P_E / 1e3,
                'Brake Power [kW]': P_B / 1e3,
                'Porpoising Limit [deg]': limit_trim,
                'Stability': stability
            })
        except:
            continue
    return pd.DataFrame(results)

# Trigger calculation and plots
if st.button("Run Resistance & Stability Analysis for Both Conditions (Departure and Arrival)"):
    df_dep = run_analysis(nabla_departure, LCG_departure)
    df_arr = run_analysis(0.8 * nabla_departure, (2.5 / 3.0) * LCG_departure)

    tabs = st.tabs(["📦 Departure", "⚓ Arrival", "📊 Combined Plots"])

    # Departure Tab
    with tabs[0]:
        st.subheader(f"📊 Departure Results")
        st.dataframe(df_dep, use_container_width=True)

        # Max Speed Summary
        max_idx = df_dep['Speed [knots]'].idxmax()
        st.markdown(f"### 📌 At Maximum Speed ({df_dep['Speed [knots]'][max_idx]:.2f} knots):")
        st.markdown(f"- Total Resistance = **{df_dep['Total Resistance [kN]'][max_idx]:.2f} kN**")
        st.markdown(f"- Brake Power = **{df_dep['Brake Power [kW]'][max_idx]:.2f} kW**")

        # Resistance Plot
        st.subheader("📈 Total Resistance vs Speed")
        fig_r, ax_r = plt.subplots()
        ax_r.plot(df_dep["Speed [knots]"], df_dep["Total Resistance [kN]"], marker='o')
        ax_r.set_xlabel("Speed [knots]")
        ax_r.set_ylabel("Total Resistance [kN]")
        ax_r.grid(True)
        st.pyplot(fig_r)

        # Brake Power Plot
        st.subheader("📈 Brake Power vs Speed")
        fig_b, ax_b = plt.subplots()
        ax_b.plot(df_dep["Speed [knots]"], df_dep["Brake Power [kW]"], marker='s', color='orange')
        ax_b.set_xlabel("Speed [knots]")
        ax_b.set_ylabel("Brake Power [kW]")
        ax_b.grid(True)
        st.pyplot(fig_b)

        # Porpoising vs Speed Plot
        st.subheader("📈 Porpoising Limit vs Speed")
        fig_p1, ax_p1 = plt.subplots()
        ax_p1.plot(df_dep["Speed [knots]"], df_dep["Porpoising Limit [deg]"], linestyle='--', color='black', label="Limit")
        ax_p1.scatter(df_dep["Speed [knots]"], df_dep["Trim Angle [deg]"], label="Trim Angle", color='blue')
        ax_p1.set_xlabel("Speed [knots]")
        ax_p1.set_ylabel("Trim Angle [deg]")
        ax_p1.legend()
        ax_p1.grid(True)
        st.pyplot(fig_p1)

        # Porpoising vs √(C_L_beta / 2) Plot
        st.subheader("📈 Porpoising Limit vs √(C_L_beta / 2)")
        fig_p2, ax_p2 = plt.subplots()
        ax_p2.plot(df_dep["√(C_L_beta / 2)"], df_dep["Porpoising Limit [deg]"], linestyle='--', color='black', label="Limit")
        ax_p2.scatter(df_dep["√(C_L_beta / 2)"], df_dep["Trim Angle [deg]"], label="Trim Angle", color='blue')
        ax_p2.set_xlabel("√(C_L_beta / 2)")
        ax_p2.set_ylabel("Trim Angle [deg]")
        ax_p2.legend()
        ax_p2.grid(True)
        st.pyplot(fig_p2)

        # Critical Speed Calculation
        critical_speed = None
        for i in range(1, len(df_dep)):
            if df_dep["Stability"].iloc[i - 1] == "Stable" and df_dep["Stability"].iloc[i] == "Unstable":
                tau1, tau2 = df_dep["Trim Angle [deg]"].iloc[i - 1], df_dep["Trim Angle [deg]"].iloc[i]
                lim1, lim2 = df_dep["Porpoising Limit [deg]"].iloc[i - 1], df_dep["Porpoising Limit [deg]"].iloc[i]
                spd1, spd2 = df_dep["Speed [knots]"].iloc[i - 1], df_dep["Speed [knots]"].iloc[i]
                diff1 = tau1 - lim1
                diff2 = tau2 - lim2
                if diff1 != diff2:
                    critical_speed = spd1 + (spd2 - spd1) * (-diff1) / (diff2 - diff1)
                break

        if critical_speed:
            st.markdown(f"🧭 **Estimated critical speed before instability: {critical_speed:.2f} knots**")
        else:
            st.markdown("✅ **No instability transition detected — stable throughout range.**")

        # Excel download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_dep.to_excel(writer, index=False, sheet_name="Departure Results")
        processed_data = output.getvalue()
        st.download_button("⬇️ Download Departure Results (Excel)", data=processed_data, file_name="departure_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # Arrival Tab
    with tabs[1]:
        st.subheader(f"⚓ Arrival Results")
        st.dataframe(df_arr, use_container_width=True)

        # Max Speed Summary
        max_idx = df_arr['Speed [knots]'].idxmax()
        st.markdown(f"### 📌 At Maximum Speed ({df_arr['Speed [knots]'][max_idx]:.2f} knots):")
        st.markdown(f"- Total Resistance = **{df_arr['Total Resistance [kN]'][max_idx]:.2f} kN**")
        st.markdown(f"- Brake Power = **{df_arr['Brake Power [kW]'][max_idx]:.2f} kW**")

        # Resistance Plot
        st.subheader("📈 Total Resistance vs Speed")
        fig_r, ax_r = plt.subplots()
        ax_r.plot(df_arr["Speed [knots]"], df_arr["Total Resistance [kN]"], marker='o')
        ax_r.set_xlabel("Speed [knots]")
        ax_r.set_ylabel("Total Resistance [kN]")
        ax_r.grid(True)
        st.pyplot(fig_r)

        # Brake Power Plot
        st.subheader("📈 Brake Power vs Speed")
        fig_b, ax_b = plt.subplots()
        ax_b.plot(df_arr["Speed [knots]"], df_arr["Brake Power [kW]"], marker='s', color='orange')
        ax_b.set_xlabel("Speed [knots]")
        ax_b.set_ylabel("Brake Power [kW]")
        ax_b.grid(True)
        st.pyplot(fig_b)

        # Porpoising vs Speed Plot
        st.subheader("📈 Porpoising Limit vs Speed")
        fig_p1, ax_p1 = plt.subplots()
        ax_p1.plot(df_arr["Speed [knots]"], df_arr["Porpoising Limit [deg]"], linestyle='--', color='black', label="Limit")
        ax_p1.scatter(df_arr["Speed [knots]"], df_arr["Trim Angle [deg]"], label="Trim Angle", color='blue')
        ax_p1.set_xlabel("Speed [knots]")
        ax_p1.set_ylabel("Trim Angle [deg]")
        ax_p1.legend()
        ax_p1.grid(True)
        st.pyplot(fig_p1)

        # Porpoising vs √(C_L_beta / 2) Plot
        st.subheader("📈 Porpoising Limit vs √(C_L_beta / 2)")
        fig_p2, ax_p2 = plt.subplots()
        ax_p2.plot(df_arr["√(C_L_beta / 2)"], df_arr["Porpoising Limit [deg]"], linestyle='--', color='black', label="Limit")
        ax_p2.scatter(df_arr["√(C_L_beta / 2)"], df_arr["Trim Angle [deg]"], label="Trim Angle", color='blue')
        ax_p2.set_xlabel("√(C_L_beta / 2)")
        ax_p2.set_ylabel("Trim Angle [deg]")
        ax_p2.legend()
        ax_p2.grid(True)
        st.pyplot(fig_p2)

        # Critical Speed Calculation
        critical_speed = None
        for i in range(1, len(df_arr)):
            if df_arr["Stability"].iloc[i - 1] == "Stable" and df_arr["Stability"].iloc[i] == "Unstable":
                tau1, tau2 = df_arr["Trim Angle [deg]"].iloc[i - 1], df_arr["Trim Angle [deg]"].iloc[i]
                lim1, lim2 = df_arr["Porpoising Limit [deg]"].iloc[i - 1], df_arr["Porpoising Limit [deg]"].iloc[i]
                spd1, spd2 = df_arr["Speed [knots]"].iloc[i - 1], df_arr["Speed [knots]"].iloc[i]
                diff1 = tau1 - lim1
                diff2 = tau2 - lim2
                if diff1 != diff2:
                    critical_speed = spd1 + (spd2 - spd1) * (-diff1) / (diff2 - diff1)
                break
        if critical_speed:
            st.markdown(f"🧭 **Estimated critical speed before instability: {critical_speed:.2f} knots**")
        else:
            st.markdown("✅ **No instability transition detected — stable throughout range.**")

        # Excel download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_arr.to_excel(writer, index=False, sheet_name="Arrival Results")
        processed_data = output.getvalue()
        st.download_button("⬇️ Download Arrival Results (Excel)", data=processed_data, file_name="arrival_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
