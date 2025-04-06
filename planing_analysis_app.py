import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.markdown("""
    <style>
        .block-container {
            padding: 2rem 5rem;
            max-width: 1400px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🚢 Planing Craft Resistance and Porpoising Stability Analysis")

# === Inputs (Departure Only) ===
st.sidebar.header("🔧 Input: Departure Condition")
nabla_departure = st.sidebar.number_input("Displacement Volume (∇) [m³]", value=40.0)
LCG_departure = st.sidebar.number_input("LCG [m]", value=6.0)
Beam = st.sidebar.number_input("Beam (B) [m]", value=6.5)
beta_deg = st.sidebar.slider("Deadrise (β) [deg]", 0, 20, 6)
eta_t = st.sidebar.number_input("Transmission Efficiency (ηₜ)", value=0.95)
eta_d = st.sidebar.number_input("Propeller Efficiency (η_d)", value=0.50)

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

# === RUN ANALYSIS ===
if st.button("Run Resistance & Stability Analysis for Both Conditions (Departure and Arrival)"):
    df_dep = run_analysis(nabla_departure, LCG_departure)
    df_arr = run_analysis(0.8 * nabla_departure, (2.5 / 3.0) * LCG_departure)

    tabs = st.tabs(["📦 Departure", "⚓ Arrival", "📊 Combined Plots"])
    for df, name, tab in zip([df_dep, df_arr], ["Departure", "Arrival"], tabs[:2]):
        with tab:
            st.subheader(f"📊 {name} Results")
            st.dataframe(df, use_container_width=True)

            # Excel Download
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Results')
                writer.save()
            processed_data = output.getvalue()

            st.download_button(
                label=f"⬇️ Download {name} Results (Excel)",
                data=processed_data,
                file_name=f"{name.lower()}_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    # Combined Tab
    with tabs[2]:
        st.subheader("📊 Combined Plot Comparison")
        fig1, ax1 = plt.subplots()
        ax1.plot(df_dep["Speed [knots]"], df_dep["Total Resistance [kN]"], label="Departure")
        ax1.plot(df_arr["Speed [knots]"], df_arr["Total Resistance [kN]"], label="Arrival")
        ax1.set_title("Total Resistance vs Speed")
        ax1.set_xlabel("Speed [knots]")
        ax1.set_ylabel("Resistance [kN]")
        ax1.grid(True)
        ax1.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.plot(df_dep["Speed [knots]"], df_dep["Brake Power [kW]"], label="Departure")
        ax2.plot(df_arr["Speed [knots]"], df_arr["Brake Power [kW]"], label="Arrival")
        ax2.set_title("Brake Power vs Speed")
        ax2.set_xlabel("Speed [knots]")
        ax2.set_ylabel("Brake Power [kW]")
        ax2.grid(True)
        ax2.legend()
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        ax3.plot(df_dep["Speed [knots]"], df_dep["Trim Angle [deg]"], label="Departure Trim")
        ax3.plot(df_dep["Speed [knots]"], df_dep["Porpoising Limit [deg]"], linestyle='--', label="Departure Limit")
        ax3.plot(df_arr["Speed [knots]"], df_arr["Trim Angle [deg]"], label="Arrival Trim")
        ax3.plot(df_arr["Speed [knots]"], df_arr["Porpoising Limit [deg]"], linestyle='--', label="Arrival Limit")
        ax3.set_title("Trim vs Porpoising Limit")
        ax3.set_xlabel("Speed [knots]")
        ax3.set_ylabel("Angle [deg]")
        ax3.grid(True)
        ax3.legend()

        mid_speed = (speeds_knots[0] + speeds_knots[-1]) / 2
        ax3.text(mid_speed, 20, "REGIME OF\nPORPOISING", fontsize=10, color='white', ha='center', va='center',
                 bbox=dict(facecolor='red', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.4'))
        ax3.text(mid_speed, 2, "REGIME OF\nSTABLE PLANING", fontsize=10, color='black', ha='center', va='center',
                 bbox=dict(facecolor='lightgreen', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.4'))
        st.pyplot(fig3)

        combined_df = pd.concat([
            df_dep.assign(Condition="Departure"),
            df_arr.assign(Condition="Arrival")
        ], ignore_index=True)

        # Excel for combined
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            combined_df.to_excel(writer, index=False, sheet_name='Combined_Results')
            writer.save()
        excel_data = output.getvalue()

        st.download_button(
            label="⬇️ Download Combined Results (Departure + Arrival)",
            data=excel_data,
            file_name="combined_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Footer
st.markdown("---")
st.markdown(
    """
    📘 *This application was created by **Samithin Kongkaew**, MSc Naval Architecture, Newcastle University.*  
    🛠️ Developed as part of the module **MAR8178 Advanced Marine Propulsion Technology**  
    🎓 Supervised by **Dr. David Trodden**  
    📐 Calculations are based on **Savitsky's method (1964)**
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    🛠️ *Developed by **Samithin Kongkaew***  
    MSc Naval Architecture, Newcastle University
    
    🎓 MAR8178 — **Dr. David Trodden**  
    📐 Based on **Savitsky's method**
    """
)
