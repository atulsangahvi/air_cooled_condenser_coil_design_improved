# -------------- PDF Report --------------
st.header("🧾 Report")

def dataframe_to_string(df_in: pd.DataFrame, max_rows=1000):
    return df_in.to_string(index=False, max_rows=max_rows)

def add_text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')
    y = 0.95
    plt.text(0.5, y, title, ha='center', va='top', fontsize=14, fontweight='bold')
    y -= 0.03
    txt = "\n".join(lines)
    plt.text(0.02, y, txt, ha='left', va='top', fontsize=9, family='monospace')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

if st.button("Generate PDF report"):
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Inputs & Geometry
        inputs_lines = [
            f"Date: {dt.datetime.now().isoformat(timespec='seconds')}",
            f"Refrigerant: {fluid}",
            f"Enhanced Correlations: {use_enhanced_correlations}",
            f"Air Inlet Temp (°C): {air_temp:.2f}",
            f"Air RH (%): {air_rh:.1f}",
            f"Air Flow (m³/h): {airflow_cmh:.1f}  ({airflow_m3s:.3f} m³/s)",
            f"Mass Flow Refrigerant (kg/s): {m_dot_total:.4f}",
            f"T1 (°C): {T1:.2f},  T_cond (°C): {T_cond:.2f},  T3 (°C): {T3:.2f}",
            "",
            "Geometry:",
            f"  Face dimensions: {face_width_m:.3f} m × {face_height_m:.3f} m",
            f"  Rows available: {num_rows}",
            f"  FPI: {fpi}",
            f"  Tubes per row: {tubes_per_row}",
            f"  Tube OD: {tube_od_mm:.3f} mm, ID: {tube_id*1000:.3f} mm",
            f"  Wall thickness: {tube_thk_mm:.3f} mm",
            f"  Vertical pitch: {tube_pitch_mm:.2f} mm",
            f"  Row pitch: {row_pitch_mm:.2f} mm",
            f"  Fin thickness: {fin_thk_mm:.3f} mm",
            f"  Fin material: {fin_material}",
            f"  Free area %: {free_area_percent}%",
            f"  Circuits: {num_feeds}",
            "",
            "Fouling & Materials:",
            f"  Rf_o (air-side): {Rf_o:.5f} m²K/W",
            f"  Rf_i (refrigerant-side): {Rf_i:.5f} m²K/W",
            f"  Tube k: {k_tube:.1f} W/m·K",
            f"  Condensation enhancement factor: {cond_enhance:.2f}",
            "",
            "Headers:",
            f"  Inlet: {inlet_header_label} ({D_inlet*1000:.1f} mm)",
            f"  Outlet: {outlet_header_label} ({D_outlet*1000:.1f} mm)",
        ]
        add_text_page(pdf, "Inputs & Geometry", inputs_lines)
        
        # Page 2: Calculated Geometry & Areas
        geom_lines = [
            "Geometry & Areas:",
            f"  Frontal area: {frontal_area:.3f} m²",
            f"  Free-flow area: {free_area:.3f} m²",
            f"  Face velocity: {face_velocity:.2f} m/s",
            f"  Fin-channel velocity: {fin_velocity:.2f} m/s",
            f"  Total tubes: {total_tubes}",
            f"  Total tube length (all rows): {total_tube_length_all:.2f} m",
            f"  Tube external area: {A_tube_ext:.2f} m²",
            f"  Fin area (raw): {A_fin_raw:.2f} m²",
            f"  Total air-side area: {A_air_total:.2f} m²",
            f"  Area per row: {area_per_row:.2f} m²",
            f"  Fin efficiency η_f: {eta_fin:.3f}",
            f"  Overall surface efficiency η_o: {eta_o:.3f}",
            "",
            "Refrigerant per circuit:",
            f"  ṁ per circuit: {m_dot_circuit:.4f} kg/s",
            f"  A_i (one tube): {A_i*1e6:.2f} mm²",
            f"  G_i: {G_i:.1f} kg/m²·s",
            f"  u_i (vapor): {u_i_v:.2f} m/s",
            f"  u_i (liquid): {u_i_l:.2f} m/s",
            f"  Re_i (vapor): {Re_v_i:.0f}",
            f"  Re_i (liquid): {Re_l_i:.0f}",
            "",
            "Heat Transfer Correlations Used:",
            f"  Single-phase: {'Gnielinski' if use_enhanced_correlations else 'Dittus-Boelter'}",
            f"  Condensation: {'Shah2016' if use_enhanced_correlations else 'Simple multiplier'}",
            f"  Fin efficiency: {'Schmidt' if use_enhanced_correlations else 'Standard'}",
        ]
        add_text_page(pdf, "Calculated Geometry & Heat Transfer", geom_lines)
        
        # Page 3: Air Properties & HTCs
        air_lines = [
            "Air Properties:",
            f"  Density: {rho_air:.3f} kg/m³",
            f"  Viscosity: {mu_air:.6f} Pa·s",
            f"  Thermal conductivity: {k_air:.4f} W/m·K",
            f"  Specific heat: {cp_air:.1f} J/kg·K",
            f"  Prandtl number: {Pr_air:.3f}",
            f"  Air heat capacity rate: {C_air_WK/1000:.3f} kW/K",
            "",
            "External HTC (air-side):",
            f"  Re_air (Do-based): {Re_air:.0f}",
            f"  Nu_air: {Nu_air:.2f}",
            f"  h_air: {h_air:.1f} W/m²·K",
            "",
            "Refrigerant-side HTCs:",
            f"  h_i_desuperheat: {h_i_desuper:.1f} W/m²·K",
            f"  h_i_condensation: {h_i_cond:.1f} W/m²·K",
            f"  h_i_subcool: {h_i_subcool:.1f} W/m²·K",
            "",
            "Overall U (air-side basis):",
            f"  Uo_desuperheat: {Uo_desuper:.1f} W/m²·K",
            f"  Uo_condensation: {Uo_cond:.1f} W/m²·K",
            f"  Uo_subcool: {Uo_subcool:.1f} W/m²·K",
        ]
        add_text_page(pdf, "Properties & Heat Transfer Coefficients", air_lines)
        
        # Page 4: Zone Results
        zone_header = "Zone, Q_req (kW), Area needed (m²), Length per circuit (m), Rows required, Air out (°C), Uo (W/m²K), NTU, C_r, C_min (kW/K), ΔT_in (°C), Q_actual (kW), ε_used"
        zone_lines = [zone_header]
        for row in results:
            zone_lines.append(
                f"{row[0]}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}, {row[4]:.3f}, "
                f"{row[5]:.2f}, {row[6]:.1f}, {row[7]:.3f}, {row[8]:.3f}, "
                f"{row[9]:.3f}, {row[10]:.2f}, {row[11]:.3f}, {row[12]:.3f}"
            )
        
        # Add summary at the end
        zone_lines.append("")
        zone_lines.append("SUMMARY:")
        zone_lines.append(f"Total heat duty: {(Q_subcool_W + Q_condense_W + Q_desuper_W)/1000:.2f} kW")
        zone_lines.append(f"Total rows required (sum): {df['Rows required'].sum():.3f}")
        zone_lines.append(f"Rows available (input): {num_rows}")
        zone_lines.append(f"Air outlet after last zone: {results[-1][5]:.2f} °C")
        
        add_text_page(pdf, "Zone Results (Subcool → Condense → Desuperheat)", zone_lines)
        
        # Page 5: Pressure Drops
        dp_lines = [
            "AIR-SIDE PRESSURE DROP:",
            f"Core Δp Model: {dp_model}",
            f"Fin spacing s: {s_fin*1000:.3f} mm",
            f"Hydraulic diameter D_h: {D_h*1000:.3f} mm",
            f"Channel Reynolds Re_ch: {Re_ch:.0f}",
            f"Darcy friction f_D: {f_D:.4f}",
            f"Flow length L_flow: {L_flow:.3f} m",
            f"Core friction Δp: {dp_fric:.1f} Pa",
            f"Minor losses Δp: {dp_minor:.1f} Pa",
            f"Core Δp total: {dp_core_total:.1f} Pa",
            "",
            "Top Plenum & 90° Turn:",
            f"Upward area A_up: {A_up:.3f} m²",
            f"Upward velocity V_up: {V_up:.2f} m/s",
            f"K_turn total: {K_turn_total:.2f}",
            f"Turn Δp: {dp_turn:.1f} Pa",
            "",
            "TOTAL EXTERNAL STATIC:",
            f"Core + Turn = {dp_ext_total:.1f} Pa",
            "",
            "REFRIGERANT-SIDE PRESSURE DROP (per circuit):",
        ]
        
        # Add refrigerant pressure drop if calculated
        try:
            if 'df_dp' in locals():
                dp_lines.append("Zone, L_used (m), Phase model, Re, f, Δp_zone (kPa)")
                for _, row in df_dp.iterrows():
                    dp_lines.append(
                        f"{row['Zone']}, {row['L_used per circuit (m)']:.2f}, "
                        f"{row['Phase model']}, {row['Re']:.0f}, {row['f']:.4f}, "
                        f"{row['Δp_zone (kPa)']:.3f}"
                    )
                dp_lines.append(f"Total friction Δp: {total_dp_kPa:.3f} kPa")
            
            if 'df_extra' in locals():
                dp_lines.append("")
                dp_lines.append("Bends & Headers:")
                for _, row in df_extra.iterrows():
                    dp_lines.append(
                        f"{row['Component']}, {row['Count/Length']:.2f}, "
                        f"{row['f or K']:.4f}, {row['Δp (kPa)']:.3f}"
                    )
                dp_lines.append(f"Total bends & headers: {dp_extra_kPa:.3f} kPa")
                dp_lines.append(f"GRAND TOTAL (per circuit): {total_with_extras:.3f} kPa")
                
                # Comment on pressure drop magnitude
                if total_with_extras > 50:
                    dp_lines.append("")
                    dp_lines.append("NOTE: High refrigerant-side pressure drop detected!")
                    dp_lines.append("Consider: Larger tubes, fewer circuits, or shorter circuits.")
        except:
            dp_lines.append("(Refrigerant Δp calculation not available)")
        
        add_text_page(pdf, "Pressure Drop Analysis", dp_lines)
        
        # Page 6: Advisories if shown
        if show_advice and advisories:
            adv_lines = ["ADVISORY (non-binding):"]
            adv_lines.append("Zone, Feasible?, ε_cap, C_min (kW/K), C_min_needed (kW/K), Multiplier, ΔT_in (°C), Q_cap (kW), Q_req (kW)")
            for adv in advisories:
                adv_lines.append(
                    f"{adv['Zone']}, {adv['Feasible?']}, {adv['ε_cap']:.3f}, "
                    f"{adv['C_min (kW/K)']:.3f}, "
                    f"{adv.get('C_min_needed (kW/K)', 'N/A')}, "
                    f"{adv.get('Multiplier vs current', 'N/A')}, "
                    f"{adv['ΔT_in (°C)']:.2f}, {adv['Q_cap (kW)']:.2f}, "
                    f"{adv['Q_req (kW)']:.2f}"
                )
            add_text_page(pdf, "Design Advisories", adv_lines)
        
    buffer.seek(0)
    st.download_button("Download PDF report", data=buffer.getvalue(),
                       file_name="condenser_report.pdf", mime="application/pdf")
