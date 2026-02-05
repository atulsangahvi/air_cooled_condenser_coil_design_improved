Air-Cooled Freon Condenser Design Tool
=======================================

A Streamlit-based application for designing air-cooled condenser coils using 
ε–NTU method with enhanced heat transfer correlations.

Features:
- Zone-by-zone sizing (Subcool → Condense → Desuperheat)
- Enhanced heat transfer correlations (Gnielinski, Shah2016, Schmidt)
- Air-side and refrigerant-side pressure drop calculations
- Multiple circuit support
- PDF reporting capability
- Case management with cloning

Enhanced Heat Transfer Correlations:
- Single-phase HTC: Gnielinski correlation (replaces Dittus-Boelter)
- Condensation HTC: Shah2016 correlation (latest for in-tube condensation)
- Fin Efficiency: Schmidt's method for plate-fin-tube geometry
- Void Fraction: Zivi model with slip ratio

Installation:
1. Install Python 3.8+ from python.org
2. Install required packages: pip install -r requirements.txt
3. Run the app: streamlit run air_cooled_condenser_enhanced.py

Usage:
1. Enter password: "S-----ju" (case-sensitive)
2. Configure geometry, refrigerant, and air conditions in sidebar
3. Enable "Use enhanced HT correlations" for best accuracy
4. View zone results, pressure drops, and summaries
5. Clone cases for comparison
6. Generate PDF reports

Password Protection:
The app is protected with password "Semaanju" to restrict access.

Dependencies:
- Streamlit for web interface
- CoolProp for refrigerant properties
- NumPy/Pandas for calculations
- Matplotlib for plotting

Version History:
v27 (2025-08-12): Enhanced HT correlations added
v26 (2025-08-12): Original version

Author: Condenser Design Tool
Contact: For support or questions
