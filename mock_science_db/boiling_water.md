# Boiling Water (Dun soi / Nhiet hoc)

## Scope
This document provides grounded chemistry/physics facts for boiling-water experiments.
Use this file as authoritative context for simulator scripts.

## Core Rules
- Boiling occurs when vapor pressure of liquid water equals external pressure.
- At standard pressure 1 atm (101.325 kPa), pure water boils at 100.0 C.
- Lower pressure -> lower boiling point.
- Higher pressure -> higher boiling point.

## Pressure-Temperature Reference Table (Water)
Approximate boiling points:

| External pressure (kPa) | Boiling point (C) |
| --- | --- |
| 50.0 | 81.3 |
| 70.0 | 90.0 |
| 80.0 | 93.5 |
| 90.0 | 96.7 |
| 101.3 | 100.0 |
| 120.0 | 104.8 |

## Heat and Energy Relations
Sensible heating before boiling:
Q1 = m * c * DeltaT

Latent heat during boiling:
Q2 = m * Lv

Where:
- m = mass of water (kg)
- c = 4.18 kJ/(kg.K) for liquid water
- DeltaT in K or C difference
- Lv (at ~100 C) = 2260 kJ/kg

Example calculation:
- m = 0.50 kg water
- Heat from 25 C to 100 C:
  Q1 = 0.50 * 4.18 * (100 - 25) = 156.75 kJ
- Boil all 0.50 kg at 100 C:
  Q2 = 0.50 * 2260 = 1130 kJ
- Total: Q_total = 1286.75 kJ

## Observable Boiling Phenomena
1. Pre-boiling (below boiling point):
- Small bubbles may form from dissolved gas and collapse before reaching surface.

2. Onset of boiling:
- Continuous vapor bubbles form at nucleation sites and rise to surface.
- Temperature stabilizes near boiling point while liquid is actively boiling (at fixed pressure).

3. Vigorous boiling:
- Bubble generation rate increases with heater power.
- Surface agitation becomes strong and continuous.

## Simulator-Ready Experiment Data
### Experiment A: Boiling at Standard Pressure
Objective:
- Observe phase change and temperature plateau at 1 atm.

Materials:
- Beaker (600 mL)
- 500 mL water
- Hot plate
- Thermometer or digital temperature probe

Control parameters:
- Water volume: 500 mL
- Initial water temperature: 25 C (+/-2 C)
- Pressure: 101.3 kPa
- Heater power setting: fixed medium-high

Procedure and expected visuals:
1. Heat water from 25 C.
   - Visual: weak convection currents; occasional tiny bubbles on container wall.
2. Near 95 to 100 C:
   - Visual: frequent bubbles rising, steam visible above surface.
3. At sustained boil (~100 C at 101.3 kPa):
   - Visual: continuous bubble columns and strong surface agitation.
   - Data cue: temperature remains near 100 C while boiling continues.

### Experiment B: Pressure Effect on Boiling Point (Simulation)
Objective:
- Show that boiling point changes with pressure.

Control parameters:
- Case 1 pressure: 101.3 kPa -> target boil 100.0 C
- Case 2 pressure: 80.0 kPa -> target boil 93.5 C
- Same water mass and heater profile in both cases

Expected comparative visuals/data:
- At 80.0 kPa, vigorous boiling starts at lower temperature than in 101.3 kPa case.
- Temperature plateau shifts from ~100.0 C to ~93.5 C.

Safety notes:
- Use heat-resistant gloves.
- Keep face away from steam plume.
- Do not seal a heated container completely.

Grounding constraints for script generation:
- Use only this file's pressure-boiling table and equations for boiling-water scripts.
