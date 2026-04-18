# Magnets (Nam cham / Tu truong)

## Scope
This document provides grounded middle-school scientific facts for magnet experiments.
Use this file as authoritative context for simulator scripting.

## Core Rules
- Every magnet has two poles: North (N) and South (S).
- Opposite poles attract (N-S).
- Same poles repel (N-N, S-S).
- Magnetic poles always appear in pairs; cutting a bar magnet does not isolate a single pole.

## Magnetic Materials
- Ferromagnetic materials: iron, steel, nickel, cobalt.
- Non-magnetic or weakly magnetic examples for school context: plastic, wood, paper, aluminum.

## Field Concepts and Quantitative Notes
- Magnetic field symbol: B, unit tesla (T).
- Earth magnetic field magnitude is about 25 to 65 microtesla (uT), depending on location.
- For a small bar magnet, field strength decreases quickly with distance.
  Approximate dipole trend: B proportional to 1/r^3 when r is much larger than magnet size.

Practical classroom reference values (approximate):
- Very near pole surface (few mm): about 5 to 50 mT, depends on magnet grade.
- At around 2 cm from a small classroom bar magnet: often much lower, commonly in sub-mT to a few mT range.

## Observable Classroom Phenomena
1. Pole interaction:
- N near S: magnets move together.
- N near N or S near S: magnets push apart.

2. Iron filings pattern:
- Filings align along field lines.
- Highest filings density near poles.

3. Compass response:
- Without nearby magnets, needle aligns roughly North-South due to Earth field.
- Bringing bar magnet near compass causes needle deflection.

## Simulator-Ready Experiment Data
### Experiment A: Attraction and Repulsion of Poles
Objective:
- Demonstrate pole interaction rules with clear visual outcomes.

Materials:
- 2 bar magnets with labeled poles
- Low-friction track/table

Control parameters:
- Initial center-to-center distance: 8 cm
- Approach speed: 0.5 to 1.0 cm/s
- Keep magnets on same horizontal plane

Procedure and expected visuals:
1. Orient N pole of magnet A toward S pole of magnet B.
   - Visual: gap closes over time (attraction).
2. Reset to 8 cm.
3. Orient N pole of A toward N pole of B.
   - Visual: moving magnet slows then reverses direction (repulsion).

### Experiment B: Mapping Magnetic Field with Iron Filings
Objective:
- Show field-line pattern around bar magnet.

Materials:
- 1 bar magnet
- A4 paper sheet
- Fine iron filings

Control parameters:
- Filings mass spread: 0.5 to 1.0 g
- Paper-magnet separation: single sheet thickness (~0.1 mm)

Procedure and expected visuals:
1. Place magnet under paper.
2. Sprinkle filings uniformly.
3. Tap paper gently 3 to 5 times.
   - Visual: filings form curved lines from N region to S region.
   - Visual: highest density near both poles.

### Experiment C: Compass Deflection
Objective:
- Observe directional effect of external magnetic field.

Materials:
- 1 compass
- 1 bar magnet

Control parameters:
- Magnet-compass distance sequence: 10 cm, 6 cm, 3 cm
- Magnet orientation fixed during each trial

Procedure and expected visuals:
1. Record baseline compass heading (no magnet nearby).
2. Bring magnet to 10 cm.
   - Visual: small needle deflection.
3. Reduce distance to 6 cm, then 3 cm.
   - Visual: deflection angle increases with decreasing distance.

Safety notes:
- Keep strong magnets away from bank cards, watches, and sensitive electronics.
- Avoid pinching fingers between magnets.

Grounding constraints for script generation:
- Use only these rules/values for magnet-related scripts.
