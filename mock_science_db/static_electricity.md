# Static Electricity (Tinh dien / Tinh dien hoc)

## Scope
This document provides middle-school-level scientific facts for experiments about static electricity.
Use these facts as authoritative context for simulation scripts.

## Core Rules
- Electric charge has two types: positive (+) and negative (-).
- Like charges repel, unlike charges attract.
- Charge is conserved in an isolated system.
- Unit of electric charge: coulomb (C).
- Electron charge magnitude: e = 1.602e-19 C.
- Typical classroom static effects are strongest on dry days (relative humidity below about 50%).

## Material Behavior
- Conductors (example: metals) allow charge movement.
- Insulators (example: plastic, rubber, dry paper) hold localized charge.
- Triboelectric effect: rubbing two materials can transfer electrons.
  - Balloon rubbed on dry hair: balloon often becomes negatively charged.
  - Hair tends to become positively charged after electron transfer.

## Quantitative Relation (Reference)
Coulomb law for point charges:
F = k * |q1 * q2| / r^2

Where:
- F is electric force in newtons (N)
- k = 8.99e9 N.m^2/C^2
- q1, q2 are charges in coulombs (C)
- r is distance in meters (m)

Direction rule:
- Opposite signs -> attractive force
- Same signs -> repulsive force

Worked numeric example:
- q1 = +2.0e-6 C
- q2 = -1.0e-6 C
- r = 0.050 m
- F = 8.99e9 * (2.0e-6 * 1.0e-6) / (0.050)^2 = 7.19 N (attractive)

## Observable Classroom Phenomena
1. Balloon and paper bits:
- If charged balloon is brought to paper bits at distance 2 to 3 cm, paper bits start moving toward balloon.
- At distance below about 1 cm, some paper bits jump and briefly stick.

2. Balloon and wall:
- Charged balloon can stick to a painted wall for several seconds to minutes due to induced polarization.

3. Hair standing:
- After rubbing balloon on hair for 15 to 30 seconds, hair strands separate and stand due to repulsion of similarly charged strands.

4. Humidity effect:
- At high humidity (about 70%+), charge leaks faster and visual effects become weaker.

## Simulator-Ready Experiment Data
### Experiment A: Charge a Balloon by Friction
Objective:
- Show charging by friction and electrostatic attraction.

Materials:
- 1 latex balloon
- Dry cotton/wool cloth or dry hair
- Small dry paper pieces (3 to 5 mm)
- Non-metal table

Control parameters:
- Rubbing duration: 20 s
- Target room humidity: 30% to 50%
- Initial balloon-paper distance: 5 cm
- Approach speed: about 1 cm/s

Procedure and expected visuals:
1. Rub balloon with dry cloth/hair for 20 s.
   - Visual: no spark required; balloon marked as charged.
2. Move balloon toward paper from 5 cm down to 0.5 cm.
   - Visual at 2 to 3 cm: nearest paper pieces begin to slide/tilt toward balloon.
   - Visual below 1 cm: paper jumps to balloon surface.
3. Lift balloon away slowly.
   - Visual: some paper pieces drop after short sticking time.

Safety notes:
- Keep away from open flames.
- Avoid rubbing near sensitive electronics.

### Experiment B: Repulsion of Similar Charges (Conceptual)
Objective:
- Demonstrate that like charges repel.

Materials:
- 2 light pith balls or 2 lightweight aluminum-foil balls on strings
- 1 charged rod/balloon

Control parameters:
- Initial separation between hanging balls: 1 cm
- Contact time with charging rod: 1 to 2 s each

Procedure and expected visuals:
1. Touch both hanging balls with same charged object.
2. Release both balls.
   - Visual: distance between balls increases (repulsion).
3. Record maximum separation angle for each string.

Grounding constraints for script generation:
- Use only values and rules from this file when topic is static electricity.
