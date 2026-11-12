## Design

**Strategy.** The submarine stays near columns 5–6 and only moves up and down. Every enemy and diver crosses the whole row, so they all come to it. In order of priority, the policy does the following:

- **Dodge:** move up or down when an enemy bullet, or an enemy it can't shoot in time, is close in its own row. It never moves into a row that has danger near column 5–6.
- **Move to targets:** go to rows with divers (weighted most), or with enemies in nearby rows.
- **Stay put:** remain in the current row while a diver or a far-off enemy is in it.
- **Face the threat:** turn toward an enemy that is far away behind it. A turn from (front 5, facing left) to (front 6, facing right) keeps the submarine on the same two cells.
- **Fire:** the default action whenever nothing else applies.
- **Surface:** when oxygen drops below 40, it heads up. Going up from row 1 to the surface is only allowed if at least one diver is on board; this check works because, when oxygen is below 40, gauge cell (9,8) can only be lit by the diver bar.
- **Recentre:** if a sticky action shifts it out of columns 5–6, it steps back.

**Layer 1 (30 units).** Each unit for row r is gated by "submarine in row r" (weight +15 per player cell, bias −30). An inactive row's unit sits at exactly −1, so layer 2 can subtract a known constant.

- **T_r (units 0–7), turn signal.** Computed as 0.6·(enemies in columns 8–9 − enemies in columns 0–3) − 0.6·orientation. Orientation comes from where the submarine's back cell is in the trail channel. The result is about ±1.2 when the submarine should turn, and 0 or ±0.6 otherwise.
- **V_r (units 8–15), vertical preference (positive means down).**
  - Pull toward rows with divers or enemies: +1.2 per cell for the adjacent row, +0.72 for farther rows.
  - Strong blocking for danger near columns 3–8 in the adjacent rows (up to ±6.5).
  - A dodge bias of +1 for danger near the submarine in its own row (−1 in row 8, where it can't go lower).
  - An upward push of −3 when oxygen is below 40.
- **S_r (units 16–23), stay versus urgency.**
  - Raised by a diver in the row (+2), a far-off enemy (net +1), or danger in an adjacent row (+2 per cell).
  - Lowered sharply by a close bullet or enemy in the row (−4 per cell), and by low oxygen.
- **Single units:**
  - Z0: submarine in row 0.
  - R0: in row 0 and row 1 is clear near columns 3–8.
  - R1: in row 1.
  - SURF: diver gauge cell lit and oxygen below 40.
  - CL / CR: submarine has drifted left or right of columns 5–6.

**Layer 2.**
- **D2 / U2 (down / up):** fire when V exceeds a threshold of 0.05 + 0.6·(S + 1). Z0 corrects the inactive-unit offset so that everything reads neutral in row 0.
- **PR2 / PL2:** turn right or left when the active T value exceeds ±0.69.
- **D0:** dive from the surface when R0 says row 1 is clear.
- **CL, CR:** passed straight through for recentring.
- **NoUp:** active when R1 is on and SURF is off; it blocks moving up from row 1.

**Output.** Firing has a baseline logit of 6. Active moves get logits of about 10–12, and inactive actions sit near 0.

```edits
# 