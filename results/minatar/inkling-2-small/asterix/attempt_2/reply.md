## Design

**Strategy:** A "greedy treasure hunter with evasive tendencies". At every step, the net computes a noisy gradient: gold pulls toward it (horizontally and vertically), enemies repel away from them, any enemy on screen biases the player into vertical dodging (enemies sweep rows horizontally, so vertical motion is the only reliable dodge), and a weak center-pull stabilizes empty boards. Fire is never chosen; no-op is mildly discouraged. The softmax keeps decisions stochastic, which is fine because interest is in expected returns.

**What the hidden units compute:**
- Units 0–3: *signed distance-difference* features through `tanh` (near-sign functions). Unit 0: `tanh(0.8·(Σ gold col − px))`; unit 1: same for rows; units 2–3: same for enemy instead of gold (using the player-position sum as reference — the only linearly-computable relative feature).
- Units 5–7: enemy-count level, and two center-pull features `tanh(0.45·(4.5 − p))` for x and y.
- Units 8–17: per-row enemy detector `tanh(4·(E_row + player_in_row − 1))` ≈ +1 if enemy shares my row, −1 if row is empty, 0 otherwise.
- Units 18–27: same construction for gold rows.
- Layer 2: units 0–3 sharpen the four difference features (`tanh(2·h)`), unit 4 sums the enemy-row units (`tanh(Σ + 9)` ≈ 0 when no enemies, ~0.76 with enemies on screen, ~1 with an enemy in my row → vertical escape boost), unit 5 does the same for gold (horizontal urgency), units 6–7 pass the center pulls, unit 8 turns enemy count into a no-op suppressor.
- Layer 3: weighted sum → logits. Gold x/y attraction (±2.0–2.5), weak enemy repulsion (1.0 horizontal, 2.5 vertical), strong vertical "dodge" boost when enemies are near (3.5), weak center pull (1.2), small constant tilt toward right/down to counter the edge-clamping bias of the difference features (the player is confined to rows 1–8, and difference features saturate, producing an asymmetric drift), `noop = −1.6`, `fire = −6`.

Gold is chased hard, enemies are avoided harder, fire never happens.

```edits
# ---- Layer 1 ----
# Unit 0: gold-column vs player-column difference: tanh(0.8*(sum gold col - px))
W1[8:600:60, 0] = 0.8
W1[6:600:60, 0] = -0.8
W1[14:600:60, 0] = 1.6
W1[12:600:60, 0] = -1.6
W1[20:600:60, 0] = 2.4
W1[18:600:60, 0] = -2.4
W1[26:600:60, 0] = 3.2
W1[24:600:60, 0] = -3.2
W1[32:600:60, 0] = 4.0
W1[30:600:60, 0] = -4.0
W1[38:600:60, 0] = 4.8
W1[36:600:60, 0] = -4.8
W1[44:600:60, 0] = 5.6
W1[42:600:60, 0] = -5.6
W1[50:600:60, 0] = 6.4
W1[48:600:60, 0] = -6.4
W1[56:600:60, 0] = 7.2
W1[54:600:60, 0] = -7.2
# Unit 1: gold-row vs player-row difference: tanh(0.8*(sum gold row - py))
W1[62:120:6, 1] = 0.8
W1[60:120:6, 1] = -0.8
W1[122:180:6, 1] = 1.6
W1[120:180:6, 1] = -1.6
W1[182:240:6, 1] = 2.4
W1[180:240:6, 1] = -2.4
W1[242:300:6, 1] = 3.2
W1[240:300:6, 1] = -3.2
W1[302:360:6, 1] = 4.0
W1[300:360:6, 1] = -4.0
W1[362:420:6, 1] = 4.8
W1[360:420:6, 1] = -4.8
W1[422:480:6, 1] = 5.6
W1[420:480:6, 1] = -5.6
W1[482:540:6, 1] = 6.4
W1[480:540:6, 1] = -6.4
W1[542:600:6, 1] = 7.2
W1[540:600:6, 1] = -7.2
# Unit 2: enemy-column vs player-column difference: tanh(0.7*(sum enemy col - px))
W1[7:600:60, 2] = 0.7
W1[6:600:60, 2] = -0.7
W1[13:600:60, 2] = 1.4
W1[12:600:60, 2] = -1.4
W1[19:600:60, 2] = 2.1
W1[18:600:60, 2] = -2.1
W1[25:600:60, 2] = 2.8
W1[24:600:60, 2] = -2.8
W1[31:600:60, 2] = 3.5
W1[30:600:60, 2] = -3.5
W1[37:600:60, 2] = 4.2
W1[36:600:60, 2] = -4.2
W1[43:600:60, 2] = 4.9
W1[42:600:60, 2] = -4.9
W1[49:600:60, 2] = 5.6
W1[48:600:60, 2] = -5.6
W1[55:600:60, 2] = 6.3
W1[54:600:60, 2] = -6.3
# Unit 3: enemy-row vs player-row difference: tanh(0.7*(sum enemy row - py))
W1[61:120:6, 3] = 0.7
W1[60:120:6, 3] = -0.7
W1[121:180:6, 3] = 1.4
W1[120:180:6, 3] = -1.4
W1[181:240:6, 3] = 2.1
W1[180:240:6, 3] = -2.1
W1[241:300:6, 3] = 2.8
W1[240:300:6, 3] = -2.8
W1[301:360:6, 3] = 3.5
W1[300:360:6, 3] = -3.5
W1[361:420:6, 3] = 4.2
W1[360:420:6, 3] = -4.2
W1[421:480:6, 3] = 4.9
W1[420:480:6, 3] = -4.9
W1[481:540:6, 3] = 5.6
W1[480:540:6, 3] = -5.6
W1[541:600:6, 3] = 6.3
W1[540:600:6, 3] = -6.3
# Unit 5: count of enemies (danger level): tanh(0.6*count - 0.6)
W1[1:600:6, 5] = 0.6
b1[5] = -0.6
# Unit 6: horizontal center pull: tanh(0.45*(4.5 - px))
W1[6:600:60, 6] = -0.45
W1[12:600:60, 6] = -0.9
W1[18:600:60, 6] = -1.35
W1[24:600:60, 6] = -1.8
W1[30:600:60, 6] = -2.25
W1[36:600:60, 6] = -2.7
W1[42:600:60, 6] = -3.15
W1[48:600:60, 6] = -3.6
W1[54:600:60, 6] = -4.05
b1[6] = 2.025
# Unit 7: vertical center pull: tanh(0.45*(4.5 - py))
W1[60:120:6, 7] = -0.45
W1[120:180:6, 7] = -0.9
W1[180:240:6, 7] = -1.35
W1[240:300:6, 7] = -1.8
W1[300:360:6, 7] = -2.25
W1[360:420:6, 7] = -2.7
W1[420:480:6, 7] = -3.15
W1[480:540:6, 7] = -3.6
W1[540:600:6, 7] = -4.05
b1[7] = 2.025
# Units 8..17: per-row enemy detector tanh(4*(E_row + py_row - 1))
W1[1:60:6, 8] = 4
W1[0:60:6, 8] = 4
b1[8] = -4
W1[61:120:6, 9] = 4
W1[60:120:6, 9] = 4
b1[9] = -4
W1[121:180:6, 10] = 4
W1[120:180:6, 10] = 4
b1[10] = -4
W1[181:240:6, 11] = 4
W1[180:240:6, 11] = 4
b1[11] = -4
W1[241:300:6, 12] = 4
W1[240:300:6, 12] = 4
b1[12] = -4
W1[301:360:6, 13] = 4
W1[300:360:6, 13] = 4
b1[13] = -4
W1[361:420:6, 14] = 4
W1[360:420:6, 14] = 4
b1[14] = -4
W1[421:480:6, 15] = 4
W1[420:480:6, 15] = 4
b1[15] = -4
W1[481:540:6, 16] = 4
W1[480:540:6, 16] = 4
b1[16] = -4
W1[541:600:6, 17] = 4
W1[540:600:6, 17] = 4
b1[17] = -4
# Units 18..27: per-row gold detector tanh(4*(G_row + py_row - 1))
W1[2:60:6, 18] = 4
W1[0:60:6, 18] = 4
b1[18] = -4
W1[62:120:6, 19] = 4
W1[60:120:6, 19] = 4
b1[19] = -4
W1[122:180:6, 20] = 4
W1[120:180:6, 20] = 4
b1[20] = -4
W1[182:240:6, 21] = 4
W1[180:240:6, 21] = 4
b1[21] = -4
W1[242:300:6, 22] = 4
W1[240:300:6, 22] = 4
b1[22] = -4
W1[302:360:6, 23] = 4
W1[300:360:6, 23] = 4
b1[23] = -4
W1[362:420:6, 24] = 4
W1[360:420:6, 24] = 4
b1[24] = -4
W1[422:480:6, 25] = 4
W1[420:480:6, 25] = 4
b1[25] = -4
W1[482:540:6, 26] = 4
W1[480:540:6, 26] = 4
b1[26] = -4
W1[542:600:6, 27] = 4
W1[540:600:6, 27] = 4
b1[27] = -4
# ---- Layer 2 ----
W2[0, 0] = 2
W2[1, 1] = 2
W2[2, 2] = 2
W2[3, 3] = 2
W2[5, 8] = 1.3
W2[6, 6] = 1.3
W2[7, 7] = 1.3
W2[8:18, 4] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
W2[18:28, 5] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
b2[4] = 9
b2[5] = 9
# ---- Layer 3 ----
# gold horizontal attraction: left / right
W3[0, 1] = -2.5
W3[0, 3] = 2.5
# gold vertical attraction: up / down
W3[1, 2] = -2.0
W3[1, 4] = 2.0
# enemy horizontal repulsion (weak): enemy right -> drift left
W3[2, 1] = 1.0
W3[2, 3] = -1.0
# enemy vertical repulsion: enemy below pushes up
W3[3, 2] = 2.5
W3[3, 4] = -2.5
# enemies on screen: boost both vertical moves (rows are swept, escape vertically)
W3[4, 2] = 3.5
W3[4, 4] = 3.5
# gold on screen: boost horizontal moves (bias toward chasing)
W3[5, 1] = 1.5
W3[5, 3] = 1.5
# weak horizontal center pull
W3[6, 1] = -1.2
W3[6, 3] = 1.2
# weak vertical center pull
W3[7, 2] = -1.2
W3[7, 4] = 1.2
# enemy count suppresses standing still
W3[8, 0] = -1.2
# biases: no-op discouraged, fire never used, small right/down tilt
b3[0] = -1.6
b3[1] = -1.4
b3[2] = -0.4
b3[3] = 1.4
b3[4] = 0.4
b3[5] = -6.0
```