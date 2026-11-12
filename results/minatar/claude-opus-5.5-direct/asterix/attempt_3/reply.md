## Design

**Strategy.** The player stays at column 5 and dodges only vertically. Enemies only move horizontally, and each row holds at most one entity. So the decision reduces to: "is my row about to be hit at column 5, and is the row above or below safer or more rewarding?" A row with treasure contains no enemy, so treasure rows are safe and attractive.

**Layer 1 (26 units).**
- **Units 0–7: danger per row r = 1..8.** Each computes D_r = tanh(Σ wE(c)·enemy(r,c) + Σ wT(c)·trail(r,c) − 1.2).
  - Enemy weights grow toward column 5.
  - Trail weights are negative and strongest near the centre. A trail between the enemy and column 5 means the enemy is moving away, so it cancels the enemy term.
  - Because each row has only one entity, the linear combination separates the cases:
    - an enemy at column 5: ≈ +0.98
    - an enemy one step away and approaching: +0.83
    - two steps away and approaching: +0.38
    - three steps away: −0.38
    - far, receding or empty: about −0.8 to −1
- **Units 8–15: treasure value per row.** These use the same weights, applied to the treasure channel.
- **Units 16–23: player-row indicators.** Each is ±1, from the player channel summed over the row.
- **Unit 24: player is right of column 5. Unit 25: player is left of column 5.**

**Layer 2 (16 units).**
- **Units 0–6: "move up" from row y = 2..8.** Each computes tanh(6·Prow_y − 6.6 + 2(D_y − D_{y−1}) + 1.5(G_{y−1} − G_y)). It sits saturated at −1 unless the player is in row y. When active, it measures how much better the row above is than staying.
- **Units 7–13: "move down" from row y = 1..7.** These mirror the up units.
- **Units 14–15:** pass the left/right recentring signals through.

**Output.**
- Up logit = 8·Σ(up units) + 48; down logit works the same way. The inactive units contribute exactly −1 each, which the bias cancels. In row 1 the up logit is −8, and in row 8 the down logit is −8.
- Left and right logits are about +4.7 when the player is off-centre and −4.7 otherwise.
- No-op logit is 0, so the player stays when both neighbouring rows are no better.
- Fire is suppressed.

```edits
# ---------- Layer 1: danger units 0-7 (rows 1..8), gold units 8-15 ----------
W1[61:120:6, 0] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[64:120:6, 0] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[62:120:6, 8] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[64:120:6, 8] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[121:180:6, 1] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[124:180:6, 1] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[122:180:6, 9] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[124:180:6, 9] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[181:240:6, 2] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[184:240:6, 2] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[182:240:6, 10] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[184:240:6, 10] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[241:300:6, 3] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[244:300:6, 3] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[242:300:6, 11] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[244:300:6, 11] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[301:360:6, 4] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[304:360:6, 4] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[302:360:6, 12] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[304:360:6, 12] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[361:420:6, 5] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[364:420:6, 5] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[362:420:6, 13] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[364:420:6, 13] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[421:480:6, 6] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[424:480:6, 6] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[422:480:6, 14] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[424:480:6, 14] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[481:540:6, 7] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[484:540:6, 7] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
W1[482:540:6, 15] = [0.2, 0.6, 1.0, 2.4, 3.6, 6.0, 3.6, 2.4, 1.0, 0.6]
W1[484:540:6, 15] = [-0.2, -0.2, -0.8, -1.2, -2.4, -4.0, -2.4, -1.2, -0.8, -0.2]
b1[0:16] = -1.2
# ---------- player row indicators, units 16-23 (rows 1..8) ----------
W1[60:120:6, 16] = 6
W1[120:180:6, 17] = 6
W1[180:240:6, 18] = 6
W1[240:300:6, 19] = 6
W1[300:360:6, 20] = 6
W1[360:420:6, 21] = 6
W1[420:480:6, 22] = 6
W1[480:540:6, 23] = 6
b1[16:24] = -3
# ---------- horizontal centering: unit 24 = player right of col 5, unit 25 = left of col 5 ----------
W1[36:600:60, 24] = 3
W1[42:600:60, 24] = 3
W1[48:600:60, 24] = 3
W1[54:600:60, 24] = 3
W1[0:600:60, 25] = 3
W1[6:600:60, 25] = 3
W1[12:600:60, 25] = 3
W1[18:600:60, 25] = 3
W1[24:600:60, 25] = 3
b1[24] = -1.5
b1[25] = -1.5
# ---------- Layer 2: up units 0-6 (player row y=2..8) ----------
W2[0:8, 0:7] = [[-2, 0, 0, 0, 0, 0, 0], [2, -2, 0, 0, 0, 0, 0], [0, 2, -2, 0, 0, 0, 0], [0, 0, 2, -2, 0, 0, 0], [0, 0, 0, 2, -2, 0, 0], [0, 0, 0, 0, 2, -2, 0], [0, 0, 0, 0, 0, 2, -2], [0, 0, 0, 0, 0, 0, 2]]
W2[8:16, 0:7] = [[1.5, 0, 0, 0, 0, 0, 0], [-1.5, 1.5, 0, 0, 0, 0, 0], [0, -1.5, 1.5, 0, 0, 0, 0], [0, 0, -1.5, 1.5, 0, 0, 0], [0, 0, 0, -1.5, 1.5, 0, 0], [0, 0, 0, 0, -1.5, 1.5, 0], [0, 0, 0, 0, 0, -1.5, 1.5], [0, 0, 0, 0, 0, 0, -1.5]]
W2[17, 0] = 6
W2[18, 1] = 6
W2[19, 2] = 6
W2[20, 3] = 6
W2[21, 4] = 6
W2[22, 5] = 6
W2[23, 6] = 6
# ---------- down units 7-13 (player row y=1..7) ----------
W2[0:8, 7:14] = [[2, 0, 0, 0, 0, 0, 0], [-2, 2, 0, 0, 0, 0, 0], [0, -2, 2, 0, 0, 0, 0], [0, 0, -2, 2, 0, 0, 0], [0, 0, 0, -2, 2, 0, 0], [0, 0, 0, 0, -2, 2, 0], [0, 0, 0, 0, 0, -2, 2], [0, 0, 0, 0, 0, 0, -2]]
W2[8:16, 7:14] = [[-1.5, 0, 0, 0, 0, 0, 0], [1.5, -1.5, 0, 0, 0, 0, 0], [0, 1.5, -1.5, 0, 0, 0, 0], [0, 0, 1.5, -1.5, 0, 0, 0], [0, 0, 0, 1.5, -1.5, 0, 0], [0, 0, 0, 0, 1.5, -1.5, 0], [0, 0, 0, 0, 0, 1.5, -1.5], [0, 0, 0, 0, 0, 0, 1.5]]
W2[16, 7] = 6
W2[17, 8] = 6
W2[18, 9] = 6
W2[19, 10] = 6
W2[20, 11] = 6
W2[21, 12] = 6
W2[22, 13] = 6
b2[0:14] = -6.6
# ---------- centering pass-through ----------
W2[24, 14] = 2
W2[25, 15] = 2
# ---------- Output ----------
W3[0:7, 2] = 8
W3[7:14, 4] = 8
b3[2] = 48
b3[4] = 48
W3[14, 1] = 5
W3[15, 3] = 5
b3[5] = -10
```