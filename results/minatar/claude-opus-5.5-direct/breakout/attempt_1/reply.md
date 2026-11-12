## Strategy

The paddle moves to the column where the ball will land.

- **Ball falling.** Ball moves are diagonal, and a wall bounce makes the ball stay one extra frame in the wall column. So the landing column is a simple fold of a linear function of the ball position:
  - Moving down-right: u = x − y + 9, and landing L = min(u, 19 − u).
  - Moving down-left: u = x + y − 9, and landing L = max(u, −1 − u).
  - The paddle moves right if L > P, left if L < P, and otherwise waits.
- **Ball rising.** The paddle tracks the ball's next column, x + dx.
  - While tracking, the paddle stays within one cell of the ball. That is close enough to reach the landing spot once the ball turns down.
  - The game also catches the ball if the paddle is under the ball's last column before row 9, which gives one cell of slack.
- **Ball just bounced off the paddle** (ball and trail both in row 8). The horizontal direction may have reversed but the trail doesn't show it yet, so the paddle targets the ball's current column.

Every quantity is linear in the one-hot inputs: x, y and dx = x_ball − x_trail, dy = y_ball − y_trail, and paddle P = Σ c·paddle(c). Comparisons use a margin of 0.5 on integers, multiplied by 6, so tanh saturates.

## Layer 1 (15 units)

**Landing comparisons:**

| Unit | Name | Condition |
|---|---|---|
| 2 | c1 | u_dr > P |
| 9 | c2 | 19 − u_dr > P |
| 10 | c3 | u_dr < P |
| 3 | c4 | 19 − u_dr < P |
| 5 | d1 | u_dl > P |
| 0 | d2 | −1 − u_dl > P |
| 1 | d3 | u_dl < P |
| 6 | d4 | −1 − u_dl < P |

**Tracking comparisons:**

| Unit | Name | Condition |
|---|---|---|
| 12 | tA_R | x + dx > P |
| 13 | tA_L | x + dx < P |
| 4 | tB_R | x > P |
| 11 | tB_L | x < P |

**Direction gates:**

| Unit | Name | Condition | Meaning |
|---|---|---|---|
| 7 | G_DR | 2dx + dy + [x=0] − 2[trail row 8] > 1.5 | heading down-right; includes ceiling/brick bounces and the right-wall frame |
| 8 | G_DL | the mirror of G_DR | heading down-left |
| 14 | G_PE | ball and trail both in row 8 | paddle event |

## Layer 2 (8 units, AND/OR logic)

| Unit | Vote | Rule |
|---|---|---|
| R1 | right | G_DR ∧ c1 ∧ c2 |
| L1 | left | G_DR ∧ (c3 ∨ c4) |
| R2 | right | G_DL ∧ (d1 ∨ d2) |
| L2 | left | G_DL ∧ d3 ∧ d4 |
| R3 / L3 | right / left | no gate on ∧ tA_R / tA_L |
| R4 / L4 | right / left | G_PE ∧ tB_R / tB_L |

## Output

- The right logit is 10·ΣR + 30, giving about +10 if any right vote is on and −10 otherwise.
- The left logit is built the same way from the left votes.
- No-op and the other do-nothing actions stay at 0.

```edits
# ---------------- Layer 1: ball channel (3), column part ----------------
W1[3:600:60, 7] = 6
W1[9:600:60, 0:2] = -6
W1[9:600:60, 2:7] = 6
W1[9:600:60, 7:13:5] = 12
W1[9:600:60, 8:14:5] = -12
W1[9:600:60, 9:12] = -6
W1[15:600:60, 0:2] = -12
W1[15:600:60, 2:7] = 12
W1[15:600:60, 7:13:5] = 24
W1[15:600:60, 8:14:5] = -24
W1[15:600:60, 9:12] = -12
W1[21:600:60, 0:2] = -18
W1[21:600:60, 2:7] = 18
W1[21:600:60, 7:13:5] = 36
W1[21:600:60, 8:14:5] = -36
W1[21:600:60, 9:12] = -18
W1[27:600:60, 0:2] = -24
W1[27:600:60, 2:7] = 24
W1[27:600:60, 7:13:5] = 48
W1[27:600:60, 8:14:5] = -48
W1[27:600:60, 9:12] = -24
W1[33:600:60, 0:2] = -30
W1[33:600:60, 2:7] = 30
W1[33:600:60, 7:13:5] = 60
W1[33:600:60, 8:14:5] = -60
W1[33:600:60, 9:12] = -30
W1[39:600:60, 0:2] = -36
W1[39:600:60, 2:7] = 36
W1[39:600:60, 7:13:5] = 72
W1[39:600:60, 8:14:5] = -72
W1[39:600:60, 9:12] = -36
W1[45:600:60, 0:2] = -42
W1[45:600:60, 2:7] = 42
W1[45:600:60, 7:13:5] = 84
W1[45:600:60, 8:14:5] = -84
W1[45:600:60, 9:12] = -42
W1[51:600:60, 0:2] = -48
W1[51:600:60, 2:7] = 48
W1[51:600:60, 7:13:5] = 96
W1[51:600:60, 8:14:5] = -96
W1[51:600:60, 9:12] = -48
W1[57:600:60, 0:2] = -54
W1[57:600:60, 2:7] = 54
W1[57:600:60, 7:13:5] = 108
W1[57:600:60, 8:14:5] = -108
W1[57:600:60, 9:12] = -54
W1[57:600:60, 8] += 6
# ---------------- Layer 1: ball channel, row part ----------------
W1[63:120:6, 0:4] += -6
W1[63:120:6, 5:11] += 6
W1[123:180:6, 0:4] += -12
W1[123:180:6, 5:11] += 12
W1[183:240:6, 0:4] += -18
W1[183:240:6, 5:11] += 18
W1[243:300:6, 0:4] += -24
W1[243:300:6, 5:11] += 24
W1[303:360:6, 0:4] += -30
W1[303:360:6, 5:11] += 30
W1[363:420:6, 0:4] += -36
W1[363:420:6, 5:11] += 36
W1[423:480:6, 0:4] += -42
W1[423:480:6, 5:11] += 42
W1[483:540:6, 0:4] += -48
W1[483:540:6, 5:11] += 48
W1[543:600:6, 0:4] += -54
W1[543:600:6, 5:11] += 54
W1[483:540:6, 14] = 6
# ---------------- Layer 1: trail channel (4) ----------------
W1[10:600:60, 7] = -12
W1[10:600:60, 8] = 12
W1[10:600:60, 12] = -6
W1[10:600:60, 13] = 6
W1[16:600:60, 7] = -24
W1[16:600:60, 8] = 24
W1[16:600:60, 12] = -12
W1[16:600:60, 13] = 12
W1[22:600:60, 7] = -36
W1[22:600:60, 8] = 36
W1[22:600:60, 12] = -18
W1[22:600:60, 13] = 18
W1[28:600:60, 7] = -48
W1[28:600:60, 8] = 48
W1[28:600:60, 12] = -24
W1[28:600:60, 13] = 24
W1[34:600:60, 7] = -60
W1[34:600:60, 8] = 60
W1[34:600:60, 12] = -30
W1[34:600:60, 13] = 30
W1[40:600:60, 7] = -72
W1[40:600:60, 8] = 72
W1[40:600:60, 12] = -36
W1[40:600:60, 13] = 36
W1[46:600:60, 7] = -84
W1[46:600:60, 8] = 84
W1[46:600:60, 12] = -42
W1[46:600:60, 13] = 42
W1[52:600:60, 7] = -96
W1[52:600:60, 8] = 96
W1[52:600:60, 12] = -48
W1[52:600:60, 13] = 48
W1[58:600:60, 7] = -108
W1[58:600:60, 8] = 108
W1[58:600:60, 12] = -54
W1[58:600:60, 13] = 54
W1[64:120:6, 7:9] += -6
W1[124:180:6, 7:9] += -12
W1[184:240:6, 7:9] += -18
W1[244:300:6, 7:9] += -24
W1[304:360:6, 7:9] += -30
W1[364:420:6, 7:9] += -36
W1[424:480:6, 7:9] += -42
W1[484:540:6, 7:9] += -60
W1[544:600:6, 7:9] += -54
W1[484:540:6, 14] = 6
# ---------------- Layer 1: paddle channel (0), row 9 ----------------
W1[540:600:6, 0] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 1] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
W1[540:600:6, 2] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 3] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
W1[540:600:6, 4] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 5] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 6] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
W1[540:600:6, 9] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 10] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
W1[540:600:6, 11] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
W1[540:600:6, 12] = [0, -6, -12, -18, -24, -30, -36, -42, -48, -54]
W1[540:600:6, 13] = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
# ---------------- Layer 1 biases ----------------
b1[0:15] = [45, 51, 51, -63, -3, -57, -51, -9, -9, 57, -57, -3, -3, -3, -9]
# ---------------- Layer 2 ----------------
W2[7, 0] = 3
W2[2, 0] = 3
W2[9, 0] = 3
b2[0] = -6
W2[7, 1] = 3
W2[10, 1] = 3
W2[3, 1] = 3
W2[8, 2] = 3
W2[5, 2] = 3
W2[0, 2] = 3
W2[8, 3] = 3
W2[1, 3] = 3
W2[6, 3] = 3
b2[3] = -6
W2[12, 4] = 3
W2[7, 4] = -3
W2[8, 4] = -3
W2[14, 4] = -3
b2[4] = -9
W2[13, 5] = 3
W2[7, 5] = -3
W2[8, 5] = -3
W2[14, 5] = -3
b2[5] = -9
W2[14, 6] = 3
W2[4, 6] = 3
b2[6] = -3
W2[14, 7] = 3
W2[11, 7] = 3
b2[7] = -3
# ---------------- Output ----------------
W3[0, 3] = 10
W3[2, 3] = 10
W3[4, 3] = 10
W3[6, 3] = 10
W3[1, 1] = 10
W3[3, 1] = 10
W3[5, 1] = 10
W3[7, 1] = 10
b3[1] = 30
b3[3] = 30
```