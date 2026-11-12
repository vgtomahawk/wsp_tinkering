# Hand-written policy for MinAtar Asterix: a dodge-and-collect network on column 5

## Strategy

The player never moves horizontally. It starts in column 5, and the left, right and fire logits are set about 3·30 below the others, so it stays there for the whole episode. Column 5 has two advantages:

- Spawns at columns 0 and 9 can never hit the player in one step.
- Every treasure crossing the board passes through column 5.

Each step, the agent compares three moves: stay, move up, or move down. Each move gets a value for the row it leads to:

**value = −1.5·danger(row) + 0.6·gold(row) + small bonuses**

The bonuses are:
- Gold farther away in the same direction gives a decaying bonus: 0.2, 0.12, 0.08, then 0.05 per row.
- Moving toward the middle rows gives +0.05.
- Entering or staying in an edge row costs a little.

Each row holds at most one entity, so a row with gold is a safe row.

## Layer 1 (24 units used): row-wise features that do not depend on the player

**Units 0–7: player row one-hot P_y** for rows y = 1..8.

**Units 8–15: graded, direction-aware enemy danger for each row**, computed as tanh(4·score − 2). The score depends on how far an approaching enemy is from column 5:

| Enemy position | Score |
|---|---|
| At column 5, or one step away and approaching | 1 |
| Two steps away | 0.5 |
| Three steps away | 0.25 |
| Four steps away | 0.1 |
| Five steps away | 0.05 |
| Moving away | 0 |

Direction comes from the trail channel. Since each row holds at most one entity, a chain of weights (positive on enemy cells, negative on trail cells) gives exactly 0 for receding enemies. Gold in the same row only adds negative trail input, so it is clipped away by the threshold.

**Units 16–23: approaching-gold value for each row**, built the same way. The values are 1, 1, 0.75, 0.5, 0.3, 0.2 by distance, and receding gold scores 0.

## Layer 2 (22 units used): gated evaluators

There is one unit per (action, player row): 8 for stay, 7 for up, 7 for down. Each has pre-activation 10·P_y − 10 + value. This means:
- Only the unit for the current row is "on", and it outputs tanh(value).
- All other units saturate at exactly −1, a constant that the output biases cancel.

The danger and gold activations are rescaled into the range 0..1 inside these weights, using 0.5187·h + 0.5.

## Output

- Each action's logit is 30 times the sum of its units, plus a bias.
- Up from row 1 and down from row 8 automatically come out at the "terrible" level.
- Left, right and fire get a bias of −90, so they are effectively never chosen.

```edits
# ---------- Layer 1: player row one-hot (units 0..7 = rows 1..8)
W1[60:120:6, 0] = 10
W1[120:180:6, 1] = 10
W1[180:240:6, 2] = 10
W1[240:300:6, 3] = 10
W1[300:360:6, 4] = 10
W1[360:420:6, 5] = 10
W1[420:480:6, 6] = 10
W1[480:540:6, 7] = 10
b1[0:8] = -5
# ---------- Layer 1: danger per row (units 8..15 = rows 1..8): enemy ch1, trail ch4
W1[61:120:6, 8] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[64:120:6, 8] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[121:180:6, 9] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[124:180:6, 9] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[181:240:6, 10] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[184:240:6, 10] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[241:300:6, 11] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[244:300:6, 11] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[301:360:6, 12] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[304:360:6, 12] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[361:420:6, 13] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[364:420:6, 13] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[421:480:6, 14] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[424:480:6, 14] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
W1[481:540:6, 15] = [0.2, 0.4, 1.2, 2.4, 5.2, 6.4, 5.2, 2.4, 1.2, 0.4]
W1[484:540:6, 15] = [0, -0.2, -0.4, -1.2, -2.4, -5.2, -2.4, -1.2, -0.4, -0.2]
# ---------- Layer 1: approaching gold per row (units 16..23 = rows 1..8): gold ch2, trail ch4
W1[62:120:6, 16] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[64:120:6, 16] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[122:180:6, 17] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[124:180:6, 17] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[182:240:6, 18] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[184:240:6, 18] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[242:300:6, 19] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[244:300:6, 19] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[302:360:6, 20] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[304:360:6, 20] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[362:420:6, 21] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[364:420:6, 21] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[422:480:6, 22] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[424:480:6, 22] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
W1[482:540:6, 23] = [0.8, 1.2, 2.8, 4.2, 6.8, 8.2, 6.8, 4.2, 2.8, 1.2]
W1[484:540:6, 23] = [0, -0.8, -1.2, -2.8, -4.2, -6.8, -4.2, -2.8, -1.2, -0.8]
b1[8:24] = -2
# ---------- Layer 2: STAY units (0..7 = player rows 1..8)
W2[0, 0] = 10
W2[8, 0] = -0.778
W2[16, 0] = 0.311
b2[0] = -10.5
W2[1, 1] = 10
W2[9, 1] = -0.778
W2[17, 1] = 0.311
b2[1] = -10.45
W2[2, 2] = 10
W2[10, 2] = -0.778
W2[18, 2] = 0.311
b2[2] = -10.45
W2[3, 3] = 10
W2[11, 3] = -0.778
W2[19, 3] = 0.311
b2[3] = -10.45
W2[4, 4] = 10
W2[12, 4] = -0.778
W2[20, 4] = 0.311
b2[4] = -10.45
W2[5, 5] = 10
W2[13, 5] = -0.778
W2[21, 5] = 0.311
b2[5] = -10.45
W2[6, 6] = 10
W2[14, 6] = -0.778
W2[22, 6] = 0.311
b2[6] = -10.45
W2[7, 7] = 10
W2[15, 7] = -0.778
W2[23, 7] = 0.311
b2[7] = -10.5
# ---------- Layer 2: UP units (8..14 = player rows 2..8)
W2[1, 8] = 10
W2[8, 8] = -0.778
W2[16:24, 8] = [0.311, 0, 0, 0, 0, 0, 0, 0]
b2[8] = -10.55
W2[2, 9] = 10
W2[9, 9] = -0.778
W2[16:24, 9] = [0.1037, 0.311, 0, 0, 0, 0, 0, 0]
b2[9] = -10.35
W2[3, 10] = 10
W2[10, 10] = -0.778
W2[16:24, 10] = [0.0622, 0.1037, 0.311, 0, 0, 0, 0, 0]
b2[10] = -10.29
W2[4, 11] = 10
W2[11, 11] = -0.778
W2[16:24, 11] = [0.0415, 0.0622, 0.1037, 0.311, 0, 0, 0, 0]
b2[11] = -10.25
W2[5, 12] = 10
W2[12, 12] = -0.778
W2[16:24, 12] = [0.0259, 0.0415, 0.0622, 0.1037, 0.311, 0, 0, 0]
b2[12] = -10.175
W2[6, 13] = 10
W2[13, 13] = -0.778
W2[16:24, 13] = [0.0259, 0.0259, 0.0415, 0.0622, 0.1037, 0.311, 0, 0]
b2[13] = -10.15
W2[7, 14] = 10
W2[14, 14] = -0.778
W2[16:24, 14] = [0.0259, 0.0259, 0.0259, 0.0415, 0.0622, 0.1037, 0.311, 0]
b2[14] = -10.125
# ---------- Layer 2: DOWN units (15..21 = player rows 1..7)
W2[0, 15] = 10
W2[9, 15] = -0.778
W2[16:24, 15] = [0, 0.311, 0.1037, 0.0622, 0.0415, 0.0259, 0.0259, 0.0259]
b2[15] = -10.125
W2[1, 16] = 10
W2[10, 16] = -0.778
W2[16:24, 16] = [0, 0, 0.311, 0.1037, 0.0622, 0.0415, 0.0259, 0.0259]
b2[16] = -10.15
W2[2, 17] = 10
W2[11, 17] = -0.778
W2[16:24, 17] = [0, 0, 0, 0.311, 0.1037, 0.0622, 0.0415, 0.0259]
b2[17] = -10.175
W2[3, 18] = 10
W2[12, 18] = -0.778
W2[16:24, 18] = [0, 0, 0, 0, 0.311, 0.1037, 0.0622, 0.0415]
b2[18] = -10.25
W2[4, 19] = 10
W2[13, 19] = -0.778
W2[16:24, 19] = [0, 0, 0, 0, 0, 0.311, 0.1037, 0.0622]
b2[19] = -10.29
W2[5, 20] = 10
W2[14, 20] = -0.778
W2[16:24, 20] = [0, 0, 0, 0, 0, 0, 0.311, 0.1037]
b2[20] = -10.35
W2[6, 21] = 10
W2[15, 21] = -0.778
W2[16:24, 21] = [0, 0, 0, 0, 0, 0, 0, 0.311]
b2[21] = -10.55
# ---------- Output
W3[0:8, 0] = 30
W3[8:15, 2] = 30
W3[15:22, 4] = 30
b3[0:6] = [210, -90, 180, -90, 180, -90]
```