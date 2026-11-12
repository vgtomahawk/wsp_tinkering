Design: hand-code a “dive for divers, shoot what is in my lane, surface only when oxygen is about gone and a diver is aboard” controller. Layer 1 detects (a) player+enemy in the same grid row via high-threshold sums, (b) player+reward in the same row for firing, (c) oxygen/diver gauge cells on the bottom row, (d) danger/reward mass in upper/lower halves and left/right halves, and (e) the bottom submarine row. Layer 2 turns those into a few soft-OR/AND gates: “same-row danger”, “same-row reward”, “must surface”, pass-throughs for direction. The output biases strongly toward diving when safe, toward fire when a reward target shares the row, vertically away from same-row danger, sideways toward reward/danger imbalance, upward only when low oxygen meets diver carried, and suppresses horizontal/down choices then. Left/right actions also set facing for future shots.

```edits
# same-row player+danger detectors (h1 0..8), one per grid row
W1[0:60:6,0] = 10      # row0 player cells
W1[1:60:6,0] = 2       # row0 danger
W1[60:120:6,1] = 10
W1[61:120:6,1] = 2
W1[120:180:6,2] = 10
W1[121:180:6,2] = 2
W1[180:240:6,3] = 10
W1[181:240:6,3] = 2
W1[240:300:6,4] = 10
W1[241:300:6,4] = 2
W1[300:360:6,5] = 10
W1[301:360:6,5] = 2
W1[360:420:6,6] = 10
W1[361:420:6,6] = 2
W1[420:480:6,7] = 10
W1[421:480:6,7] = 2
W1[480:540:6,8] = 10
W1[481:540:6,8] = 2
# same-row player+reward detectors (h1 9..17)
W1[0:60:6,9] = 10
W1[2:60:6,9] = 2
W1[60:120:6,10] = 10
W1[62:120:6,10] = 2
W1[120:180:6,11] = 10
W1[122:180:6,11] = 2
W1[180:240:6,12] = 10
W1[182:240:6,12] = 2
W1[240:300:6,13] = 10
W1[242:300:6,13] = 2
W1[300:360:6,14] = 10
W1[302:360:6,14] = 2
W1[360:420:6,15] = 10
W1[362:420:6,15] = 2
W1[420:480:6,16] = 10
W1[422:480:6,16] = 2
W1[480:540:6,17] = 10
W1[482:540:6,17] = 2
b1[0:18] = -21
# oxygen gauge: bottom-row gauge cells 0..2 are pure oxygen; low means oxygen <~60
W1[545,18] = -1.7
W1[551,18] = -1.7
W1[557,18] = -1.7
b1[18] = 5.0
# diver gauge cells 6..8 are pure diver evidence once oxygen is low; any means at least one diver
W1[581,19] = 2.0
W1[587,19] = 2.0
W1[593,19] = 2.0
b1[19] = -1.0
# global imbalance features, all linear and count-like
W1[1:60:6,21] = 0.8    # danger, rows 0-4
W1[61:120:6,21] = 0.8
W1[121:180:6,21] = 0.8
W1[181:240:6,21] = 0.8
W1[241:300:6,21] = 0.8
W1[301:360:6,22] = 0.8 # danger, rows 5-9
W1[361:420:6,22] = 0.8
W1[421:480:6,22] = 0.8
W1[481:540:6,22] = 0.8
W1[541:600:6,22] = 0.8
W1[2:600:60,23] = 0.8  # reward, columns 0-4
W1[8:600:60,23] = 0.8
W1[14:600:60,23] = 0.8
W1[20:600:60,23] = 0.8
W1[26:600:60,23] = 0.8
W1[32:600:60,24] = 0.8 # reward, columns 5-9
W1[38:600:60,24] = 0.8
W1[44:600:60,24] = 0.8
W1[50:600:60,24] = 0.8
W1[56:600:60,24] = 0.8
W1[1:600:60,25] = 0.8  # danger, columns 0-4
W1[7:600:60,25] = 0.8
W1[13:600:60,25] = 0.8
W1[19:600:60,25] = 0.8
W1[25:600:60,25] = 0.8
W1[31:600:60,26] = 0.8 # danger, columns 5-9
W1[37:600:60,26] = 0.8
W1[43:600:60,26] = 0.8
W1[49:600:60,26] = 0.8
W1[55:600:60,26] = 0.8
W1[2:60:6,27] = 0.8    # reward, rows 0-4
W1[62:120:6,27] = 0.8
W1[122:180:6,27] = 0.8
W1[182:240:6,27] = 0.8
W1[242:300:6,27] = 0.8
W1[302:360:6,28] = 0.8 # reward, rows 5-9
W1[362:420:6,28] = 0.8
W1[422:480:6,28] = 0.8
W1[482:540:6,28] = 0.8
W1[542:600:6,28] = 0.8
# bottom row indicator (front row y=8)
W1[480:540:6,29] = 2.0
b1[29] = -3.2
# layer 2: sharp one-sided OR for any same-row danger/reward, AND for surface, passes otherwise
W2[0:9,0] = 4.0
W2[9:18,1] = 4.0
b2[0:2] = 34.85
W2[18,2] = 4.0        # need-surface if low oxygen and diver aboard
W2[19,2] = 4.0
b2[2] = -5.2
W2[21,3] = 2.0        # pass/sharpen danger-upper
W2[22,4] = 2.0        # pass/sharpen danger-lower
W2[23,5] = 1.8        # pass/sharpen reward-left
W2[24,6] = 1.8        # pass/sharpen reward-right
W2[25,7] = 1.8        # pass/sharpen danger-left
W2[26,8] = 1.8        # pass/sharpen danger-right
W2[27,9] = 1.8        # pass/sharpen reward-upper
W2[28,10] = 1.8       # pass/sharpen reward-lower
W2[8,13] = 6.0        # one-sided “I am on row 8 and in danger”
b2[13] = 4.37
W2[29,15] = 6.0       # one-sided “I am on the bottom playable row”
b2[15] = 5.8
# output layer: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire
b3[0] = -3.2
# left
W3[5,1] = 3.0
W3[7,1] = -3.8
W3[2,1] = -4.5
W3[0,1] = -2.5
b3[1] = -4.0
# up
W3[2,2] = 7.0
W3[0,2] = 2.0
W3[4,2] = 2.6
W3[3,2] = -2.6
W3[9,2] = 1.6
W3[10,2] = -1.2
W3[13,2] = 3.0
W3[15,2] = 2.0
b3[2] = 10.0
# right
W3[6,3] = 3.0
W3[8,3] = -3.8
W3[2,3] = -4.5
W3[0,3] = -2.5
b3[3] = -4.0
# down
W3[2,4] = -6.0
W3[0,4] = 2.0
W3[3,4] = 2.6
W3[4,4] = -2.6
W3[10,4] = 1.4
W3[9,4] = -0.8
W3[13,4] = -4.0
W3[15,4] = -3.5
b3[4] = -7.85
# fire
W3[1,5] = 2.2
W3[0,5] = -2.5
b3[5] = 0.8
```