Design: hand-code a “steer toward reward, away from danger, fire when contacts exist, surface when oxygen is nearly empty and a diver is aboard” policy. Hidden unit 0–2 decode the oxygen/diver bars, units 3–6 are linear relative-position scores for rewards/dangers versus the submarine (row sum and column sum), unit 7 detects any danger/enemy, unit 8 any reward/divers, units 9–10 locate the submarine row/column for edge/bottom control. Layer 2 mostly copies/sharpens these and forms two AND gates: low oxygen AND diver present, and low oxygen AND many divers. Output biases favor moving down early, firing are driven by danger/reward presence, and surfacing gets a large up-bias only from the safe-surface AND.

```edits
# gauge decode: low oxygen (left bar empty), diver present/many (right bar, valid when oxygen is low)
W1[545,0] = -0.85; W1[551,0] = -0.85; W1[557,0] = -0.85; b1[0] = 1.4
W1[581,1] = 1.8; W1[587,1] = 1.8; W1[593,1] = 1.8; b1[1] = -0.9
W1[569,2] = 1.3; W1[575,2] = 1.3; W1[581,2] = 1.3; b1[2] = -1.5

# relative column scores: reward right(+), danger right(+), minus 0.5*player column sum
W1[8:600:60,3] = 0.5;  W1[14:600:60,3] = 1.0; W1[20:600:60,3] = 1.5; W1[26:600:60,3] = 2.0; W1[32:600:60,3] = 2.5
W1[38:600:60,3] = 3.0; W1[44:600:60,3] = 3.5; W1[50:600:60,3] = 4.0; W1[56:600:60,3] = 4.5
W1[6:600:60,3] = -0.25; W1[12:600:60,3] = -0.5; W1[18:600:60,3] = -0.75; W1[24:600:60,3] = -1.0; W1[30:600:60,3] = -1.25
W1[36:600:60,3] = -1.5; W1[42:600:60,3] = -1.75; W1[48:600:60,3] = -2.0; W1[54:600:60,3] = -2.25
W1[7:600:60,4] = 0.5;  W1[13:600:60,4] = 1.0; W1[19:600:60,4] = 1.5; W1[25:600:60,4] = 2.0; W1[31:600:60,4] = 2.5
W1[37:600:60,4] = 3.0; W1[43:600:60,4] = 3.5; W1[49:600:60,4] = 4.0; W1[55:600:60,4] = 4.5
W1[6:600:60,4] = -0.25; W1[12:600:60,4] = -0.5; W1[18:600:60,4] = -0.75; W1[24:600:60,4] = -1.0; W1[30:600:60,4] = -1.25
W1[36:600:60,4] = -1.5; W1[42:600:60,4] = -1.75; W1[48:600:60,4] = -2.0; W1[54:600:60,4] = -2.25

# relative row scores: reward below(+), danger below(+), minus 0.5*player row sum
W1[62:120:6,5] = 0.5;  W1[122:180:6,5] = 1.0; W1[182:240:6,5] = 1.5; W1[242:300:6,5] = 2.0; W1[302:360:6,5] = 2.5
W1[362:420:6,5] = 3.0; W1[422:480:6,5] = 3.5; W1[482:540:6,5] = 4.0; W1[542:600:6,5] = 4.5
W1[60:120:6,5] = -0.25; W1[120:180:6,5] = -0.5; W1[180:240:6,5] = -0.75; W1[240:300:6,5] = -1.0; W1[300:360:6,5] = -1.25
W1[360:420:6,5] = -1.5; W1[420:480:6,5] = -1.75; W1[480:540:6,5] = -2.0; W1[540:600:6,5] = -2.25
W1[61:120:6,6] = 0.5;  W1[121:180:6,6] = 1.0; W1[181:240:6,6] = 1.5; W1[241:300:6,6] = 2.0; W1[301:360:6,6] = 2.5
W1[361:420:6,6] = 3.0; W1[421:480:6,6] = 3.5; W1[481:540:6,6] = 4.0; W1[541:600:6,6] = 4.5
W1[60:120:6,6] = -0.25; W1[120:180:6,6] = -0.5; W1[180:240:6,6] = -0.75; W1[240:300:6,6] = -1.0; W1[300:360:6,6] = -1.25
W1[360:420:6,6] = -1.5; W1[420:480:6,6] = -1.75; W1[480:540:6,6] = -2.0; W1[540:600:6,6] = -2.25

# simple counts: any danger/enemy, any reward
W1[1:600:6,7] = 0.8; b1[7] = -0.4
W1[2:600:6,8] = 0.8; b1[8] = -0.4

# submarine location for edge/bottom control
W1[60:120:6,9] = 0.22; W1[120:180:6,9] = 0.44; W1[180:240:6,9] = 0.66; W1[240:300:6,9] = 0.88; W1[300:360:6,9] = 1.10
W1[360:420:6,9] = 1.32; W1[420:480:6,9] = 1.54; W1[480:540:6,9] = 1.76; W1[540:600:6,9] = 1.98; b1[9] = -2.0
W1[6:600:60,10] = 0.22; W1[12:600:60,10] = 0.44; W1[18:600:60,10] = 0.66; W1[24:600:60,10] = 0.88; W1[30:600:60,10] = 1.10
W1[36:600:60,10] = 1.32; W1[42:600:60,10] = 1.54; W1[48:600:60,10] = 1.76; W1[54:600:60,10] = 1.98; b1[10] = -2.0

# layer 2: mostly sharpened copies, plus surface AND gates
W2[0,0] = 1.0; W2[1,1] = 1.0; W2[2,2] = 1.0
W2[3,3] = 1.8; W2[4,4] = 1.8; W2[5,5] = 1.8; W2[6,6] = 1.8
W2[7,7] = 1.5; W2[8,8] = 1.5; W2[9,9] = 1.0; W2[10,10] = 1.0
W2[0,14] = 2.2; W2[1,14] = 2.2; b2[14] = -2.2   # oxygen low AND diver present
W2[0,15] = 2.0; W2[2,15] = 2.0; b2[15] = -2.0   # oxygen low AND many divers

# outputs: 0 noop, 1 left, 2 up, 3 right, 4 down, 5 fire
b3[0] = -0.2; b3[4] = 1.6; b3[5] = -0.8
W3[3,1] = -2.4; W3[3,3] = 2.4    # rewards to the right pull right
W3[4,1] = 2.6;  W3[4,3] = -2.6   # danger to the right pushes left
W3[5,2] = -2.5; W3[5,4] = 2.5    # rewards below pull down
W3[6,2] = 1.5;  W3[6,4] = -1.5   # danger below nudges up
W3[7,2] = 2.3;  W3[7,4] = 2.3;  W3[7,5] = 3.2   # vertical dodge bias + shoot on contacts
W3[8,4] = 1.4;  W3[8,5] = 1.0    # reward presence favors descending/firing
W3[9,2] = 0.9;  W3[9,4] = -0.9   # do not press down at the bottom row
W3[10,1] = 0.8; W3[10,3] = -0.8  # stay off extreme sides
W3[14,2] = 7.0; W3[14,4] = -2.5  # must surface when low oxygen and any diver aboard
W3[15,2] = 2.5; W3[15,4] = -1.0  # extra urgency for a big delivery