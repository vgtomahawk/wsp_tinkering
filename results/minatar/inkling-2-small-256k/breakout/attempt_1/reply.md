Design: use two first-layer units as near-linear “column number” scalers, one for the paddle (channel 0) and one for the ball (channel 3), both with small weights so `tanh` stays almost linear but remains sign-correct. The first used hidden-layer-2 unit computes `tanh(50*(paddle_col - ball_col))`, i.e. a saturated sign of “paddle is left/right of the ball”. The output logits make left or right dominant accordingly, keep no-op preferred exactly when the columns match, and make up/down/fire almost impossible. This tracks the ball horizontally; once aligned, the paddle is moved onto the ball’s old cell, so catches go through the `ball_x == pos` vertical-bounce branch and preserve horizontal sweep.

```edits
# R repeated for 10 grid rows: channel 0 cells, then channel 3 cells
W1[0:600:6, 0] = [
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225
]
W1[3:600:6, 1] = [
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225,
 -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225, -0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225
]
# hidden-2 unit 0: saturated sign of paddle column minus ball column
W2[0, 0] = 50
W2[1, 0] = -50
# output: 1 = left, 3 = right; 0 = no-op preferred only exactly on equal columns
W3[0, 1] = 20
W3[0, 3] = -20
b3[0] = 8
b3[1] = 0
b3[2] = -30
b3[3] = 0
b3[4] = -30
b3[5] = -30
```