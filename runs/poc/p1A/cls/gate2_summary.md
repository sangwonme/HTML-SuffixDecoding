# Draft-policy wall-clock comparison (runs/poc/p1A/cls)

requests n=50, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 7.042 | 6.462 | 6.333 |
| target forwards | 113.460 | 103.200 | 102.620 |
| tokens / forward | 3.943 | 4.233 | 4.446 |
| verify time (s) | 6.775 | 6.169 | 6.030 |
| draft time incl. init (s) | 0.016 | 0.042 | 0.053 |
| output tokens | 384.420 | 362.320 | 371.860 |
| TTFT (s) | 0.238 | 0.239 | 0.237 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | +0.129 [-0.221, +0.433] | +2.63% [-2.09, +6.64] | +0.580 [-5.481, +5.720] | +0.70% [-5.13, +5.49] |
| PLD - RANKER | +0.710 [+0.270, +1.144] | +9.59% [+4.44, +14.17] | +10.840 [+4.200, +17.660] | +9.38% [+4.54, +13.78] |
| PLD - DOM | +0.581 [+0.208, +1.001] | +6.21% [+1.23, +10.71] | +10.260 [+4.300, +16.920] | +7.19% [+2.16, +11.79] |

- PLD: draft 0.144 ms/step (steps 5623), init 0.0 ms/request, draft share 0.23%; peak GPU 7032 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- DOM: draft 0.324 ms/step (steps 5110), init 8.6 ms/request, draft share 0.64%; peak GPU 7032 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- RANKER: draft 0.478 ms/step (steps 5081), init 4.4 ms/request, draft share 0.84%; peak GPU 7032 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 18/50
- outputs identical PLD vs RANKER (rep 0): 23/50
- outputs identical DOM vs RANKER (rep 0): 17/50

**Gate 2 PASSED** (PLD - DOM wall-clock +0.581 s [+0.208, +1.001])
**Gate 2b NOT passed** (DOM - RANKER wall-clock +0.129 s [-0.221, +0.433])
