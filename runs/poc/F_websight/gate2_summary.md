# Draft-policy wall-clock comparison (runs/poc/F_websight)

requests n=50, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 39.788 | 38.275 | 38.281 |
| target forwards | 683.880 | 651.300 | 649.480 |
| tokens / forward | 2.063 | 2.160 | 2.151 |
| verify time (s) | 39.357 | 37.698 | 37.626 |
| draft time incl. init (s) | 0.103 | 0.245 | 0.320 |
| output tokens | 1420.020 | 1412.180 | 1401.820 |
| TTFT (s) | 0.238 | 0.238 | 0.238 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | -0.005 [-1.073, +1.089] | -0.53% [-3.35, +2.22] | +1.820 [-16.180, +20.160] | -0.20% [-3.02, +2.54] |
| PLD - RANKER | +1.507 [+0.491, +2.499] | +3.50% [+0.85, +5.97] | +34.400 [+17.160, +51.240] | +4.76% [+2.15, +7.18] |
| PLD - DOM | +1.512 [+0.143, +2.805] | +3.22% [-0.24, +6.46] | +32.580 [+9.119, +54.820] | +4.17% [+0.71, +7.38] |

- PLD: draft 0.151 ms/step (steps 34144), init 0.0 ms/request, draft share 0.26%; peak GPU 6898 MiB, peak RSS 10042 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- DOM: draft 0.363 ms/step (steps 32515), init 8.4 ms/request, draft share 0.64%; peak GPU 6898 MiB, peak RSS 10042 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- RANKER: draft 0.487 ms/step (steps 32424), init 4.0 ms/request, draft share 0.84%; peak GPU 6898 MiB, peak RSS 10042 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 1/50
- outputs identical PLD vs RANKER (rep 0): 2/50
- outputs identical DOM vs RANKER (rep 0): 1/50

**Gate 2 PASSED** (PLD - DOM wall-clock +1.512 s [+0.143, +2.805])
**Gate 2b NOT passed** (DOM - RANKER wall-clock -0.005 s [-1.073, +1.089])
