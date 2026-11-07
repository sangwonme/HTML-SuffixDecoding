# Draft-policy wall-clock comparison (runs/poc/p1A/dom)

requests n=50, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 7.066 | 6.567 | 6.335 |
| target forwards | 102.900 | 94.460 | 92.100 |
| tokens / forward | 5.438 | 5.891 | 6.110 |
| verify time (s) | 6.771 | 6.244 | 5.995 |
| draft time incl. init (s) | 0.017 | 0.048 | 0.063 |
| output tokens | 526.520 | 518.820 | 497.220 |
| TTFT (s) | 0.261 | 0.259 | 0.261 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | +0.232 [-0.206, +0.710] | +3.69% [-0.92, +8.32] | +2.360 [-4.020, +9.140] | +3.16% [-1.62, +7.83] |
| PLD - RANKER | +0.731 [+0.303, +1.188] | +10.79% [+6.51, +15.06] | +10.800 [+4.540, +17.560] | +11.41% [+7.08, +15.81] |
| PLD - DOM | +0.498 [+0.272, +0.745] | +6.88% [+4.19, +9.56] | +8.440 [+5.160, +12.080] | +8.04% [+5.52, +10.66] |

- PLD: draft 0.170 ms/step (steps 5095), init 0.0 ms/request, draft share 0.24%; peak GPU 7074 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- DOM: draft 0.405 ms/step (steps 4673), init 9.9 ms/request, draft share 0.73%; peak GPU 7074 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- RANKER: draft 0.640 ms/step (steps 4555), init 5.0 ms/request, draft share 1.00%; peak GPU 7074 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 28/50
- outputs identical PLD vs RANKER (rep 0): 24/50
- outputs identical DOM vs RANKER (rep 0): 31/50

**Gate 2 PASSED** (PLD - DOM wall-clock +0.498 s [+0.272, +0.745])
**Gate 2b NOT passed** (DOM - RANKER wall-clock +0.232 s [-0.206, +0.710])
