# Draft-policy wall-clock comparison (runs/poc/p1A/fmt)

requests n=50, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 17.174 | 15.358 | 16.118 |
| target forwards | 274.680 | 241.380 | 247.980 |
| tokens / forward | 2.432 | 2.799 | 2.560 |
| verify time (s) | 16.817 | 14.965 | 15.687 |
| draft time incl. init (s) | 0.044 | 0.082 | 0.117 |
| output tokens | 655.380 | 651.800 | 621.180 |
| TTFT (s) | 0.273 | 0.274 | 0.273 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | -0.760 [-1.608, +0.113] | -11.45% [-27.82, -1.10] | -6.600 [-18.980, +5.641] | -8.86% [-25.41, +0.98] |
| PLD - RANKER | +1.056 [-0.214, +2.906] | +2.88% [-1.17, +7.64] | +26.700 [+8.820, +54.540] | +6.83% [+3.59, +10.94] |
| PLD - DOM | +1.816 [+0.324, +3.776] | +6.41% [+0.37, +12.66] | +33.300 [+11.320, +62.982] | +8.33% [+2.83, +14.09] |

- PLD: draft 0.159 ms/step (steps 13684), init 0.0 ms/request, draft share 0.25%; peak GPU 7096 MiB, peak RSS 10121 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- DOM: draft 0.289 ms/step (steps 12019), init 12.6 ms/request, draft share 0.53%; peak GPU 7096 MiB, peak RSS 10121 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- RANKER: draft 0.454 ms/step (steps 12349), init 4.7 ms/request, draft share 0.73%; peak GPU 7096 MiB, peak RSS 10121 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 18/50
- outputs identical PLD vs RANKER (rep 0): 19/50
- outputs identical DOM vs RANKER (rep 0): 14/50

**Gate 2 PASSED** (PLD - DOM wall-clock +1.816 s [+0.324, +3.776])
**Gate 2b NOT passed** (DOM - RANKER wall-clock -0.760 s [-1.608, +0.113])
