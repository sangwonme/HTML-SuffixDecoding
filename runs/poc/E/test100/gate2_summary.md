# Draft-policy wall-clock comparison (runs/poc/E/test100)

requests n=100, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 5.282 | 5.173 | 4.863 |
| target forwards | 88.530 | 85.970 | 80.450 |
| tokens / forward | 5.201 | 5.740 | 6.222 |
| verify time (s) | 5.033 | 4.905 | 4.585 |
| draft time incl. init (s) | 0.012 | 0.032 | 0.042 |
| output tokens | 399.180 | 432.830 | 424.340 |
| TTFT (s) | 0.226 | 0.226 | 0.226 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | +0.310 [+0.083, +0.581] | +5.90% [+3.15, +8.72] | +5.520 [+1.570, +10.190] | +6.36% [+3.53, +9.25] |
| PLD - RANKER | +0.419 [+0.149, +0.655] | +9.51% [+5.03, +13.29] | +8.080 [+3.380, +12.190] | +10.87% [+6.23, +14.77] |
| PLD - DOM | +0.109 [-0.179, +0.352] | +2.30% [-2.93, +6.70] | +2.560 [-2.420, +6.760] | +3.21% [-2.21, +7.77] |

- PLD: draft 0.136 ms/step (steps 8753), init 0.0 ms/request, draft share 0.23%; peak GPU 7006 MiB, peak RSS 10122 MiB; outputs identical across reps 100/100; wall-clock CV median nan%
- DOM: draft 0.284 ms/step (steps 8497), init 7.6 ms/request, draft share 0.61%; peak GPU 7006 MiB, peak RSS 10122 MiB; outputs identical across reps 100/100; wall-clock CV median nan%
- RANKER: draft 0.474 ms/step (steps 7945), init 3.9 ms/request, draft share 0.86%; peak GPU 7006 MiB, peak RSS 10122 MiB; outputs identical across reps 100/100; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 52/100
- outputs identical PLD vs RANKER (rep 0): 49/100
- outputs identical DOM vs RANKER (rep 0): 53/100

**Gate 2 NOT passed** (PLD - DOM wall-clock +0.109 s [-0.179, +0.352])
**Gate 2b PASSED** (DOM - RANKER wall-clock +0.310 s [+0.083, +0.581])
