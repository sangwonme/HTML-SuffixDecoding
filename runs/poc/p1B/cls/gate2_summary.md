# Draft-policy wall-clock comparison (runs/poc/p1B/cls)

requests n=50, conditions ['pld', 'dom', 'ranker'], reps 1; exact verification; order shuffled per sample/rep.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total (s) | 6.402 | 6.085 | 5.834 |
| target forwards | 105.880 | 98.940 | 94.820 |
| tokens / forward | 4.506 | 4.821 | 5.111 |
| verify time (s) | 6.146 | 5.804 | 5.545 |
| draft time incl. init (s) | 0.015 | 0.039 | 0.049 |
| output tokens | 428.000 | 424.700 | 421.920 |
| TTFT (s) | 0.228 | 0.229 | 0.228 |

| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |
|---|---|---|---|---|
| DOM - RANKER | +0.251 [+0.068, +0.450] | +4.59% [+2.21, +7.08] | +4.120 [+1.000, +7.500] | +4.63% [+2.05, +7.21] |
| PLD - RANKER | +0.568 [+0.338, +0.809] | +9.15% [+6.33, +11.85] | +11.060 [+7.500, +15.000] | +10.74% [+8.19, +13.26] |
| PLD - DOM | +0.317 [+0.163, +0.498] | +4.67% [+2.85, +6.47] | +6.940 [+4.420, +10.000] | +6.22% [+4.51, +7.94] |

- PLD: draft 0.139 ms/step (steps 5244), init 0.0 ms/request, draft share 0.23%; peak GPU 7020 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- DOM: draft 0.317 ms/step (steps 4897), init 8.2 ms/request, draft share 0.64%; peak GPU 7020 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- RANKER: draft 0.479 ms/step (steps 4691), init 4.2 ms/request, draft share 0.84%; peak GPU 7020 MiB, peak RSS 10122 MiB; outputs identical across reps 50/50; wall-clock CV median nan%
- outputs identical PLD vs DOM (rep 0): 26/50
- outputs identical PLD vs RANKER (rep 0): 28/50
- outputs identical DOM vs RANKER (rep 0): 24/50

**Gate 2 PASSED** (PLD - DOM wall-clock +0.317 s [+0.163, +0.498])
**Gate 2b PASSED** (DOM - RANKER wall-clock +0.251 s [+0.068, +0.450])
