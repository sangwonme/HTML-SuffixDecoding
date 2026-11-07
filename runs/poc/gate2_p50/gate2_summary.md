# Gate 2: PLD vs DOM-aware draft (runs/poc/gate2_p50)

requests n=50, reps per condition: 3; exact verification; condition order shuffled per sample and rep.

| metric (mean over requests) | PLD | DOM | PLD - DOM [95% CI] |
|---|---|---|---|
| wall-clock total (s) | 5.850 | 5.480 | +0.370 [+0.181, +0.572] |
| target forwards | 96.860 | 90.000 | +6.860 [+3.600, +10.240] |
| tokens / forward | 4.983 | 5.521 | -0.538 [-0.709, -0.383] |
| verify (target fwd) time (s) | 5.597 | 5.207 | +0.390 [+0.202, +0.591] |
| draft proposal time (s) | 0.013 | 0.034 | -0.021 [-0.024, -0.018] |
| output tokens | 422.520 | 428.420 | -5.900 [-17.101, +4.580] |
| TTFT (s) | 0.228 | 0.228 | -0.000 [-0.001, +0.001] |

PLD: draft per step 0.138 ms (steps 14379), draft init (prompt annotate+index) mean 0.0 ms, draft share of wall-clock 0.23%, peak GPU alloc max 7020 MiB, process peak RSS 10122 MiB

DOM: draft per step 0.297 ms (steps 13350), draft init (prompt annotate+index) mean 7.7 ms, draft share of wall-clock 0.62%, peak GPU alloc max 7020 MiB, process peak RSS 10122 MiB

relative wall-clock gain (PLD-DOM)/PLD: +6.95% [+4.06, +10.00]
relative forward reduction: +7.77% [+4.76, +10.93]
output identical PLD vs DOM (rep 0): 22/50; outputs identical across reps: 100/100; run-to-run CV of wall-clock: median 1.75%

**Gate 2 PASSED**: wall-clock PLD-DOM = +0.370 s [+0.181, +0.572] (CI > 0).
