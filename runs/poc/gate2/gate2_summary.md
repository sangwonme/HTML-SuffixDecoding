# Gate 2: PLD vs DOM-aware draft (runs/poc/gate2)

requests n=10, reps per condition: 3; exact verification; condition order shuffled per sample and rep.

| metric (mean over requests) | PLD | DOM | PLD - DOM [95% CI] |
|---|---|---|---|
| wall-clock total (s) | 6.639 | 6.190 | +0.449 [+0.219, +0.709] |
| target forwards | 103.700 | 96.400 | +7.300 [+4.200, +10.900] |
| tokens / forward | 4.583 | 5.016 | -0.433 [-0.675, -0.220] |
| verify (target fwd) time (s) | 6.378 | 5.907 | +0.470 [+0.241, +0.732] |
| draft proposal time (s) | 0.016 | 0.039 | -0.023 [-0.028, -0.018] |
| output tokens | 442.700 | 445.800 | -3.100 [-28.300, +18.900] |
| TTFT (s) | 0.231 | 0.230 | +0.001 [-0.002, +0.003] |

PLD: draft per step 0.151 ms (steps 3081), draft init (prompt annotate+index) mean 0.0 ms, draft share of wall-clock 0.23%, peak GPU alloc max 6923 MiB, process peak RSS 10121 MiB

DOM: draft per step 0.323 ms (steps 2862), draft init (prompt annotate+index) mean 8.0 ms, draft share of wall-clock 0.63%, peak GPU alloc max 6923 MiB, process peak RSS 10121 MiB

relative wall-clock gain (PLD-DOM)/PLD: +6.94% [+3.50, +10.95]
relative forward reduction: +7.29% [+4.23, +10.89]
output identical PLD vs DOM (rep 0): 5/10; outputs identical across reps: 20/20; run-to-run CV of wall-clock: median 9.21%

**Gate 2 PASSED**: wall-clock PLD-DOM = +0.449 s [+0.219, +0.709] (CI > 0).
