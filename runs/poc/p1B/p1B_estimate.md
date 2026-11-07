# P1 condition B: ESTIMATED wall-clock (simulated forwards x measured per-forward and draft cost)

draft cost per step (gate2b measured): pld 0.158 ms (+init 0.0 ms), dom 0.335 ms (+init 9.1 ms), ranker 0.529 ms (+init 4.6 ms)

Validation on original source (estimate vs gate2b measured mean wall-clock):
- PLD: estimate 6.514 s vs measured 6.528 s (ratio 0.998; note: sim target = PLD output for all policies)
- DOM: estimate 6.116 s vs measured 6.119 s (ratio 0.999; note: sim target = PLD output for all policies)
- RANKER: estimate 5.855 s vs measured 5.826 s (ratio 1.005; note: sim target = PLD output for all policies)

| source | PLD est. (s) | DOM est. (s) | RANKER est. (s) | DOM - RANKER s [95% CI] | % | PLD - RANKER % [95% CI] | PLD - DOM % [95% CI] |
|---|---|---|---|---|---|---|---|
| orig | 6.51 | 6.12 | 5.86 | +0.260 [+0.210, +0.312] | +5.22% | +11.36% [+9.79, +13.00] | +6.50% [+5.37, +7.67] |
| fmt | 9.43 | 8.85 | 8.62 | +0.230 [+0.156, +0.307] | +2.75% | +9.16% [+7.97, +10.36] | +6.58% [+5.59, +7.57] |
| css | 13.45 | 12.78 | 12.68 | +0.100 [+0.049, +0.150] | +0.75% | +5.70% [+5.11, +6.29] | +4.98% [+4.45, +5.50] |
| dom | 6.63 | 6.29 | 6.06 | +0.232 [+0.187, +0.274] | +4.44% | +9.43% [+7.98, +10.94] | +5.26% [+4.21, +6.37] |
| cls | 7.21 | 6.78 | 6.53 | +0.249 [+0.198, +0.303] | +4.34% | +10.17% [+8.87, +11.49] | +6.08% [+5.04, +7.17] |

All condition-B wall-clock numbers are ESTIMATES (no GPU run).
