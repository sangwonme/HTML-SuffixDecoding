| model | cond | n | in_tok_mean | out_tok_mean | ttft_p50 | ttft_p95 | total_p50 | total_p95 | decode_tps_p50 | peak_reserved_mib_max | hit_max_rate | gpu_shared_rate | render_ok | elem_f1 | type_f1 | text_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E2B | short_zs | 10 | 130 | 1e+03 | 0.0828 | 0.0857 | 61.1 | 75.6 | 17.3 | 6652 | 0 | 0 | 1 | 0.43 | 0.631 | 0.23 |
| E2B | rich_zs | 10 | 193 | 1.27e+03 | 0.111 | 0.131 | 87.8 | 139 | 13.6 | 6660 | 0.1 | 0 | 1 | 0.648 | 0.719 | 0.93 |
| E4B | rich_zs | 10 | 193 | 1.2e+03 | 0.65 | 0.747 | 108 | 178 | 10.7 | 9252 | 0.1 | 0 | 1 | 0.733 | 0.778 | 0.91 |
| 12B | rich_zs | 10 | 197 | 1.5e+03 | 0.214 | 0.253 | 142 | 195 | 10.3 | 8400 | 0.2 | 0 | 1 | 0.639 | 0.739 | 0.883 |
| E2B | rich_fs | 10 | 1.76e+03 | 716 | 0.273 | 0.373 | 24.4 | 94.3 | 17.7 | 7094 | 0 | 0 | 1 | 0.734 | 0.83 | 1 |
| E4B | rich_fs | 10 | 1.76e+03 | 541 | 10.8 | 16.3 | 45.4 | 90.6 | 12.7 | 9726 | 0 | 0.2 | 1 | 0.831 | 0.874 | 0.993 |
| 12B | rich_fs | 10 | 1.76e+03 | 1.06e+03 | 0.983 | 1.47 | 78.2 | 138 | 12.5 | 9384 | 0.1 | 0 | 1 | 0.817 | 0.819 | 0.96 |
| B1_template | - | 10 |  |  |  |  |  |  |  |  |  |  | 1 | 0.493 | 0.671 | 0.0111 |
| B2_retrieval | - | 10 |  |  |  |  |  |  |  |  |  |  | 1 | 0.631 | 0.78 | 0.326 |

Notes: pilot first 10 rows (same ids for every row). n=10, so p95 is effectively the max.
Gemma 4 QAT checkpoints loaded as bitsandbytes NF4 (approximation of q4_0), greedy, max_new_tokens 2048,
concurrency 1, 3 warm-up runs (64 tokens). rich_fs = query_rich + 3 TF-IDF-retrieved train examples.
gpu_shared = another process at >=10% SM before the sample. Baselines B1/B2 scored on the same 10 ids.
