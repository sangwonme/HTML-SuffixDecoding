# P1-A cause analysis (RANKER trajectory replay on measured RANKER outputs)

| set | steps | n-gram hit rate | hit n=3 | mean #cand | multi-cand share | acc MLP | acc rule (same state) | MLP<rule | MLP>rule | MLP != top1 | oracle gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orig | 4223 | 0.786 | 0.299 | 3.27 | 0.540 | 3.99 | 3.50 | 0.048 | 0.119 | 0.457 | 0.64 |
| fmt | 12349 | 0.848 | 0.267 | 4.00 | 0.649 | 1.51 | 1.62 | 0.094 | 0.068 | 0.458 | 0.84 |

Structural feature means over candidates at multi-candidate steps (shift vs orig):

| feature | orig | fmt |
|---|---|---|
| dom_score | 6.872 | 6.081 |
| dom_rank | 2.302 | 2.431 |
| lex_eq | 0.848 | 0.741 |
| key_eq | 0.098 | 0.118 |
| path_suf | 1.656 | 1.208 |
| parent_eq | 0.610 | 0.639 |
| depth_cur | 3.677 | 2.433 |
| depth_cand | 3.429 | 2.677 |
| depth_diff | 0.637 | 0.679 |
| sib_diff | 0.966 | 0.593 |
| event_eq | 0.767 | 0.795 |
| relpos_diff | 0.151 | 0.160 |

## fmt: steps where MLP choice < DOM-rule choice (n=400), lexical state counts: {"attrname": 58, "attrval": 13, "cssval": 96, "cssprop": 140, "text": 72, "tagname": 5, "boundary": 5, "none": 8, "csssel": 3}

- `s2w_8482` pos 12 lex=attrname a_mlp=0 a_rule=16 | ctx `'```html\n<!DOCTYPE html>\n<html\n  '` | truth `'<head>\n    <'` | rule `'<head>\n    <'` | mlp `'</body>\n</html'`
- `s2w_8482` pos 13 lex=attrname a_mlp=0 a_rule=16 | ctx `'html\n<!DOCTYPE html>\n<html\n  <'` | truth `'head>\n    <meta'` | rule `'head>\n    <meta'` | mlp `'body>\n    <div'`
- `s2w_8482` pos 33 lex=attrval a_mlp=0 a_rule=1 | ctx `'utf-8">\n    <meta\n      name="'` | truth `'viewport"\n      content=\''` | rule `'viewport"><title>page displaying'` | mlp `'utf-8">\n    '`
- `s2w_8482` pos 74 lex=cssval a_mlp=0 a_rule=1 | ctx `'\n      * {\n        box-sizing: border-'` | truth `'box;\n      }\n'` | rule `'box}body{margin:'` | mlp `'block;width:24'`
- `s2w_8482` pos 101 lex=cssprop a_mlp=0 a_rule=1 | ctx `';\n        width: 360px;\n'` | truth `'        font: 14'` | rule `'        width: 36'` | mlp `'      }\n      body {'`
- `s2w_8482` pos 130 lex=cssprop a_mlp=0 a_rule=1 | ctx `'}\n      .row {\n        display: flex;'` | truth `'\n        flex-direction:'` | rule `'\n      }\n      .'` | mlp `'height:24px;'`
- `s2w_8482` pos 138 lex=cssprop a_mlp=1 a_rule=2 | ctx `'display: flex;\n        flex-direction: row;'` | truth `'\n        align-items:'` | rule `'\n        flex-direction:'` | mlp `'\n      }\n      .'`
- `s2w_8482` pos 156 lex=cssprop a_mlp=3 a_rule=4 | ctx `' center;\n        gap: 4px;\n      '` | truth `'}\n      .col {'` | rule `'}\n      .row {'` | mlp `'}\n      body {\n'`
