# P1-A cause analysis (RANKER trajectory replay on measured RANKER outputs)

| set | steps | n-gram hit rate | hit n=3 | mean #cand | multi-cand share | acc MLP | acc rule (same state) | MLP<rule | MLP>rule | MLP != top1 | oracle gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orig | 4223 | 0.786 | 0.299 | 3.27 | 0.540 | 3.99 | 3.50 | 0.048 | 0.119 | 0.457 | 0.64 |
| dom | 4555 | 0.789 | 0.336 | 3.32 | 0.561 | 4.46 | 3.89 | 0.053 | 0.153 | 0.451 | 0.74 |
| cls | 5081 | 0.785 | 0.213 | 3.59 | 0.593 | 2.66 | 2.38 | 0.043 | 0.076 | 0.443 | 0.59 |
| css | 6161 | 0.822 | 0.398 | 3.91 | 0.611 | 4.62 | 4.21 | 0.065 | 0.109 | 0.467 | 0.92 |

Structural feature means over candidates at multi-candidate steps (shift vs orig):

| feature | orig | dom | cls | css |
|---|---|---|---|---|
| dom_score | 6.872 | 6.707 | 7.118 | 6.727 |
| dom_rank | 2.302 | 2.261 | 2.364 | 2.529 |
| lex_eq | 0.848 | 0.859 | 0.821 | 0.922 |
| key_eq | 0.098 | 0.091 | 0.149 | 0.027 |
| path_suf | 1.656 | 1.533 | 1.743 | 1.469 |
| parent_eq | 0.610 | 0.609 | 0.620 | 0.573 |
| depth_cur | 3.677 | 4.127 | 3.378 | 3.860 |
| depth_cand | 3.429 | 4.008 | 3.225 | 3.540 |
| depth_diff | 0.637 | 1.027 | 0.573 | 0.751 |
| sib_diff | 0.966 | 1.679 | 0.697 | 1.208 |
| event_eq | 0.767 | 0.788 | 0.810 | 0.746 |
| relpos_diff | 0.151 | 0.173 | 0.137 | 0.321 |

## dom: steps where MLP choice < DOM-rule choice (n=136), lexical state counts: {"tagname": 11, "text": 16, "attrname": 11, "cssval": 26, "boundary": 49, "cssprop": 12, "csssel": 3, "attrval": 8}

- `s2w_8482` pos 352 lex=tagname a_mlp=0 a_rule=4 | ctx `'Text Button">Keep On It</button></div><div'` | truth `' class="col"><p data'` | rule `' class="col"><button data'` | mlp `' style="display:contents"><'`
- `s2w_8482` pos 365 lex=text a_mlp=0 a_rule=2 | ctx `'="col"><p data-label="Text">Browse Radio'` | truth `'</p><ul class="'` | rule `'</p></div><div'` | mlp `' & Podcast</p></div>'`
- `s2w_8482` pos 418 lex=attrname a_mlp=7 a_rule=16 | ctx `'-label="Icon" aria-label="radio" class'` | truth `'="icon" style="width'` | rule `'="icon" style="width'` | mlp `'="icon" style="width'`
- `s2w_70325` pos 355 lex=cssval a_mlp=0 a_rule=16 | ctx `'="emoji" class="icon" style="width:4'` | truth `'0px;height:4'` | rule `'0px;height:4'` | mlp `'2px;height:4'`
- `s2w_70325` pos 478 lex=text a_mlp=0 a_rule=2 | ctx `'button data-label="Text Button">Submit</button>'` | truth `'</div></div></div></body></'` | rule `'</div></div><p data-'` | mlp `'\n```<turn|>\n<|turn>user'`
- `s2w_49685` pos 388 lex=boundary a_mlp=1 a_rule=9 | ctx `'-label="Text">Seat: 12B</'` | truth `'p><p data-label'` | rule `'p><p data-label'` | mlp `'p></li><li data'`
- `s2w_59299` pos 146 lex=cssprop a_mlp=0 a_rule=2 | ctx `'background:#ccc;display:inline-block}.button{'` | truth `'background:#4CAF50'` | rule `'background:#ccc;display:'` | mlp `'display:flex;flex-'`
- `s2w_59299` pos 202 lex=csssel a_mlp=7 a_rule=16 | ctx `'bold;cursor:pointer;text-align:center}</'` | truth `'style></head><body><'` | rule `'style></head><body><'` | mlp `'style></head><body><'`

## cls: steps where MLP choice < DOM-rule choice (n=131), lexical state counts: {"cssprop": 20, "boundary": 22, "tagname": 7, "cssval": 48, "csssel": 8, "text": 11, "attrname": 7, "attrval": 8}

- `s2w_70325` pos 134 lex=cssprop a_mlp=1 a_rule=5 | ctx `';display:inline-block}.submit-btn{display'` | truth `':inline-block;padding'` | rule `':inline-block;width'` | mlp `':flex;flex-direction'`
- `s2w_70325` pos 212 lex=boundary a_mlp=0 a_rule=2 | ctx `'</p></div><div class="review-content"><'` | truth `'p>Think about your experience'` | rule `'p>Review</p>'` | mlp `'header data-label="Toolbar'`
- `s2w_70325` pos 213 lex=tagname a_mlp=0 a_rule=1 | ctx `'p></div><div class="review-content"><p'` | truth `'>Think about your experience and'` | rule `'>Review</p></div>'` | mlp `' data-label="Text">'`
- `s2w_70325` pos 261 lex=cssval a_mlp=0 a_rule=1 | ctx `';height:30px;background:#ff00'` | truth `'00;border-radius'` | rule `'000;border-'` | mlp `'px;height:30'`
- `s2w_70325` pos 318 lex=cssval a_mlp=0 a_rule=16 | ctx `':30px;background:#ff9900'` | truth `';border-radius:5'` | rule `';border-radius:5'` | mlp `'00;border-radius'`
- `s2w_70325` pos 371 lex=cssval a_mlp=0 a_rule=16 | ctx `';height:30px;background:#ffff00'` | truth `';border-radius:5'` | rule `';border-radius:5'` | mlp `'0;border-radius:'`
- `s2w_70325` pos 425 lex=cssval a_mlp=0 a_rule=1 | ctx `'height:30px;background:#8888'` | truth `'88;border-radius'` | rule `'888;border-'` | mlp `';border-radius:4'`
- `s2w_49685` pos 364 lex=boundary a_mlp=0 a_rule=1 | ctx `' data-label="Text">Class: Economy</p></'` | truth `'li><li data-label'` | rule `'li></ul><article data'` | mlp `'header><h1 data-'`

## css: steps where MLP choice < DOM-rule choice (n=244), lexical state counts: {"cssval": 169, "boundary": 23, "text": 29, "attrname": 11, "attrval": 2, "tagname": 10}

- `s2w_8482` pos 335 lex=cssval a_mlp=0 a_rule=2 | ctx `'-box; padding: 10px 0 '` | truth `'0 0; display:'` | rule `'0 0 0;'` | mlp `'2px 4px;">'`
- `s2w_8482` pos 336 lex=cssval a_mlp=0 a_rule=2 | ctx `'box; padding: 10px 0 0'` | truth `' 0; display: flex'` | rule `' 0 0; display'` | mlp `'px; padding: 0'`
- `s2w_8482` pos 340 lex=cssval a_mlp=7 a_rule=12 | ctx `' 10px 0 0 0; display'` | truth `': flex; flex-direction'` | rule `': flex; flex-direction'` | mlp `': flex; flex-direction'`
- `s2w_8482` pos 431 lex=boundary a_mlp=0 a_rule=2 | ctx `' 16px; cursor: pointer;">PodcastRepublic</'` | truth `'button><button data-label'` | rule `'button><input data-label'` | mlp `'p></header><div class'`
- `s2w_8482` pos 613 lex=cssval a_mlp=0 a_rule=1 | ctx `'box-sizing: border-box; padding: 8'` | truth `'px; border: 1'` | rule `'px"><p data-label'` | mlp `'3px"></span></div'`
- `s2w_8482` pos 634 lex=text a_mlp=0 a_rule=1 | ctx `'ccc; border-radius: 4px;"></div></div>'` | truth `'</div>\n```<turn|>'` | rule `'</div></div>\n```<turn|>'` | mlp `'<div class="col"'`
- `s2w_70325` pos 361 lex=text a_mlp=0 a_rule=13 | ctx `' Button" style="box-sizing: border-box;">'` | truth `'Submit</button></div></'` | rule `'Submit</button></div></'` | mlp `'SUBMIT</button></div'`
- `s2w_49685` pos 340 lex=cssval a_mlp=0 a_rule=1 | ctx `' margin: 4px 0 10px '` | truth `'0;">Adults</h1><'` | rule `'0 10px '` | mlp `'4px;">Allow app to'`
