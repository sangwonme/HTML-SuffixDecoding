# RESEARCH_CONTEXT — 연구 맥락 전체 (새 세션은 이 문서부터)

## 현재 상태 요약 (2026-11-08 아침)
1. **권고는 Conditional Go다.** 근거는 `REPORT_2026-11-08.md` §1이다.
2. **확인된 기여.** 학습한 구조 조건 초안 선택(train-split MLP)이 독립 test 100개 GPU 실측에서 다음처럼 빨랐고, 품질 차이는 ±0.01 이내로 유의하지 않았다(POC_method.md §6.5, `runs/poc/E/test100/gate2_summary.md`).

   | 비교 | wall-clock 이득 |
   |---|---|
   | DOM 규칙 대비 | **+5.9% [+3.2, +8.7]** |
   | 표준 PLD 대비 | **+9.5% [+5.0, +13.3]** |

3. **외부 데이터 WebSight 50개(1회, POC_method.md §6.6, `runs/poc/F_websight/`).**
   - DOM 규칙 이득은 옮겨간다: PLD − DOM +1.51초 [+0.14, +2.81], fwd +4.17% [+0.71, +7.38]. 요청별 비율은 +3.22% [−0.24, +6.46]로 0을 조금 포함한다.
   - 학습 모델 이득은 옮겨가지 않는다: DOM − RANKER −0.53% [−3.35, +2.22], 유의하지 않음. PLD − RANKER +3.50% [+0.85, +5.97]는 규칙의 몫과 같다.
   - 품질 동일: 요소 F1 0.454/0.467/0.469, render_ok 1.0.
4. **한계.**
   - 정식 SuffixDecoding 위에 DOM 특징을 더한 추가 이득은 1–3%(f16)다. 공식 기본값 f1에서는 0.5% 또는 유의하지 않다(POC_study.md §12b).
   - 학습 모델은 서식 변환(fmt)에 약하다. 조건 A에서 규칙 대비 −11.5%였다(POC_method.md §6.4.2).
   - 학습 모델 이득은 S2W 변환기 형식 안에서만 확인됐다. WebSight에서는 규칙과 같다.
5. **다음 단계.**
   - SD+DOM GPU 실측
   - 형식을 섞은 데이터와 외부 출처(WebSight pool 150)를 섞어 학습
   - 전역 캐시를 포함한 위치 검색(오라클 46.9 vs SD f16 94.9, POC_study.md §15)
6. **사용자 결정 대기.**
   - 공식 ArcticInference suffix tree 모듈 빌드 승인(RETRIEVAL.md "Official SuffixDecoding vs reproduction")
   - 코드 라이선스
   - 데이터 공개 범위

작성: 2026-11-08, 기존 문서(PLAN.md, NOTES.md, DATASET.md, POC_method.md, POC_study.md, RETRIEVAL.md)와 `runs/` 결과 파일에서 옮겨 적었다. 모든 수치에 출처를 붙였다. 출처로 확인되지 않는 내용은 "확인 필요"로 표시했다.

용어:
- fwd: target forward 수
- PLD: prompt lookup decoding
- SD: SuffixDecoding
- pilot / dev / test: DATASET.md의 split
- CI: 요청 단위 짝 부트스트랩 95% 구간

---

## 1. 연구 목표
- **생성형 UI(자연어 질의 → HTML)의 디코딩 지연을 줄이고, 정확도는 같거나 낫게 유지한다.**
- 대회(KSEF)용 연구이므로 **AI 기여**를 중심에 둔다. 현재 기여는 "부분 DOM 구조를 조건으로 한 초안 위치 선택과 그 학습 순위 모델"이다.
- 자원 제약: 로컬 RTX 4070 12 GB, Gemma 4(E2B, E4B, 12B) QAT 체크포인트를 bitsandbytes NF4로 로드, 배치 1. 상용 API는 쓰지 않는다(PLAN.md 머리말).
- 논문 원칙(PLAN.md "실행 목표와 분기 규칙"):
  - 검증된 결과와 가설을 구분한다.
  - 큰 속도 향상(예: 11배)은 기존 기술인 예시 기반 투기 디코딩의 성과임을 밝힌다.
    - 11배는 C0 zero-shot greedy 59.4 s와 C4 struct + RNSD-L2 5.3 s의 비다(POC_method.md §5.3 표에서 계산).
  - 독립 test 평가는 E(ii) 100개 1회뿐이다(POC_method.md §6.5). 그 범위를 밝힌다.
  - 통계적으로 확인된 효과만 주장한다.

## 2. 타임라인

### 2.1 문제 설정과 데이터셋 (2026-11-07 오전–오후)
- "Screen2Text"라는 데이터셋은 없었다. Screen2Words로 확정했다(NOTES.md §1).
- **S2W-Query2HTML v0.1**(DATASET.md):
  - 원천: HF `rootsautomation/RICO-Screen2Words`(CC-BY-4.0, 22,417화면). 공식 split 사이에 겹치는 앱은 0개다.
  - 질의 세 종류:
    - `query_short`
    - `query_rich`: 요소 개수와 핵심 텍스트 포함
    - `query_mid`: 목적 + 핵심 요소 2–4개, v0.2에서 추가하고 체크리스트 채점과 짝지음
  - GT HTML은 RICO semantic annotation을 결정론적으로 변환한 `gt_html_flow`(구조 비교용)와 `gt_html_abs`(시각 비교용)다.
  - 분할: test 500, pilot 50(test의 부분집합), dev 100(val), train_pool 14,522. 렌더 성공률은 100%다.
- 지표(`src/genui/metrics.py`, `checklist.py`):
  - 요소/종류 multiset F1, 텍스트 재현율.
  - query_mid의 주 지표는 **체크리스트 충족률**이다. GT 자체 점수는 1.000이다.
  - 정규화기 자체 오차: GT flow 요소 F1 0.926(NOTES.md §7).

### 2.2 기준선 (NOTES.md §7, §9)
| | test 요소 F1 | test 텍스트 재현율 | 체크리스트 (pilot / dev / test) | 출처 |
|---|---|---|---|---|
| B1 template | 0.500 | 0.010 | 0.105 / 0.067 / 0.080 | `runs/baselines/summary_test.json`, `runs/baselines_mid/summary.json` |
| B2 retrieval (query_rich) | 0.613 | 0.251 | 0.318 / 0.332 / 0.348 | 같음 |
| B2 retrieval (query_mid) | 0.581 | – | 0.382 / 0.374 / 0.367 | `runs/baselines_mid/summary.json` |

### 2.3 지연 측정 (2026-11-07 오후–저녁)
- 실행기:
  - Ollama는 GUI 창이 떠서 금지했다(PLAN.md §4, NOTES.md §5).
  - transformers + bitsandbytes NF4로 바꿨다: torch 2.11+cu128, transformers 5.22.2, bitsandbytes 0.50.2.
- 파일럿 n=10, greedy(`runs/gemma/summary.md`):

| 조건 | total p50 |
|---|---|
| E2B rich_zs | 87.8 s |
| E2B rich_fs | 24.4 s |
| 12B rich_zs | 142 s |

  - decode는 10–18 tok/s다.
  - 이 속도 때문에 **test 500은 보류했다**(사용자 결정, NOTES.md §8).
- Preliminary study(POC_study.md, CPU):
  - wall-clock의 98.8–99.9%가 디코딩이다.
  - 출력의 41–69%가 `<body>` 이전(head+CSS)이다.
  - 보이는 텍스트는 2.1–4.5%다.
  - few-shot 출력은 검색 예시의 4-gram으로 0.96–0.97이 덮인다(π, E2B/E4B fs).
- **예시 기반 투기 디코딩**(POC_method.md §5.1, E2B pilot 10):
  - PLD는 greedy 27.1 s → 6.6 s로 **4.47배**다.
  - "화면 절반 표시 시점" t50은 C1 greedy 20.3 s → C4 3.3 s다(§5.4).

### 2.4 탐색했다가 폐기하거나 축소한 방향

| 방향 | 무엇을 했나 | 결과 | 결정과 이유 | 출처 |
|---|---|---|---|---|
| **렌더 동치 수용 (RNSD L1/L2/L4)** | 렌더 결과가 같은 초안 토큰 불일치를 수용(파서 동치, 클래스 α-재명명, CSS 수치 허용) | L1은 한 번도 발동하지 않았다. L2 v1은 렌더 동일 0/10(바인딩된 클래스명을 바꿈), v2는 2–6/10. L4는 체크리스트 1.00 → 0.883(손실) | **폐기.** greedy target이 재명명에 등변이 아니라 렌더 무손실이 성립하지 않는다 | POC_method.md §5.2, NOTES.md "결정 로그" |
| **렌더 공간 검색** (κ 정규형 위 4-gram 겹침 ω를 최대화하는 예시 선택) | φ̂(query) ridge + TF-IDF 혼합(β=0.9), dev 튜닝 | dev top-1 ω₄ 0.807 → 0.836 (+0.029 [+0.011, +0.048]). 복사 체크리스트 −0.083(유의). GPU에서 C4 struct 예시는 pilot 50 체크리스트 +0.033 [+0.002, +0.072]. oracle-ω 예시(C5)는 +1.40 s 느림 | **부록으로만 둔다.** 결과가 엇갈렸고(mix 검색 C6은 유의하지 않음), ω는 지연의 좋은 대리 지표가 아니다 | RETRIEVAL.md, POC_method.md §5.3 |
| **CSS 재배치 (인터리브, C7)** | CSS 규칙을 처음 쓰이는 블록 앞으로 옮김(픽셀 동일) | greedy에서는 t50 20.3 → 16.2 s, L_perc/T −10~−23%. RNSD 위에서는 이득이 없다(C7 t50 4.0 s vs C3 3.5 s). body-first는 오히려 악화(0.73 → 0.97) | **폐기.** 투기 디코딩이 CSS head를 이미 몇 forward 만에 복사한다 | POC_study.md §6b, POC_method.md §5.4 |
| **지연 목표 검색 학습** (요청마다 후보 예시 6개의 실제 지연을 재고 학습) | E2B L2, 예시 1개, pilot 50 × 6 = 300회 | 가장 빠른 후보가 tfidf1보다 −4.60 s [−6.21, −3.11](−37%), 품질 손실 없음. 분산의 75%가 출력 길이. 배포 특징 ranker는 pairwise 정확도 0.571, 격차의 13%만 회수(유의하지 않음) | **"차이 크지만 예측 불가"로 판정, 중단** | RETRIEVAL.md "Latency oracle gap", `runs/retrieval/cands6_analysis.json` |
| **순차 결정 (MDP, sequential oracle)** | 초안 위치를 이후 정렬까지 고려해 고르기 | any-position 오라클에서 정확한 DP = myopic(70/70 샘플, 차이 0). DOM top-M 안 lookahead와 DP 이득 ≤ 0.04% | **폐기.** C1을 한 스텝 효용 최적화로 축소(§3.3 명제) | RETRIEVAL.md "Sequential vs myopic oracle", `runs/retrieval/seq_oracle.json` |
| **구조 점프 / 문맥 밖 구조 생성** | 접미사 일치가 없을 때 구조 유사 위치로 점프, 닫는 태그 초안 생성 | O3−O2 = 0.33 fwd(poc), 0(gemma). 실행 규칙은 m≤0에서 +0.3%, m≤1에서 −18 fwd(악화) | **폐기** | RETRIEVAL.md "Ranking vs construction gap" |
| SuffixDecoding + DOM 가중 (관문 1) | SD 후보 출현 위치에 구조 유사도 가중 | 통과. 단 효과 2–3%: SD − (SD+DOM) poc 3.8 [2.6, 5.1], gemma 5.4 [3.3, 7.6] fwd | 유지(작은 효과) | POC_study.md §10 |

### 2.5 방향 확정까지 (2026-11-07 22:55 → 2026-11-08 00:55)
- 초안 위치 시뮬레이터(POC_study.md §9):
  - 표준 PLD 재생이 실제 로그와 10/10 일치한다.
  - DOM 인지 규칙은 표준 대비 −7.9% [6.4, 9.5] fwd다.
  - 오라클 위치는 −42.8%이고, 격차의 81–85%가 남는다.
- **관문 2 (GPU)**: DOM 규칙이 PLD보다 실제 wall-clock에서 빠르다(POC_method.md §6.2).
- **Stage 1–2 학습 순위 모델**: DOM 규칙보다 fwd −6.9 [4.5, 10.2], 구조 특징 기여 +3.7 [3.3, 4.1](POC_study.md §11).
- **관문 2b (GPU)**: 학습 순위 모델이 DOM 규칙보다 wall-clock +5.13% [+1.83, +8.10] 빠르다(POC_method.md §6.3).

### 2.6 밤사이 결과 (2026-11-08 01:30–06:00)

| 실험 | 결과 | 출처 |
|---|---|---|
| **P1-A** (예시와 출력 형식을 모두 변형, GPU, pilot 50 × 1회) | PLD − DOM은 4개 변형 모두 유의: fmt +6.4%, css +11.9%, dom +6.9%, cls +6.2%. DOM − RANKER(원래 형식 가중치)는 fmt **−11.5% [−27.8, −1.1]**, 나머지 3개는 유의하지 않음(0 ~ +3.7%). 원인은 target 쪽 형식 이동(출력 CSS까지 줄바꿈)과 특징 분포 이동 | POC_method.md §6.4, §6.4.2, `runs/poc/p1A/p1A_eval.md`, `cause_fmt.md`, `cause_dom_cls_css.md` |
| **P1-B** (초안 원천만 변형, GPU 실측) | DOM − RANKER는 fmt +2.5% [+0.7, +4.3], dom +5.7%, cls +4.6%(셋 다 유의), css −1.3%(n.s.). 추정치와 2%p 이내로 맞음. 품질 변화 없음 | POC_method.md §6.4.3, `runs/poc/p1B/<t>/gate2_summary.md` |
| **E 독립 test** (train-split MLP, test 100, 앱 92개, pilot과 겹치지 않음) | RANKER 4.86 s, DOM 5.17 s, PLD 5.28 s. DOM − RANKER +5.90% [+3.15, +8.72], PLD − RANKER +9.51%. PLD − DOM은 +2.3% (n.s., 출력 길이 차이) | POC_method.md §6.5, `runs/poc/E/test100/` |
| train-split 학습 (재생) | train 앱 150개로만 학습해도 규칙보다 3.8 [2.9, 4.9] fwd 적음(cross-fit 이득의 57–87%) | POC_study.md §15 끝("E. 독립 train-split 학습"), `runs/study/ranker_trainsplit.json` |
| **정식 SD 공식 규칙 재실행** (자기 응답을 global에 포함) | SD − (SD+학습): f1 0.52 [0.14, 0.89](기존 1.3에서 축소), f16 **2.9 [2.25, 3.61]**. MAX_SPEC 64에서는 f1 0.19 (n.s.), f16 0.96 [0.40, 1.54] | POC_study.md §12b, `runs/study/sd_official_rule_ms{16,64}.json` |
| 공식 구현 대조 | 알고리즘은 동일(114/114 재현, 동률·m≤63·트리 영향 0). 설정 차이 2개(own-global −1.1%, MAX_SPEC). C++ 빌드는 권한 정책으로 미실행 | RETRIEVAL.md "Official SuffixDecoding vs reproduction" |
| **표현 변화에서 SD vs SD+DOM** (재생) | f16 이득 orig 2.1 → css 6.0, dom 4.1. 그중 DOM 특징만의 몫은 0.96 → 1.38 / 1.70. 나머지는 SD 후보 재순위화 효과 | POC_study.md §13, `runs/study/sd_p1A*.json` |
| **SD 전역 캐시 감사** | 누수 0. 서빙 순서 캐시 f16 103.0 (cross-fit 94.9보다 +8.2), PLD 대비 이득의 79% 유지. 상용구 제거해도 82% 유지 | POC_study.md §14, `runs/study/sd_cache_audit.json` |
| **오라클 비교** | SD f16 94.9 → SD 후보 오라클 80.5 → local 오라클 74.5 → **local+global 오라클 46.9**. 구조 생성 후보(O3)는 거의 0 | POC_study.md §15, RETRIEVAL.md "Ranking vs construction gap", "O1 vs O2 verdict" |
| 외부 데이터 준비 | WebSight 200 / Design2Code 484 준비. GT 대리 재생에서 DOM 이득 WebSight −6.1%, S2W −3.6% | DATASET.md "External data (prep)" |
| 문서 | 아침 보고서 `REPORT_2026-11-08.md` (`paper/report_2026-11-08.pdf`), 영문 논문 `paper/genui_paper.pdf` | |

## 3. 현재 확정 방향

### 3.1 제목과 핵심 문제 (PLAN.md "우선순위 결정 (2026-11-08 00:55)")
- **Structure-Conditioned Speculative Draft Selection for HTML Generation**
- 어휘 기반 speculative decoding은 반복되는 HTML 토큰 패턴이 서로 다른 DOM 문맥에 나타나는 것을 구분하지 못해 부적절한 초안을 고른다.

### 3.2 C1과 C2
- **C1 (선택 규칙)**:
  - j* = argmax_{j∈J_t} E[A_j | p_t, z_t, z_j], A_j = LCP(d_j, y_{t+1:})
  - J_t는 후보 위치 집합이고, z는 구조 주석(어휘 상태, 속성/CSS 키, 태그 경로, 깊이, 형제 순번, 직전 이벤트)이다.
  - 한 스텝 최선만 쓰고 순차 정식화는 하지 않는다(§3.3).
- **C2 (학습 순위 모델)**: 부분 DOM과 후보 위치의 구조 관계로 수용 길이를 예측하는 경량 모델이다(POC_study.md §11).
  - 후보: 가장 긴 매칭 n-gram(3→1)의 출현 위치(최근 300개)를 DOM 규칙 점수로 정렬하고 continuation을 합친 상위 8개다.
  - 특징 23개: 구조 12, 어휘 6, 이력 5.
  - 모델: HGB 회귀(+DAgger 1회) 또는 2층 MLP(64-32, pairwise logistic). 배포는 numpy MLP다(`runs/study/ranker_mlp_fold{0..4}.npz`, `ranker_fold_map.json`, 앱 단위 cross-fit).
  - 추론 코드는 `scripts/study_ranker.py`의 `RankerDrafter`이고, GPU 경로는 `scripts/poc_rnsd.py`다.

### 3.3 한 스텝 최선 최적성 명제와 조건 (RETRIEVAL.md "Sequential vs myopic oracle")
- **명제**: 초안 후보 집합이 "문맥의 모든 시작 위치"이면, 매 스텝 수용 길이가 최대인 위치를 고르는 myopic 정책이 전체 forward 수를 최소화한다.
- **조건**:
  1. greedy target에 정확 검증(LCP + 보너스 1)을 쓴다. 이때 위치 pos의 문맥(prompt + y[:pos])이 고정이라 이후 진행은 pos에만 달려 있다.
  2. 후보 집합이 위치 이동에 닫혀 있다(시작 위치 s가 후보면 s+k도 후보). 이 조건 때문에 도달점 r(pos) = pos + a*(pos) + 1이 단조이고, jump-game 교환 논증이 성립한다.
  3. γ 상한이 고정이다.
- **실측**: 70/70 샘플에서 정확한 DP = myopic이다.
- 후보 집합을 제한하면(DOM top-M) 조건 2가 깨진다. 그래도 차이는 ≤ 0.04%였다.

## 4. 핵심 결과 표

### 4.1 GPU 실측, E2B NF4, pilot 50, 3 reps, 정확 검증, 요청 단위 짝 CI

| 비교 | wall-clock 차이 | 상대 | forward 차이 | 출처 |
|---|---|---|---|---|
| PLD − DOM (관문 2) | +0.370 s [+0.181, +0.572] | +6.95% [+4.06, +10.00] | +6.86 [+3.60, +10.24] | `runs/poc/gate2_p50/gate2_summary.md` |
| DOM − RANKER (관문 2b) | +0.293 s [+0.049, +0.531] | +5.13% [+1.83, +8.10] | +4.54 [+0.82, +8.14] | `runs/poc/gate2b/gate2_summary.md` |
| PLD − RANKER | +0.702 s [+0.434, +0.980] | +11.87% [+8.49, +15.24] | +11.40 [+7.14, +16.08] | 같음 |

- 평균 wall-clock: PLD 6.53 s, DOM 6.12 s, RANKER 5.83 s(관문 2b 세션).
- 초안 비용 비중: 0.23% / 0.64% / 0.85%. peak GPU는 7.0 GB로 모두 같다.
- 시뮬레이터 예측 forward와 실측이 **450/450** 일치한다(POC_method.md §6.3). 그래서 CPU 시뮬레이터 결과는 forward 수로는 오차 없이 옮겨진다.

### 4.2 CPU 시뮬레이터 (이상적 greedy target 재생, γ=16)
| 정책 (fwd/요청) | poc greedy n=40 | gemma fs n=30 (근사) | 전체 n=444 | 출처 |
|---|---|---|---|---|
| 표준 PLD n=3→1 | 187.2 | 217.9 | 159.0 | `runs/study/draft_sim.json`, `ranker.json` |
| SuffixDecoding 근사 (α=16) | 183.2 | 199.0 | – | `runs/study/suffix_sim.json` |
| SD + DOM 가중 | 179.3 | 193.6 | – | 같음 |
| DOM 규칙 | 176.5 | 204.8 | 152.4 | `draft_sim.json`, `ranker.json` |
| 학습 HGB(+DAgger) | 173.1 | 186.1 | 145.5 | `runs/study/ranker.json` |
| DOM top-8 오라클 | 157.3 | 162.9 | 130.6 | 같음 |
| O1: 접미사 일치 후보 오라클 | 154.2 | 149.9 | – | `runs/retrieval/rank_construct.json` |
| O2: 문맥 전체 위치 오라클 | 103.4 | 107.0 (106.9*) | 86.7 | 같음, `draft_sim.json` |

\* 같은 오라클을 다른 스크립트에서 계산하면 12B 샘플 하나의 max_new 경계 처리 차이로 0.1이 다르다.

- 학습 − 규칙 차이(`ranker.json`, 전체): DOM 규칙 − 학습 = 6.9 [4.5, 10.2] fwd. 앱 단위 GroupKFold, poc만 보면 3.4 [1.7, 5.2]다.
- ablation: 구조 특징 제거 +3.7 [3.3, 4.1], 이력 특징 제거 +0.1 [−0.2, 0.4](POC_study.md §11).
- O1/O2 판정(RETRIEVAL.md "O1 vs O2 verdict"):
  - O2는 O1보다 29–33% 적다. 그래서 병목은 **후보 검색(접미사 매칭)**이다.
  - 학습 순위 모델은 O1보다 11–19%, O2보다 40–43% 높다. 그래서 추가 개발을 보류하지 않는다.
  - O2만 찾는 스텝의 44–70%는 직전 토큰이 문맥에 처음 나온 경우다. 다음 토큰은 태그 경계 약 40%, CSS 값과 속성 약 42%다.
- 오라클은 미래 정답을 쓰는 이상적 상한이고, 실행 가능한 알고리즘이 아니다.

### 4.3 디코딩 조건 표 (E2B, pilot 10; pilot 50은 별도 표기)
| 행 | 예시 | 디코딩 | 체크리스트 | total p50 | tok/fwd | C1 대비 | 출처 |
|---|---|---|---|---|---|---|---|
| C0 | 없음 | greedy | 0.867 | 59.4 s | 1.00 | 0.47x | POC_method.md §5.3 (`runs/poc/E2B_Ctable.md`) |
| C1 | TF-IDF | greedy | 1.000 | 27.1 s | 1.00 | 1.00x | 같음 |
| C2 | TF-IDF | PLD | 1.000 | 6.6 s | 4.58 | 4.47x | 같음 |
| C4 | struct | RNSD-L2 v2 | 1.000 | 5.3 s | 5.65 | 5.66x | 같음 |
| C4 − C1 (pilot 50, L2) | | | +0.033 [+0.002, +0.072] | +0.15 s (ns) | | | `runs/poc/E2B_ci.md` |
| C5 − C1 (pilot 50, oracle 예시) | | | +0.013 (ns) | +1.40 s (유의) | | | 같음 |

### 4.4 P1 표현 강건성 (GPU, pilot 50 × 1회, POC_method.md §6.4.2–6.4.3)
| 변형 | A: PLD − DOM | A: DOM − RANKER | B: DOM − RANKER (실측) |
|---|---|---|---|
| orig (gate2b, 3회) | +6.8% [+3.7, +9.9] | +5.1% [+1.8, +8.1] | – |
| fmt | +6.4% [+0.4, +12.7] | **−11.5% [−27.8, −1.1]** | +2.5% [+0.7, +4.3] |
| css | +11.9% [+6.4, +16.8] | 0.0% [−6.9, +5.3] | −1.3% [−5.6, +1.9] |
| dom | +6.9% [+4.2, +9.6] | +3.7% [−0.9, +8.3] | +5.7% [+0.5, +9.7] |
| cls | +6.2% [+1.2, +10.7] | +2.6% [−2.1, +6.6] | +4.6% [+2.2, +7.1] |

- 체크리스트는 0.89–0.96이다. 변형 때문에 품질이 체계적으로 떨어지지는 않았다.

### 4.5 독립 test E(ii) (GPU, test 100 × 1회, train-split MLP, POC_method.md §6.5)
| | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock | 5.28 s | 5.17 s | **4.86 s** |
| forwards | 88.5 | 86.0 | 80.5 |
| checklist | 0.944 | 0.938 | 0.943 |

- DOM − RANKER: +5.90% [+3.15, +8.72]
- PLD − RANKER: +9.51% [+5.03, +13.29]
- PLD − DOM: +2.30% [−2.93, +6.70] (n.s.)

### 4.6 정식 SD 공식 규칙과 MAX_SPEC 민감도 (재생, 주 데이터 n=114, POC_study.md §12b)
| forward/요청 | MAX_SPEC 16 | MAX_SPEC 64 |
|---|---|---|
| SD f1 | 108.1 | 103.0 |
| SD + 학습 f1 | 107.5 | 102.8 |
| SD f16 | 94.3 | 81.9 |
| SD + 학습 f16 | 91.4 | 80.9 |
| SD − (SD+학습), f1 | 0.52 [0.14, 0.89] | 0.19 [−0.09, 0.47] |
| SD − (SD+학습), f16 | **2.9 [2.25, 3.61]** | 0.96 [0.40, 1.54] |

- 검증 토큰/forward(f16)는 13.1–13.5(16)와 29.6–31.5(64)다.
- 캐시 감사(§14): 서빙 순서 캐시 f16은 103.0 fwd로, cross-fit보다 +8.2 [5.6, 11.4] 많다.
- 오라클(§15): SD 후보 오라클 80.5, local 74.5, local+global 46.9다.

### 4.7 Preliminary study (n=10 × 7조건, POC_study.md)
| 항목 | 값 | 출처 |
|---|---|---|
| 예측 가능 토큰 π | zs 0.65–0.72, fs 0.76–0.97 | `runs/study/s1_predictability.json` |
| 새 토큰 N 중 CSS 비중 | 48–56% | 같음 |
| `<body>` 이전 토큰 | 41–69% | `runs/study/p1_time.json` |
| L_perc/T | 0.70–0.85 | `runs/study/s3_vc.json` |
| ω(e,y)와 지연 / 요소 F1의 상관 | ρ −0.57 [−0.72, −0.36] / +0.46 [+0.25, +0.64] | `runs/study/s5_omega.json` |

## 5. 실험 상태와 결과 위치 (2026-11-08 아침)

| 실험 | 내용 | 결과 위치 | 상태 |
|---|---|---|---|
| **관문 2b** | 학습 순위 모델 GPU 실측 | `runs/poc/gate2b/gate2_summary.md` | **완료, 통과**(§4.1) |
| **P1-A** | Reference와 Target 모두 변형(fmt, css, dom, cls) | `runs/poc/p1A/<t>/gate2_summary.md`, `p1A_eval.md`, `cause_*.md`; POC_method.md §6.4.2 | **완료**(§4.4) |
| **P1-B** | Reference만 변형 | `runs/poc/p1B/<t>/gate2_summary.md`, `quality.md`, `p1B_estimate.md`; POC_method.md §6.4.3 | **완료**(§4.4) |
| P1 준비 | 변형 함수 `src/genui/perturb.py`, 예시 `runs/retrieval/pilot_exemplar_html_{fmt,css,dom,cls}.json` | `runs/study/p1_plan.md`, `runs/study/p1_perturb_report.json` | 완료(픽셀 동일: fmt/css/cls 150/150, dom 146/150) |
| **E 독립 test** | train-split MLP, test 100, GPU | `runs/poc/E/test100/gate2_summary.md`, `quality.md`; 학습 `runs/poc/E/train150/`, `runs/study/ranker_mlp_trainsplit.npz` | **완료**(§4.5) |
| **정식 SuffixDecoding** | 재구현 + 공식 global 규칙 재실행 + MAX_SPEC 민감도 + 표현 변화 + 캐시 감사 + 오라클 | `runs/study/sd_formal.json`, `sd_official_rule_ms{16,64}.json`, `sd_p1A*.json`, `sd_cache_audit.json`; POC_study.md §12–15 | **완료**(§4.6). GPU 실측은 없음 |
| 공식 구현 대조 | 소스 대조와 규칙 이식 | `runs/retrieval/sd_official_check.json`; RETRIEVAL.md | 완료. C++ 빌드는 승인 대기 |
| **F WebSight** | WebSight v0.2 eval 50 × {PLD, DOM, RANKER(train-split)} GPU | `runs/poc/F_websight/gate2_summary.md`, `websight_quality.md`; POC_method.md §6.6 | **완료**. 규칙 이득은 유지(+1.51초), 학습 모델은 규칙 대비 −0.53% n.s. 준비와 대리 재생은 DATASET.md "External data (prep)" |

## 6. 알려진 한계와 편향
- **같은 변환기 형식**:
  - GT와 few-shot 예시가 모두 같은 결정론적 변환기(`build_dataset.py`)의 출력이다. 그래서 반복 형식이 강하다.
  - 예: ω(e, GT)가 0.69–0.99로 몰린다(POC_study.md §6).
  - 개선이 이 변환기 형식에만 의존하는지가 P1의 핵심 질문이다.
  - WebSight 50개(1회)에서 답의 일부가 나왔다. 규칙 이득은 남고(초 단위 유의), 학습 모델의 추가 이득은 사라졌다. 학습 모델의 이득은 변환기 형식에 묶여 있을 수 있다. Design2Code는 생성 예산에 16개만 들어가 아직 하지 않았다.
- **test 출신 앱**:
  - cross-fit 순위 모델 데이터의 앱 50개는 모두 pilot(= test split) 출신이다(POC_study.md §11 한계). 학습 데이터의 2/3는 cands6 근사 target이다.
  - 이를 보완하려고 train 앱 150개로만 학습한 모델을 만들어, pilot과 겹치지 않는 test 100(앱 92개)에서 GPU로 평가했다(§4.5).
  - 다만 독립 평가는 1회 반복, E2B 하나, S2W 변환기 형식 하나뿐이다.
- **시뮬레이터 가정**:
  - 저장된 greedy 출력을 고정 target으로 재생한다(이상적 수용 분석).
  - gemma fs 데이터는 응답을 다시 토크나이즈했고, 12B는 프롬프트 길이가 10/10 불일치한다(근사).
  - 실측 forward와는 450/450 일치했다. 하지만 wall-clock은 GPU 측정으로만 판정한다.
- **4비트 수치 차이**:
  - NF4는 q4_0의 근사다.
  - 다중 토큰 검증 forward가 1토큰 forward와 수치가 달라 greedy와 출력이 갈라진다. PLD도 greedy와 같은 것은 3/10뿐이다(POC_method.md §5.1, §7).
  - 정책 사이 출력 동일은 21–22/50이다.
- **예시 선택 4000자 버그와 C7 영향**. 출처는 PoC 에이전트의 2026-11-08 01:17 보고이고, 코드 근거는 `runs/poc/poc_rnsd_pre_p1b.py.bak`(수정 전)과 `scripts/poc_rnsd.py`(수정 후)다.
  - **버그**: `--exemplar-html`로 예시 HTML을 바꿔 넣으면, TF-IDF 후보의 4000자 필터가 원본이 아니라 **바뀐 HTML 길이**에 적용됐다. HTML이 길어지는 변환(fmt, css)에서는 원본과 다른 예시 id가 선택될 수 있었다.
  - **수정**: P1-B를 구현하면서 고쳤다. 이제 예시 선택은 항상 원본 HTML 기준이다.
    - 원본 선택은 gate2_p50과 50/50 일치한다.
    - 변환 4종 모두 조건 A의 예시 id가 원본과 50/50 같다.
    - P1-A를 시작하기 전에 반영했다.
  - **C7 영향**(C7g, C7: 인터리브 CSS 예시, 버그 있는 코드로 실행): 버그 있는 선택을 CPU에서 다시 재현해 원본 선택과 비교했다(retrieval 에이전트, 2026-11-08).
    - 다른 예시 집합이 선택된 요청은 **pilot 50 중 0개, C7이 쓴 앞 10개 중 0개**다.
    - 인터리브 예시 150개 중 4000자를 넘는 것이 0개이기 때문이다(POC_study.md §8).
    - 따라서 C7 결과는 이 버그의 영향을 받지 않았다. C7은 이미 핵심 연구에서 빠졌으므로 결론에도 영향이 없다.
- 기타:
  - 체크리스트는 pilot 10에서 포화되므로, 비교에는 50개 이상이 필요하다.
  - gemma 파일럿 수치는 n=10이다.

## 7. 다음 단계와 판단 규칙 (PLAN.md "우선순위 결정", "실행 목표와 분기 규칙" 요약)

**현재 권고 (Conditional Go, `REPORT_2026-11-08.md` §1)**
- 계속할 것: 구조 규칙, SD 후보 재순위화.
- 조건:
  - 독립 test에서 학습 모델이 규칙보다 느리지 않을 것. **E(ii)에서 충족했다(+5.9%).** WebSight에서는 규칙과 같다(−0.53%, n.s.).
  - 다른 출처에서 학습 모델 이득을 회복할 것(외부 출처 혼합 학습). 아직이다.
  - SD+DOM의 GPU 실측 이득을 확인할 것. 아직이다.
- 멈출 것:
  - "구조만으로 SD를 이긴다"는 주장
  - "학습 모델이 형식 변화나 다른 HTML 출처에 강건하다"는 주장
- 다음 연구 후보: 전역 캐시를 포함한 위치 검색, 형식을 섞은 데이터로 학습.

**원래 우선순위**(모두 진행했다): GPU 실측(P0) → 오라클 격차 분석(P3) → 표현 강건성(P1) → 강한 기준선 비교(P2, 정식 SD) → 논문 초안. 논문 초안은 `paper/genui_paper.pdf`다.
- 후속(필수): Design2Code 데이터, E4B와 12B 모델. WebSight는 완료(§5).
- 보류: O3(구조 기반 새 후보 생성), O4(14,522개 전체 구조 인덱스).

**관문 2b 분기**:
1. wall-clock이 유의하게 빠르다 ← 현재 결과
2. 빠르지만 유의하지 않다
3. forward만 줄고 시간은 줄지 않는다
4. 감소가 재현되지 않는다

**P1 분기**:
- A(Reference와 Target 모두 변형)와 B(Reference만 변형)를 따로 본다.
- 성능이 크게 떨어지면 새 방법보다 먼저 변환기 편향과 후보 가용성을 분석한다.
- 효과가 사라지면 변환기 특화 최적화일 수 있으므로 문제를 다시 정의한다.

**O1/O2 분기**:

| 결과 | 조치 |
|---|---|
| O1 ≈ O2 | 순위 문제에 집중 |
| O2 ≫ O1 ← 현재 결과 | 후보 검색 한계 분석 |
| 두 오라클 모두 학습 모델과 차이가 작음 | 추가 개발 보류 |

**방향 유지 조건(넷 다 필요)**:
1. GPU 실제 시간이 개선된다.
2. 구조 특징이 독립적으로 기여한다.
3. 표현 변화에 합리적으로 강건하다.
4. 정식 SuffixDecoding보다 추가 효용이 있다.

하나라도 실패하면 새 모델을 만들기 전에 원인부터 분석한다.

**원칙**: 학습 모델의 실제 성능과 일반화를 확인하기 전에는 새 알고리즘 모듈을 추가하지 않는다.

## 8. 파일 지도
| 질문 | 문서 | 코드 / 결과 |
|---|---|---|
| 데이터는 어떻게 만들었나, 필드는? | DATASET.md | `scripts/build_dataset.py`, `build_query_mid.py`, `validate_dataset.py`, `data/processed/` |
| 지표 정의 | DATASET.md, PLAN.md §5 | `src/genui/metrics.py`, `src/genui/checklist.py` |
| 기준선 수치 | NOTES.md §7, §9 | `scripts/eval_baselines.py`, `eval_mid_baselines.py`, `runs/baselines*/` |
| Gemma 파일럿 지연 | NOTES.md §8 | `scripts/bench_gemma.py`, `summarize_runs.py`, `runs/gemma/summary.md` |
| 투기 디코딩 구현(PLD, RNSD, DOM, RANKER), GPU 실측 | POC_method.md | `scripts/poc_rnsd.py`, `poc_eval.py`, `poc_ci.py`, `poc_table.py`, `poc_perc.py`, `gate2_analyze.py`, `runs/poc/` |
| UI 출력의 특성(π, η, L_perc, ω) | POC_study.md §1–6b | `scripts/study_s*.py`, `study_canon.py`, `study_equiv.py`, `runs/study/` |
| 초안 위치 시뮬레이터, SD, 학습 순위 모델 | POC_study.md §9–11 | `scripts/study_draft_sim.py`, `study_suffix_sim.py`, `study_ranker.py`, `study_sd_formal.py`, `runs/study/ranker*.{json,npz}` |
| 예시 검색, ω, 지연 오라클, sequential, ranking vs construction, O1/O2 | RETRIEVAL.md | `src/genui/canon.py`, `scripts/retr_*.py`, `runs/retrieval/` |
| 표현 강건성(P1) | `runs/study/p1_plan.md` | `src/genui/perturb.py`, `scripts/study_perturb_exemplars.py`, `p1A_eval.py`, `runs/poc/p1A`, `runs/poc/p1B` |
| CSS 인터리브와 지각 지연 | POC_study.md §6b, §8 | `src/genui/interleave.py`, `scripts/study_s6*.py` |
| 우선순위와 결정 기록 | PLAN.md, NOTES.md(시간순 로그) | |
| 저장소에 무엇이 들어가나 | `runs/repo_audit.md` | `.gitignore` |
| P1 원인 분석, 조건 B 추정 | POC_method.md §6.4 | `scripts/p1A_cause.py`, `p1B_estimate.py`, `p1B_chain.sh`, `runs/poc/p1A/cause_*.md`, `runs/poc/p1B/p1B_estimate.md` |
| 독립 test E | POC_method.md §6.5, POC_study.md §15 끝 | `scripts/study_ranker_trainsplit.py`, `runs/study/ranker_trainsplit.json`, `ranker_mlp_trainsplit.npz`, `runs/poc/E/` |
| MLP 특징 ablation | POC_study.md §12 끝 | `scripts/study_ranker_mlp_ablation.py`, `runs/study/ranker_mlp_ablation.json` |
| 정식 SD 공식 규칙, MAX_SPEC, 표현 변화, 캐시 감사, 오라클 | POC_study.md §12b–15 | `scripts/study_sd_official_rule.py`, `study_sd_p1A.py`, `study_sd_cache_audit.py`, `runs/study/sd_official_rule_ms{16,64}.json`, `sd_p1A*.json`, `sd_cache_audit.json` |
| 공식 ArcticInference 구현 대조 | RETRIEVAL.md "Official SuffixDecoding vs reproduction" | `scripts/retr_sd_official_check.py`, `runs/retrieval/sd_official_check.json` |
| 외부 데이터(WebSight, Design2Code) | DATASET.md "External data (prep)", POC_method.md §6.6 | `scripts/prep_external.py`, `prep_external_replay.py`, `websight_eval.py`, `F_chain.sh`, `data/processed/external/`, `runs/retrieval/external_*`, `runs/poc/F_websight/` |
| 아침 보고서와 논문 | `REPORT_2026-11-08.md` | `paper/report_2026-11-08.{tex,pdf}`, `paper/genui_paper.{tex,pdf}`, `paper/build_paper.ps1` |
