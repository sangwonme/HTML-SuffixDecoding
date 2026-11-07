# GenerativeUI 실험 계획 (PLAN)

- 최종 갱신: 2026-11-07
- 목표: Screen2Words를 바꿔 **(사용자 자연어 질의 → UI HTML)** 태스크를 만들고 베이스라인을 세운다. 그다음 **로컬 Gemma**(RTX 4070 12GB)로 이 태스크를 풀 때 토큰, 레이턴시, 정확도를 측정한다.
- 상용 API는 쓰지 않는다(2026-11-07 사용자 결정). 이전 API 단가 조사는 NOTES.md §3에 참고용으로만 남긴다.
- 데이터셋 상세는 `DATASET.md`, 조사 원자료는 `NOTES.md`에 있다.

## 현재 상태
| 단계 | 상태 |
|---|---|
| M0 결정 | 완료. Screen2Words 사용, 텍스트 질의만, 영어, query_rich를 주 실험으로 |
| M1 데이터셋 구축 | **완료(v0.1). 사용자 검토 대기 중** |
| M2 지표와 비-LLM 베이스라인 | 대기(데이터셋 승인 후 진행) |
| M3 Gemma 환경과 파일럿 | **보류**. Ollama 실행 금지(사용자 지시, 아래 참고) |
| M1b query_mid + 체크리스트 | **완료(2026-11-07 20:30). 사용자 검토 대기**(`data/review/mid_samples.md`) |
| M4 본 실험 / M5 보고 | 대기 |
| Preliminary study (S1, S2', S3, S4', S5) | **완료(2026-11-07 밤)**. `POC_study.md`, NOTES §11 |

## 1. 데이터셋 (결정됨)
- HF `rootsautomation/RICO-Screen2Words`(CC-BY-4.0, 22,417화면, 요약 5개, semantic annotations, view hierarchy, 스크린샷)를 쓴다.
- 공식 split은 앱 단위로 분리되어 있다(겹치는 앱 0개).

## 2. 데이터 변형 (구현됨, DATASET.md 참고)
- 질의: 결정론적 템플릿으로 만든다. `query_short`는 요약을 요청문으로 바꾼 것이고, `query_rich`는 여기에 요소 개수와 핵심 텍스트를 붙인 것이다(주 실험용).
  - 요약 선택: 5개 중 서로 가장 비슷한(consensus) 요약을 고른다.
- GT: semantic annotations를 HTML로 바꾼다. 구조 비교용 `gt_html_flow`와 시각 비교용 `gt_html_abs` 두 가지이고, 모든 요소에 `data-label`을 붙인다.
- 필터 F0–F7, 렌더 성공률 100%.
- 분할: test 500(카테고리 층화), pilot 50(test의 부분집합), dev 100(val), train_pool 14,522.

## 3. 베이스라인
| ID | 방법 | 비고 |
|---|---|---|
| B0 | Oracle(GT 자체) | 지표 상한, sanity check |
| B1 | 템플릿(키워드 → 고정 HTML) | 하한 |
| **B2** | **Retrieval**: TF-IDF로 query_rich와 가장 비슷한 train_pool 질의를 찾아 그 GT flow HTML 반환 | 비-LLM 주 베이스라인. GPU 불필요 |
| **L0** | Gemma zero-shot | LLM 주 베이스라인 |
| **L1** | Gemma few-shot(B2로 찾은 예시 3개) | 예시 3개는 입력 약 1.9k tokens 추가(GT flow 평균 571 tok × 3 + 질의) |

## 4. 모델: 로컬 Gemma 4 (선정안, 아직 받지 않음)
2026-11-07 기준 최신은 Gemma 4(2026-03-31)와 Gemma 4 12B Unified(2026-06-03)다. 라이선스는 Apache-2.0이다(NOTES.md §5).

| 후보 | Ollama 태그 | 크기 | 12GB 적합 |
|---|---|---|---|
| 소형 | `gemma4:e2b-it-qat` | 4.3GB | 여유 있음 |
| 중형 | `gemma4:e4b-it-qat` | 6.1GB | 맞음 |
| 대형 | `gemma4:12b-it-qat` | 7.2GB | KV 캐시 포함 약 9–10GB(ctx 8k 기준). GPU를 단독으로 쓸 때만 가능 |
| (제외) | `gemma4:26b-a4b`(16GB 이상), `31b`(19GB 이상) | | 12GB를 넘어 CPU offload가 필요하다 |

- **주의**: 측정 시점에 사용자가 게임 중이었고 GPU 6.2GB, 사용률 약 50%를 쓰고 있었다. 레이턴시 측정은 GPU가 비었을 때만 의미가 있다.
- Thinking은 끈다. 시스템 프롬프트에 `<|think|>` 토큰을 넣지 않으면 꺼진다.
- **실행기 보류**: Ollama(winget 설치본, `%LOCALAPPDATA%\Programs\Ollama`)는 CLI를 호출해도 트레이와 GUI 앱을 자동으로 띄운다. 이 때문에 게임 중인 사용자 화면에 창이 떴다.
  - 사용자 지시로 Ollama 실행, 모델 다운로드, 모델 실행을 모두 보류한다.
  - 재개하려면 사용자 허락이 필요하다. 창이 뜨지 않는 대안(예: pip로 설치하는 transformers 콘솔 실행)도 허락을 받은 뒤에 한다.
  - Gemma 모델 파일은 받지 않았다(첫 pull이 중단됨).

## 5. 평가 지표
- **토큰**: prompt_eval_count, eval_count(Ollama) 또는 tokenizer 기준 입력/출력 토큰.
- **레이턴시**: TTFT, 전체 시간, decode tokens/s, p50/p95.
  - 동시성 1, 워밍업 5회 후 측정한다.
  - GPU, 드라이버, 양자화, num_ctx, num_predict를 기록한다.
  - 측정 직전과 직후에 `nvidia-smi`로 GPU 메모리와 사용률을 기록하고, 다른 프로세스(XAIParkinson Grad-CAM, 게임)가 있으면 해당 run에 `gpu_shared=true`를 표시한다.
- **정확도(주 지표, 자동)**:
  1. 렌더 성공률(headless Playwright, 네트워크 차단)
  2. 요소 종류 multiset F1(출력 DOM 태그와 role을 componentLabel 체계로 정규화해 GT의 element_counts와 비교)
  3. 핵심 텍스트 재현율(key_texts가 렌더된 텍스트에 들어 있는 비율)
- 참고 지표: CLIP/SSIM(GT abs 렌더 대비), DOM tree edit distance.
- LLM-judge는 가장 큰 로컬 Gemma로 참고값만 낸다. 사람 평가는 나중에 한다.

## 6. 구조와 환경
```
GenerativeUI/
  PLAN.md NOTES.md DATASET.md .gitignore
  .venv/                     # uv, Python 3.11 (huggingface_hub, pyarrow, pandas, pillow, lxml, playwright, scikit-learn, tokenizers)
  .cache/                    # uv, HF, ms-playwright(chromium headless shell), gemma4 tokenizer.json
  data/raw/RICO-Screen2Words/   # 원본 parquet 2.4GB (신뢰하지 않는 데이터, 파싱만 함)
  data/processed/            # *.jsonl, screenshots/, build_report.json, validation_report.json
  data/review/               # 비교 이미지 20장
  scripts/download_data.py  inspect_schema.py  build_dataset.py  validate_dataset.py
  runs/                      # (예정) 실행 로그 JSONL
```
- 하니스(예정): `generate(prompt) -> {text, prompt_tokens, output_tokens, ttft, total, raw}`.
  - 한 호출에 JSONL 한 줄(원문 응답 포함)을 남긴다.
  - 캐시 key는 sha256(model|params|prompt)이다.
  - 시드와 temperature 0을 고정한다.

## 7. 체크리스트
### M1 데이터셋 (완료)
- [x] 다운로드, 스키마 확인, 앱 누수 확인(0)
- [x] 질의 생성(short, rich), GT flow/abs, 매핑표
- [x] 필터링과 규칙별 개수, 층화 분할, 재실행 시 결과 동일함 확인
- [x] 렌더 검증 100%, 비교 이미지 20장, 길이와 토큰 통계
- [ ] **사용자 검토와 승인**

### M2 지표와 비-LLM 베이스라인 (승인 후)
- [ ] 지표 구현(렌더, 요소 F1, 텍스트 재현율), B0 만점 확인
- [ ] B1, B2를 test 500에서 실행

### M3 Gemma (사용자 허락 후)
- [ ] 실행기 결정(Ollama 재허용 여부 또는 창이 뜨지 않는 대안)
- [ ] 모델 3종 준비
- [ ] 파일럿 50 × 모델 × {L0, L1}. 이 결과로 500개 전체 소요 시간을 추산하고, 6시간을 넘으면 샘플 수를 줄인다.

### M4–M5
- [ ] 본 실험, 레이턴시(GPU가 비었을 때), 참고용 judge, 결과 보고

## 진행 로그
- 2026-11-07 오전: 상용 API 기준 계획 초안 작성.
- 2026-11-07 오후: 사용자 결정에 따라 로컬 Gemma로 전환. 환경 구성(.venv, playwright headless), 데이터 다운로드, 데이터셋 v0.1 구축과 검증을 완료했다.
- 2026-11-07 오후: Ollama를 winget으로 설치했다. 설치 직후 앱 창이 떴고 바로 닫았다. 모델 pull 도중 사용자 지시로 Ollama가 종료되었는데, 진행 중이던 pull CLI가 앱을 다시 띄웠다. 즉시 해당 프로세스를 종료했고 pull 루프도 멈췄다. 이후 Ollama 관련 실행은 전면 보류한다.
- 2026-11-07 16:45: 사용자 지시에 따라 실행기를 transformers(.venv: torch 2.11+cu128, transformers 5.22.2, bitsandbytes 0.50.2)로 바꿨다.
  - Gemma 4 E2B, E4B, 12B QAT 체크포인트를 받았다(SHA는 NOTES §7).
  - metrics(src/genui/metrics.py), retrieval, bench_gemma.py(TTFT 스트리머, RAM·RSS 가드, CPU 4스레드, .gpu_busy), summarize_runs.py를 작성했다.
  - 비-LLM 베이스라인을 test 500에서 측정했다(B2 retrieval 요소 F1 0.613, 텍스트 재현율 0.251).
  - **레이턴시 측정은 아직 하지 않았다.** XAIParkinson images_e5_e6.py가 실행 중이고 여유 RAM이 4GB라서, 모델 로드 조건(여유 RAM 8GB 이상)을 기다리는 중이다.
- 2026-11-07 18:17: 파일럿 체인을 완료했다(E2B/E4B/12B rich_zs, E2B short_zs, E2B rich_fs, 각 10개). 12B는 NF4로 로드했고 피크 VRAM 8.4GB로 OOM 없이 동작했다.
- 2026-11-07 20:15: **test 500은 보류했다(사용자 결정).** 지연시간 비교는 파일럿 10개로 마무리한다. E4B와 12B에도 rich_fs를 추가로 측정한다.
- 2026-11-07 20:30: query_mid(목적 + 핵심 요소 2–4개, 개수·긴 원문 없음)와 체크리스트를 전 split에 추가했다(`data/processed/mid/`, `scripts/build_query_mid.py`, `src/genui/checklist.py`, `scripts/eval_mid_baselines.py`). mid 설정의 주 지표는 체크리스트 충족률, 요소/종류 F1은 보조 지표. 누수 검사 통과 test 100%, GT 자체 점수 1.000, B1 0.080 / B2 0.35–0.37(test). 다음: GPU가 비면 Gemma mid_zs/mid_fs 파일럿 10개(few-shot 예시는 train_pool의 query_mid 검색), 사용자 검토 후 템플릿 수정.

---

## 확정 연구 방향 (2026-11-07 확정, 같은 날 0단계 재구성)

### 우선순위 결정 (2026-11-08 00:55)

**제목:** Structure-Conditioned Speculative Draft Selection for HTML Generation

**핵심 문제:** 어휘 기반 speculative decoding은 반복되는 HTML 토큰 패턴이 서로 다른 DOM 문맥에서 나타나는 것을 구분하지 못해 부적절한 초안을 고른다.

**C1:** j* = argmax_{j∈J_t} E[A_j | p_t, z_t, z_j], A_j = LCP(d_j, y_{t+1:}).
- 정확 검증에서는 한 스텝 최선이 최적이다(관문 c). 그래서 순차 정식화는 하지 않는다.

**C2:** 부분 DOM과 후보 위치의 구조 관계로 수용 길이를 예측하는 경량 순위 모델.

**현재 근거**
- 학습 모델이 DOM 규칙보다 forward를 6.9회 적게 쓴다. 95% CI [4.5, 10.2].
- 구조 특징을 빼면 forward가 3.7회 늘어난다. 95% CI [3.3, 4.1].
- 이력 특징은 기여가 없다.

**핵심 검증 목표:** 학습 모델이 DOM 규칙과 기존 SuffixDecoding보다 실제 추론 비용을 줄이고, 그 개선이 특정 HTML 변환기의 반복 형식에만 의존하지 않음을 보인다.

**우선순위**
- **P0 관문 2b(학습 모델 GPU 실측):** 반드시 완료한다.
  - 판정은 두 단계다. 먼저 학습 모델과 DOM 규칙의 wall-clock을 비교하고, 다음으로 학습 모델이 prompt lookup보다 얻는 순이득을 본다.
  - prefill과 초안 선택 비용을 모두 포함한 end-to-end 시간으로 판정한다.
- **P1 HTML 표현 변화 강건성(핵심 가설 검증):** 다음 네 조건에서 측정한다.
  - Original
  - Formatting: 공백, 줄바꿈, 속성 순서
  - CSS 표현: inline 대 stylesheet
  - DOM 구조: 동등한 wrapper 추가와 제거
  - 모든 변형은 렌더 결과가 같은지 검증한다. 클래스 이름은 선택자와 class 속성을 함께 바꾼다.
  - "Reference만 변형"과 "Reference와 Target을 모두 변형"을 나눠서 본다.
- **P2 정식 SuffixDecoding 비교:** prompt lookup, DOM 규칙, SuffixDecoding, 학습 모델, SuffixDecoding + 학습 모델을 비교한다.
- **P3 O1 진단:** 진행 중인 분해 분석을 마무리만 한다.
- **보류**
  - O3(구조 기반 새 후보 생성): 현재 정책이 실패할 때만 검토한다.
  - O4(14,522개 전체 구조 인덱스): 범위가 커질 위험이 있어 보류한다.
- **후속(필수):** WebSight와 Design2Code 데이터, E4B와 12B 모델.
- **후속:** MacBook.

**판단 규칙**
- GPU에서 빠르고, 표현 변화에 강건하고, SuffixDecoding도 개선하면 방향을 유지한다.
- 빠르지만 SuffixDecoding보다 약하면 구조 기반 후보 생성을 검토한다.
- 표현 변화에서 효과가 사라지면 변환기에 특화된 최적화일 수 있으므로 문제를 다시 정의한다.
- GPU 이득이 없으면 모델을 경량화하거나 다른 추론 방법을 검토한다.

**원칙:** 학습 모델의 실제 성능과 일반화를 확인하기 전까지 새 알고리즘 모듈을 추가하지 않는다.

### 실행 목표와 분기 규칙 (2026-11-08 01:20)

**우선순위:** GPU 실측 → 오라클 격차 분석 → 표현 강건성 → 강한 기준선 비교 → 논문 초안

**분기 규칙**
- **관문 2b(학습 모델 GPU 실측)**: 결과를 네 경우로 나눈다.
  1. wall-clock이 유의하게 빠르다.
  2. 빠르지만 유의하지 않다.
  3. 호출(forward) 수만 줄고 시간은 줄지 않는다.
  4. 호출 감소도 재현되지 않는다.
- **O1·O2(오라클 격차 분석)**: 결과를 세 경우로 나눈다.
- **P1(표현 강건성)**: A(Reference와 Target 모두 변형)와 B(Reference만 변형)를 따로 본다. 성능이 크게 떨어지면 새 방법보다 먼저 변환기 편향과 후보 가용성을 분석한다.

**방향 유지 조건 (네 가지)**
1. GPU 실제 시간이 개선된다.
2. 구조 특징이 독립적으로 기여한다.
3. 표현 변화에 합리적으로 강건하다.
4. 정식 SuffixDecoding보다 추가 효용이 있다.

하나라도 충족하지 못하면 새 모델을 바로 만들지 않는다. 원인을 먼저 분석한 뒤 다시 검토한다.

**논문 원칙**
- 검증된 결과와 가설을 구분해 쓴다.
- 11배 속도 향상은 기존 기술(예시 기반 투기 디코딩)의 성과임을 명시한다.
- 독립된 test split 평가가 아직 없다는 점을 명시한다.
- 통계적으로 확인된 효과만 주장한다.
- 수치는 완료된 결과만 쓴다.

**보류:** O3(구조 기반 새 후보 생성), O4(전체 구조 인덱스).

### Research Direction: Structure-Conditioned Speculative Decoding for HTML Generation

**Core Problem.** Lexical speculative drafting cannot reliably distinguish repeated token patterns that appear in different HTML structural contexts.

- 제목에 "온디바이스"는 넣지 않는다.
- 실험의 자원 제약은 명시한다: RTX 4070 12GB, NF4, 배치 1.

### Stage 0: Feasibility and Novelty Validation (0-a, 0-b 최우선)

**0-a. 시뮬레이터에서 비교 (GPU 불필요)**
- 비교 대상: SuffixDecoding, DOM 인지 초안, SuffixDecoding + DOM 구조 특징.
- Go 조건: SuffixDecoding + DOM이 SuffixDecoding보다 유의하게 낫다. 판정은 요청 단위 짝 차이의 CI로 한다.
- DOM 규칙 단독이 SuffixDecoding을 이길 필요는 없다.

**0-b. 실제 GPU 시간 측정 (반복 측정)**
- 측정 항목:
  - target forward 횟수
  - 초안 제안 시간(DOM 매칭 비용 포함)
  - 검증 시간
  - 전체 wall-clock
  - peak memory
- Go 조건: DOM 인지 초안의 순효과가 양수다.

**0-c. myopic 오라클 대 sequential 오라클**
- myopic 오라클은 현재 스텝의 수용 길이를 최대화한다. sequential 오라클은 이후 정렬까지 고려해 전체 target 호출 수를 최소화한다.
- 처음에는 후보 집합과 rollout 깊이를 제한해 근사한다.
- Go 조건: sequential 쪽에 의미 있는 추가 여지가 있다.

### Go/No-Go Decision
- 학습 단계로 넘어가려면 **두 조건이 모두** 필요하다:
  - 0-a: HTML 구조가 강한 어휘 기준선(SuffixDecoding)보다 추가 가치를 준다.
  - 0-b: 실제 순 지연 이득이 양수다.
- 0-a나 0-b가 실패하면 연구 방향을 본격적으로 재검토한다.
- 순차 cost-to-go 학습은 0-c에서 sequential 오라클이 myopic 오라클보다 의미 있게 나을 때만 쓴다.
- 0-c가 실패하면 순차 결정 정식화를 내려놓고 "구조 기반 후보 선택" 문제로 축소한다. 연구 방향 자체는 유지한다.

### Stage 1: Learning
- 오프라인 시뮬레이션과 정책이 유도한 상태의 집계(DAgger 계열)로 초경량 구조 조건 초안 정책을 학습한다.
- 단서: 시뮬레이터는 저장된 target 출력을 재생하므로 미래 target 시퀀스가 고정되어 있다. 정책이 방문하는 상태는 다양하게 만들 수 있지만, 데이터 다양성과 일반화는 확보한 target 출력의 범위로 제한된다.

### Stage 2: Evaluation
- 평가 항목: 실제 추론 지연, 메모리, 출력 품질, 모델 크기와 질의 조건에 걸친 일반화.
- 12B와 rich 질의에서 구조 우위가 사라진 원인도 이 단계에서 다룬다.

### 오늘 PoC에서 넘겨받는 자원 (0-a, 0-b용)
- 디코드 루프와 KV 롤백 구현: `scripts/poc_rnsd.py`
- 평가 스크립트: `scripts/poc_eval.py`, `scripts/poc_table.py`, `scripts/poc_ci.py`, `scripts/poc_perc.py`
- k=1 후보 비교 실행기: `scripts/poc_cands.py`
- 시뮬레이터: `scripts/study_draft_sim.py`
- 순위 모델: `scripts/study_draft_learn.py`
- 예시 1개 × 후보 6개 × pilot 50 실측 원본(300회): `runs/poc/E2B_L2_cands6_raw.jsonl`
- 재채점 결과: `runs/retrieval/cands6_scored.jsonl`
