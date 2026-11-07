# GenerativeUI 조사 노트 (NOTES)

조사일: 2026-11-07. 웹 검색과 공식 페이지 열람만 했다. 다운로드와 API 호출은 하지 않았다.

## 1. "Screen2Text" 이름 조사
- 웹에서 "Screen2Text" 데이터셋 검색: 해당 이름의 데이터셋은 없다. 결과는 모두 Screen2Words로 연결된다.
- 판단: Screen2Words를 잘못 부른 것일 가능성이 높다. 사용자 확인이 필요하다.

## 2. 데이터셋 원자료

### Screen2Words
- 논문: Wang, Li, Zhou, Grossman, Li. "Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning", UIST 2021. https://arxiv.org/abs/2108.03353
- 공식 저장소: https://github.com/google-research-datasets/screen2words
  - `screen_summaries.csv`: 112,085개 요약, 22,417개 화면.
  - `split/`: 논문에서 쓴 train/dev/test.
  - 이미지와 VH는 들어 있지 않다. RICO에서 별도로 받아야 한다.
  - README에서 라이선스 표기를 확인하지 못했다(fetch로 본 화면에는 없었음). **확인 필요.**
- HF 미러: https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words
  - 라이선스 CC-BY-4.0. RICO-SCA 부분집합만 포함한다(VH가 누락되거나 부정확한 화면 제외).
  - split: train 15,743 / val 2,364 / test 4,310 (합계 약 20.4k). 다운로드 약 2.47GB, 전개 시 약 5.09GB.
  - 컬럼: screenId, captions(list[str]), file_name, app_package_name, play_store_name, category, average_rating, number_of_ratings, number_of_downloads, file_name_icon, file_name_semantic, semantic_annotations(JSON str), view_hierarchy(JSON str), image, image_icon, image_semantic.
  - VH 내부 구조는 카드에 설명이 없다. RICO 원본 형식(class, bounds, text, resource-id, children, visibility 등)으로 추정한다.
- **불확실**: 공식 split이 앱 단위로 분리되어 있는지 확인하지 못했다. HF split이 공식 split과 같은지도 모른다.

### RICO
- Deka et al., UIST 2017. 앱 9.7k개, 화면 약 72k개. 스크린샷, VH, 의미 주석(25개 UI 컴포넌트 종류, Liu et al. 2018), 상호작용 trace를 담고 있다.
- http://interactionmining.org/rico
- Voxel51과 HF 미러는 CC-BY-4.0으로 표기한다: https://docs.voxel51.com/dataset_zoo/datasets_hf/rico.html. **공식 라이선스는 미확인.**
- 회고 논문: https://research.google/pubs/an-early-rico-retrospective-three-years-of-uses-for-a-mobile-app-dataset/

### Widget Captioning
- Li et al., EMNLP 2020. 약 61k개 UI 요소에 약 162k개 캡션(수치는 기억에 근거, 재확인 필요).
- https://huggingface.co/datasets/rootsautomation/RICO-WidgetCaptioning

### WebSight
- https://huggingface.co/datasets/HuggingFaceM4/WebSight (논문 arXiv 2403.09029)
- v0.1 823k행, v0.2 1.92M행. CC-BY-4.0. 합성 데이터이고 HTML은 Tailwind 2.2.19 CDN을 쓴다.
- 컬럼: image, text(HTML, 344–6.3k자), llm_generated_idea(25–566자).
- 이미 (아이디어 → HTML) 쌍이라 웹 UI 버전으로 바로 쓸 수 있다. 다만 합성이라 분포가 단순하고 LLM 학습 데이터에 포함됐을 가능성(오염)이 있다.

### Design2Code
- https://huggingface.co/datasets/SALT-NLP/Design2Code (논문 arXiv 2403.03163)
- C4 validation 웹페이지 484개(뷰어에는 485행). ODC-BY. {id}.html과 {id}.png 쌍이고 이미지는 rick.jpg 플레이스홀더로 치환되어 있다.
- 논문 지표: Block-Match, Text, Position, Color, CLIP. 이 목록은 HF 카드에 없었고 논문 내용을 기억한 것이다. 구현을 참고할 때 원 코드(GitHub NoviScl/Design2Code)를 확인해야 한다.

### WebUI
- Wu et al., CHI 2023. https://arxiv.org/abs/2301.13280, https://uimodeling.github.io
- 약 400k 웹페이지, 수집 비용 약 $500. 라이선스와 다운로드 형식은 미확인.

## 3. API 단가 원자료 (확인일 2026-11-07, USD / 1M tokens, 표준 요금)

### Anthropic: https://platform.claude.com/docs/en/about-claude/pricing , 모델 ID: https://platform.claude.com/docs/en/about-claude/models/overview
- claude-haiku-5-5: 입력 $0.10 / 출력 $0.50 (프롬프트 10만 토큰 이하). 10만 토큰 초과는 $0.50 / $2.50. 캐시 히트 $0.01.
- claude-haiku-4-5: $1 / $5.
- 참고: claude-sonnet-5-5 $2 / $10 (judge 후보).
- Batch 50% 할인. Claude 4.7 이후 토크나이저는 같은 텍스트에 약 30% 더 많은 토큰을 낸다.
- Haiku 5.5는 adaptive thinking이고 기본 effort가 medium이다. thinking을 끌 수 있는지(`thinking.types.disabled.supported`)는 Models API로 확인해야 한다.

### OpenAI: https://developers.openai.com/api/docs/pricing (openai.com/api/pricing 은 403)
- gpt-6-luna $0.10 / $0.50 (cached $0.01)
- gpt-5.6-luna $0.20 / $1.20
- gpt-5.4-mini $0.75 / $4.50, gpt-5.4-nano $0.20 / $1.25
- gpt-5-mini $0.25 / $2.00, gpt-5-nano $0.05 / $0.40
- gpt-4.1-mini $0.40 / $1.60, gpt-4o-mini $0.15 / $0.60
- Batch는 표준 요금의 절반.
- "luna" 라인의 성격(추론 모델 여부, 추론 effort 설정)은 확인이 필요하다.

### Google Gemini: https://aistudio.google.com/docs/pricing (ai.google.dev 주소에서 리다이렉트됨)
- gemini-3.8-flash, gemini-3.6-flash: $0.75 / $3.75. 2026-12-31까지의 프로모션 가격이고 2027-01-01부터 $1.50 / $7.50.
- gemini-3.5-flash-lite $0.30 / $2.50
- gemini-3.1-flash-lite $0.25 / $1.50
- gemini-3-flash-preview $0.50 / $3.00 (legacy)
- gemini-2.5-flash-lite $0.10 / $0.40
- gemini-2.5-flash는 deprecated이고 2026-12-01부터 3.5 Flash-Lite로 라우팅된다.
- 출력 단가에 thinking 토큰이 포함된다. 무료 티어가 있지만 rate limit이 낮고 데이터 사용 조건이 있어 측정용으로는 유료 티어를 권장한다.

### DeepSeek: https://api-docs.deepseek.com/quick_start/pricing/
- deepseek-flash (DeepSeek-V4.1-Flash-1017): 캐시 미스 입력 피크 $0.30 / 오프피크 $0.15, 출력 피크 $1.20 / 오프피크 $0.60, 캐시 히트 $0.006 / $0.003. 컨텍스트 1M.
- deepseek-v4-pro: $1.32 / $3.96 (피크).
- 피크 시간은 평일 UTC 01–04시, 06–10시(중국 공휴일 제외)다. KST로는 10–13시, 15–19시. 그 외는 오프피크(반값).
- deepseek-v4-flash는 retired 별칭이고 Flash로 라우팅된다.

### Alibaba Qwen (Model Studio International / Singapore): https://www.alibabacloud.com/help/en/model-studio/model-pricing
- qwen3.8-flash $0.15 / $0.47
- qwen3.7-flash: 입력 32K 이하 $0.03 / $0.13, 32–256K $0.10 / $0.40
- qwen-flash (2025-07-28): $0.05 / $0.40
- qwen3.7-plus $0.4 / $1.6 (한시적 20% 할인 표기), qwen-plus $0.4 / $1.2 (thinking 출력은 $4)
- 2차 출처(가격 집계 사이트)는 서로 수치가 조금씩 다르다. 공식 페이지 값을 채택했다.

## 4. 불확실한 점과 확인할 일
- Screen2Words 공식 GitHub의 라이선스 파일, split이 앱 단위로 분리되어 있는지 여부.
- semantic_annotations JSON의 실제 키 이름(componentLabel, bounds, text, children으로 추정).
- 각 저가 모델에서 thinking/reasoning을 끄거나 최소화하는 파라미터와, 그 설정이 품질에 주는 영향. 출력 토큰 추산(1,500)이 크게 달라질 수 있다.
- Python 3.13에서 open_clip, torch CUDA, playwright 휠이 있는지.
- 평균 HTML 출력 길이 1,500 토큰은 추정치다. 파일럿에서 다시 잰다.
- 단가는 자주 바뀐다(Gemini 프로모션은 2026-12-31 종료, Qwen 한시 할인). 실험 직전에 `configs/prices.yaml`을 다시 확인한다.
- Widget Captioning 수치와 Design2Code 지표 목록은 기억에 근거한 것이라 원문 확인이 필요하다.

## 5. 로컬 Gemma 조사 (2026-11-07)
- 릴리스 로그(https://ai.google.dev/gemma/docs/releases):
  - 2026-03-31 Gemma 4(E2B, E4B, 26B A4B, 31B)
  - 2026-04-16 MTP 변형
  - 2026-06-03 Gemma 4 12B Unified
  - 2026-10-06 EmbeddingGemma 2
- 라이선스: HF 모델 카드 `google/gemma-4-12b-it` 메타데이터가 `apache-2.0`이다. 2차 출처들도 Gemma 4부터 기존 Gemma Terms of Use 대신 Apache 2.0을 쓴다고 전한다.
  - 상표권("Gemma" 명칭)은 Apache 2.0에 포함되지 않는다.
  - E2B, E4B, 26B, 31B 카드의 라이선스는 직접 확인하지 않았다. **확인 필요.**
- Ollama 태그(https://ollama.com/library/gemma4/tags):
  - e2b-it-qat 4.3GB, e4b-it-qat 6.1GB, 12b-it-qat 7.2GB, 12b-it-q4_K_M 8.0GB, 26b-a4b-it-qat 16GB, 31b-it-qat 19GB
  - 컨텍스트: e2b/e4b 128K, 12b 이상 256K
- Thinking: 시스템 프롬프트 맨 앞에 `<|think|>`를 넣으면 켜진다. 빼면 꺼지지만, 12B 이상은 빈 thought 블록을 출력한다.
- Ollama 0.40.15(2026-11-03, MIT)를 winget으로 설치했다.
  - CLI를 호출해도 서버가 없으면 GUI 앱("ollama app")을 자동 실행한다. 게임 중인 사용자에게 창이 뜨는 문제가 있다.
  - 사용자 지시로 보류 중이다. 관련 프로세스는 모두 종료된 것을 확인했다.
  - GitHub의 standalone zip(ollama-windows-amd64.zip)을 받아 실행하려던 시도는 권한 분류기가 거부했다.
- 2026-11-07 15시경 GPU 상태: Overwatch와 pythonw(XAIParkinson로 추정)가 실행 중이었고 6.2GB/12.3GB, 사용률 약 50%였다. 레이턴시 측정은 GPU가 빈 시간대에 해야 한다.

## 6. 데이터셋 구축 중 발견한 사실
- 실제 데이터는 22,417화면이다(train 15,743 / val 2,364 / test 4,310). HF 카드의 "20.4k"는 부정확하다. 모든 화면에 요약이 정확히 5개 있다.
- 공식 split 간 겹치는 앱은 0개다(train 4,390 / val 625 / test 1,254개 앱).
- semantic_annotations 노드 키: class, ancestors, bounds[x1,y1,x2,y2], clickable, resource-id, text, componentLabel(25종), iconClass, textButtonClass, children. 루트에는 componentLabel이 없다.
- componentLabel 빈도(상위): Text 144k, Icon 55k, Image 54k, List Item 43k, Text Button 42k, Toolbar 12k, Input 8.5k.
- 일부 화면은 다이얼로그만 주석되어 있다(예: 21번은 resolver 팝업만 있음). 배경 화면은 GT에 없다.
- 첫 빌드 때 void 요소(input, img) 아래의 자식과 `<p>` 중첩 때문에 DOM 요소 수가 맞지 않았다(test 렌더 검증 95.4%). 자식을 형제로 내보내고 p를 div로 바꾸도록 수정해 100%가 되었다.
- Gemma 4 tokenizer.json은 HF `google/gemma-4-12b-it`에서 받았다(`.cache/gemma4_tokenizer/`). 길이 통계에 사용했다.

## 7. 로컬 Gemma 모델 파일 (2026-11-07 다운로드, 실행기: transformers. Ollama는 사용 금지)
- HF에서 별도 동의나 토큰 없이 받을 수 있었다(gated=False). 라이선스는 모두 HF 메타데이터상 `apache-2.0`이다.
- 받은 것은 Google 공식 QAT 체크포인트 `*-it-qat-q4_0-unquantized`다. 4bit(q4_0) 양자화를 전제로 학습한 가중치를 bf16으로 저장한 것이다. 실행할 때 bitsandbytes NF4로 4bit 로드한다.
  - NF4는 q4_0과 같은 형식이 아니다. 근사이므로 결과에 함께 기록한다.

| 모델 | HF repo@revision | 파일 | SHA256 (model.safetensors) |
|---|---|---|---|
| E2B | google/gemma-4-E2B-it-qat-q4_0-unquantized@6befbaca7398925921802abd1f277b495b78b738 | models/gemma-4-E2B-it-qat-q4_0-unquantized (9.6GB) | 33fe0cece08fb527ffefbd1a3a9ce73bd71073727993a283506293e5c6bf0137 |
| E4B | google/gemma-4-E4B-it-qat-q4_0-unquantized@476025a01dbf99361c062bbeca3d6a76bb4c4566 | models/gemma-4-E4B-it-qat-q4_0-unquantized (15GB) | ad8ef515194b15ab13b6f98d54d37652b691c9a2c787cdf7ed5f951f5ed2c7fa |
| 12B | google/gemma-4-12B-it-qat-q4_0-unquantized@b6ed86275a6a5735884e208bfed95b445a684ca2 | models/gemma-4-12B-it-qat-q4_0-unquantized (23GB) | 26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0 |

- `models/SHA256SUMS.txt`에도 같은 값을 저장했다.
- 메트릭 정규화기 검증: GT flow HTML에 정규화기(data-label 무시, 태그·role·크기 규칙)를 적용한 결과와 실제 data-label 개수를 비교했다. test 500 기준 평균 요소 F1은 0.926이다. 즉 정규화기 자체에 약 7%의 오차가 있다.
- 비-LLM 베이스라인(test 500, `runs/baselines/summary_test.json`):

| | 요소 F1 | 종류 F1 | 텍스트 재현율 |
|---|---|---|---|
| B0 oracle | 1.0 | 1.0 | 1.0 |
| B1 template | 0.500 | 0.683 | 0.010 |
| B2 retrieval | 0.613 | 0.764 | 0.251 |

## 8. 결정: test 500 보류 (2026-11-07 20:15, 사용자 지시)
- Gemma 지연시간 비교는 pilot 앞 10개로 끝낸다. test 500은 돌리지 않는다.
- 이유: bitsandbytes NF4 디코딩이 10–18 tok/s라서 샘플당 25–140초가 걸린다. 500개 × 3모델 × 조건이면 수십 시간이 필요하다.
- 따라서 모든 수치는 n=10이고 p95는 사실상 최댓값이다. 정확도 차이를 유의하다고 보기 어렵다.
- 최종 비교 조건은 E2B, E4B, 12B × {rich_zs, rich_fs}이고, E2B만 short_zs를 추가로 측정했다. 결과는 `runs/gemma/summary.md`에 있다.

## 9. query_mid + 체크리스트 (2026-11-07 20:30)
- 목적: query_rich는 요소 개수와 핵심 텍스트를 모두 알려줘서 요소 F1이 사실상 "복사" 점수가 된다. 그래서 중간 수준 질의(목적 + 핵심 요소 2–4개, 개수·긴 원문 없음)와 기계 채점 체크리스트를 추가했다. 상세는 DATASET.md 마지막 절.
- GPU 벤치마크 실행 중이라 전부 CPU 규칙 기반으로 만들었다(우선순위 BelowNormal, CPU affinity 2코어, 브라우저 1개, RSS 가드 6GB, 최대 RSS 0.5GB).
- 체크리스트 항목은 GT 렌더에서 먼저 검증해 통과한 것만 쓴다. 후보 통과율 약 96%(test 3,673/3,831). 탈락 원인은 정규화기 특성이다: 48px 이하 Image는 Icon으로 분류, 텍스트 없는 버튼은 Icon, 체크박스 라벨이 부모 밖에 있음, flow 레이아웃에서 nav가 화면 아래 25%에 있지 않음.
- 독립 검증: 검증에 쓰지 않은 GT abs HTML도 pilot/dev/test에서 체크리스트 1.000.
- 체크리스트 점수(check_score, 항목 충족 비율) — `runs/baselines_mid/summary.json`:

| | pilot | dev | test | test 요소 F1 |
|---|---|---|---|---|
| B0 GT flow / abs | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.0 |
| B1 template | 0.105 | 0.067 | 0.080 | 0.500 |
| B2 retrieval(query_rich) | 0.318 | 0.332 | 0.348 | 0.613 |
| B2 retrieval(query_mid) | 0.382 | 0.374 | 0.367 | 0.581 |

- 참고: 저장된 Gemma 파일럿 출력(질의는 rich/short였음, n=10)을 체크리스트로 채점하면 rich_zs 0.74–0.80, rich_fs 0.94(E2B·E4B), E2B short_zs 0.17. 요소 F1보다 차이가 뚜렷하다. query_mid로 Gemma를 돌린 결과는 아직 없다(GPU 사용 중).

## 10. 렌더 공간 검색 (few-shot 예시 선택, 2026-11-07 21:30)
- 상세: `RETRIEVAL.md`. 코드: `src/genui/canon.py`, `scripts/retr_struct.py`. 결과: `runs/retrieval/`.
- κ(정규형) 위 4-gram 복사 커버리지 ω₄를 최대화하도록 예시를 고른다. s = β·cos(φ̂(x), φ(e)) + (1−β)·cos_tfidf. φ̂는 TF-IDF(query_mid)→φ ridge(α=10)이고, β=0.9는 dev에서 정했다.
- dev 100, top-1 ω₄: TF-IDF 0.807, struct 0.836(차이 +0.029, CI [+0.011, +0.048]), oracle 0.886.
  - body만 보면 0.633 / 0.697 / 0.794.
- 복사 베이스라인: 요소 F1은 +0.052로 올랐고, 체크리스트는 −0.083으로 떨어졌다(유의). 구조와 내용 사이에 trade-off가 있다. top-3 합집합 커버리지는 차이가 없다.
- GPU 에이전트용 `runs/retrieval/pilot_top3_{tfidf,struct,oracle}.json`(pilot 50개)을 전달했다. tfidf 파일은 poc_rnsd의 선택을 50/50 재현한다.
- 남은 일: Gemma로 ω와 수락률·지연의 관계 측정(GPU 에이전트), β를 낮추거나(0.6) 체크리스트 키워드 항을 추가하는 변형.
- 21:40 추가: mix(struct top-1 + TF-IDF top-2)는 top-3 합집합 0.877, TF-IDF 대비 +0.009(유의)다. 파일은 `pilot_top3_mix.json`이다. β=0.6 struct는 dev에서 ω +0.027, 체크리스트 −0.002(손실 없음), 요소 F1 +0.077이다. 다만 pilot 50에서는 체크리스트가 −0.055다. RETRIEVAL.md의 "Added conditions" 절을 참고.

## 11. Preliminary study (2026-11-07 밤, generativeui-study, CPU 전용)
- 상세 내용은 `POC_study.md`에 있다. 스크립트는 `scripts/study_*.py`, 결과는 `runs/study/`, 그림은 `runs/study/figures/`에 있다.
- 핵심 수치:
  - π: zs 0.65–0.72, fs 0.76–0.97.
  - 새 정보(N)의 48–56%는 CSS에 있다. 화면에 보이는 텍스트는 토큰의 2–4.5%다.
  - S_null은 22–29%, S_tol은 12–14%다.
  - η_L2(κ)는 0.13–0.35다. ddmin으로 재면 E2B fs 기준 η_L2 0.38, η_L4(ε=.05) 0.51이다.
  - `<body>` 이전 토큰이 41–69%다. L_perc/T는 0.70–0.85다.
  - ω(e,y)와의 상관: total_s ρ −0.57, elem F1 ρ +0.46(n=70). PoC에서 tok/forward와 ρ 0.88–0.92다.
- κ는 `scripts/study_canon.py`로 직접 구현했다. `src/genui/canon.py`가 생기면 정의가 맞는지 대조해야 한다.
- 사고 기록: 21:05에 정리 명령의 패턴이 겹쳐 PoC 에이전트의 `poc_chain2.sh`를 종료했다. main에 보고했고, 재실행은 PoC 에이전트가 맡는다. 이후로는 내가 띄운 PID만 종료한다.
- S6 (21:20): 생성 순서 재배치 실험.
  - body-first(CSS를 맨 뒤로)는 스타일 없는 중간 프레임 때문에 L_perc/T가 오히려 나빠졌다(E2B fs 0.73 → 0.97). 가설 기각.
  - CSS 인터리브(규칙을 처음 쓰이는 블록 바로 앞으로)는 최종 화면이 같은 상태에서 L_perc/T를 7조건 모두 줄였다. 0.70–0.85에서 0.56–0.67로, 상대 −13~−26%다.
  - 남은 일: 원래 규칙 텍스트를 그대로 써서 토큰 수가 같은 버전 만들기, 계층적 Smith.
- 인터리브 도구(21:5x):
  - `src/genui/interleave.py`에 `interleave(html)`(i', 규칙 원문을 그대로 옮김)와 `perceptual_latency(html, token_times)`를 넣었다.
  - `runs/retrieval/pilot_exemplar_html_interleaved.json`: pilot 50개분 예시 150개다. 150개 모두 원본과 픽셀이 같고, 토큰은 +2.4%다. GPU 검증(예시를 인터리브로 보여 주면 출력도 인터리브가 되는가)은 PoC 에이전트가 맡는다.
- 초안 위치 시뮬레이션(22:30, `POC_study.md` §9):
  - 표준 PLD 재생이 로그와 10/10 일치한다.
  - poc greedy n=40에서 DOM 규칙은 −7.9% forward로, 최선의 일반 실행 정책(n8)보다 짝 차이 CI [4.4, 8.9]만큼 낫다. gemma 근사 데이터에서는 유의하지 않다.
  - 오라클 위치는 −43%다. 격차의 81–85%가 남는다.
  - 단순 학습 순위 모델은 −6.6%로 DOM 규칙보다 나쁘다.
- 22:55 latency oracle gap (E2B L2, k=1, pilot 50×6): 6개 후보 중 가장 빠른 것이 tfidf1보다 −4.60초 [−6.21, −3.11](−37%)이고 품질 손실은 없다. 시간 분산의 75%는 출력 길이, 25%는 수용 효율이다. 배포 가능한 특징으로 학습한 ranker는 pairwise 정확도 0.57로, 격차의 13%만 회수했다(유의하지 않음). 판정은 "차이 크지만 예측 불가"다. GPU 측 채점기가 61행에서 죽어서 `scripts/retr_cands6_score.py`로 300행 전체를 다시 채점했다(`runs/retrieval/cands6_scored.jsonl`).

## 9. 결정 로그: 연구 방향 확정 (2026-11-07 22:55)

오늘 PoC(RNSD, 검색 조건 C0–C7, 지각 지연)를 마치고 연구 방향을 **Structure-Conditioned Sequential Speculative Decoding for HTML Generation**으로 확정했다. 상세는 PLAN.md "확정 연구 방향" 절에 있다.

**확정 근거**
- 예시를 쓰는 투기 디코딩만으로 greedy 대비 4.5–5.7배 빨라졌다. 화면 절반이 보이는 시점(t50)도 20.3초에서 3.3초로 줄었다(C1 → C4).
- 남은 비효율은 반복 구조 때문에 초안이 엉뚱한 위치를 고르는 데서 온다(POC_study §9).

**핵심에서 뺀 것과 이유**
- 렌더 동치 수용(L1/L2/L4)을 뺐다. greedy target은 이름 변경에 대해 등변(equivariant)이 아니어서 렌더 무손실이 성립하지 않았다(L2 v1은 0/10, v2는 2–6/10). L1 규칙은 아예 발동하지 않았다.
- CSS 순서 변경(인터리브)을 뺐다. greedy에서는 t50이 줄었지만 투기 디코딩 위에서는 이득이 없었다.
- 검색 쪽 조건은 결과가 엇갈렸다. struct 검색(C4)은 체크리스트를 +0.033 올렸다(유의). mix 검색(C6)은 유의하지 않았다. 이 결과는 부록으로만 남긴다.

**측정상 주의**
- 같은 조건의 greedy를 다시 돌리면 출력이 비트 단위로 재현된다.
- 반면 여러 토큰을 한 번에 검증하는 forward는 bf16/NF4 수치 차이 때문에 greedy와 출력이 갈라진다. PLD도 10개 중 3개만 greedy와 동일했다.
- 체크리스트 지표는 pilot 10개에서 포화된다. 정확도를 비교하려면 50개 이상이 필요하다.

**chain6 채점 (완료)**
- chain6(예시 1개 × 후보 6개, 300회 실측)의 GPU 실행은 22:39에 50/50 요청을 마쳤다. 원본은 `runs/poc/E2B_L2_cands6_raw.jsonl`에 있다.
- 검색기 에이전트가 이 원본 300행 전체를 다시 채점했다. 결과는 `runs/retrieval/cands6_scored.jsonl`이고, poc 채점기 결과와 59/60이 일치한다. 불일치 1건은 크래시가 난 요청이다.
- 분석은 RETRIEVAL.md의 "Latency oracle gap" 절에 있다.

**남은 일**
- poc 채점기(`scripts/poc_cands.py --score`)에서 나는 Page crashed 오류의 원인을 수정한다.
- 2026-11-07 (추가): 0단계를 가설 반증 실험으로 재구성했다. SuffixDecoding 비교를 최우선으로 둔다(PLAN.md "확정 연구 방향" 절의 Stage 0, Go/No-Go 참고).
- 23:35 관문 3 (sequential vs myopic, CPU 1코어): any-position 오라클에서 정확한 DP의 forward 수가 myopic과 70/70 샘플 모두 같다(차이 0). 도달점 r(pos)가 단조라서 myopic이 최적임이 증명된다. DOM top-M 후보 안에서도 lookahead와 DP의 이득은 0.04% 이하다. 판정: 실패. C1은 한 스텝 효용 최적화로 축소한다. 남은 개선 여지는 후보를 기대 수용 길이로 순위 매기는 것이다(DOM best-of-8 139 vs DOM 177 vs 오라클 103). RETRIEVAL.md "Sequential vs myopic oracle" 절.
- 관문 1 (23:24, `POC_study.md` §10):
  - SuffixDecoding(근사, 요청 내 풀, α=16) + DOM 가중을 SD와 비교했다. 짝 차이는 poc 3.8 [2.6, 5.1], gemma 5.4 [3.3, 7.6] forward로 **통과**다.
  - 효과는 2–3%로 작다. poc에서는 §9의 DOM 규칙이 SD+DOM보다 낫다.
- Stage 1–2 (00:29, `POC_study.md` §11): DOM top-8 학습 순위 모델 실험.
  - 데이터 n=444, 앱 단위 CV. DOM 규칙 − 학습 = 6.9 [4.5, 10.2] forward이고, top-8 오라클 격차의 32%를 메웠다.
  - 구조 특징을 빼면 +3.7 [3.3, 4.1]로 나빠진다(기여 확인). 이력 특징은 기여가 0이다.
  - 순이득은 MLP +376 ms/요청, HGB +119 ms/요청이다.
- 2026-11-08 00:55 우선순위 결정. 연구 제목을 Structure-Conditioned Speculative Draft Selection for HTML Generation으로 정했다. C1은 한 스텝 수용 길이 기대값의 argmax 후보 선택이다. 관문 c에 따라 순차 정식화는 하지 않는다. C2는 구조 기반 경량 순위 모델이다.
  - 근거:
    - 학습 모델이 DOM 규칙 대비 forward −6.9 [4.5, 10.2]다.
    - 구조 ablation은 +3.7 [3.3, 4.1]이다.
    - 이력 특징은 기여가 없다.
  - 우선순위:
    - P0: 관문 2b GPU end-to-end 실측. 학습 모델을 DOM 규칙과 비교한 뒤, prompt lookup 대비 순이득을 본다.
    - P1: HTML 표현 변화 강건성. 렌더 동일성을 검증하고, Reference만 변형한 경우와 Reference·Target을 모두 변형한 경우를 구분한다.
    - P2: 정식 SuffixDecoding 비교.
    - P3: O1 진단 마무리.
    - 보류: O3, O4.
  - 원칙: 학습 모델의 실제 성능과 일반화를 확인하기 전에는 새 알고리즘 모듈을 추가하지 않는다. 판단 규칙은 PLAN.md "우선순위 결정" 소절에 적었다.
- 01:15 ranking vs construction (진단용): SD→O2 격차 중 ranking(SD→O1)은 poc 45% / gemma 60%, construction(O1→O2)은 poc 55% / gemma 40%다. 70개 합쳐서 construction − ranking = −3.2 [−9.8, +3.6]이라 construction이 더 크지 않다. ranking 격차의 약 절반은 SD의 초안 길이 상한 ceil(2m)이다. 문맥 밖 구조 생성(O3−O2)은 거의 0이고, 실행 가능한 구조 점프 정책은 m≤1에서 오히려 나빠진다. RETRIEVAL.md "Ranking vs construction gap" 절.
- 01:40 O1/O2 판정(오라클은 미래 정답을 쓰는 상한): O2가 O1보다 29–33% 적어 "후보 검색 한계" 규칙에 해당한다. 학습 순위 모델(hgb)은 O1보다 11–19%, O2보다 40–43% 높아서 보류 조건이 아니다. 접미사 일치가 없는 스텝의 70%(poc)/44%(gemma)는 직전 토큰이 문맥에 처음 나온 경우다(새 값·텍스트 직후). 다음 토큰은 태그 경계 약 40%, CSS 값과 속성 약 42%다.
- 정식 SuffixDecoding 비교 (01:58, `POC_study.md` §12). ArcticInference 소스를 읽기만 하고 그대로 재현했다.
  - SD의 이득은 global 캐시에서 나온다. local만 쓰면 PLD보다 나쁘다.
  - 결과(forward): SD f1 109.3, f16 94.9. 우리 학습 순위 모델 단독은 121.8로 SD에 진다.
  - SD+학습(+DOM) 대비 SD 단독의 차이: f1 1.3 [0.7, 1.8], f16 2.4 [1.9, 3.0]. 조건 4는 충족했지만 효과는 1–3%로 작다.
- 02:xx 공식 SuffixDecoding 대조(D): 소스를 텍스트로 읽고 규칙을 직접 이식해 측정했다. C++ 빌드는 권한 정책으로 거부되어 하지 않았다. 재현은 study 값과 114/114 일치한다. 알고리즘(점수, 동률, m≤63, 트리)의 차이는 영향이 없다. 설정 차이 두 가지가 영향을 준다. 하나는 global에 자기 응답을 포함하는 공식 규칙으로 −1.1%이고, f1에서 SD+학습 이득(1.3)과 크기가 같으므로 다시 측정해야 한다. 다른 하나는 MAX_SPEC 64로 −5.6%이고 검증 토큰이 +13%다. 외부 데이터(F): WebSight 200, Design2Code 484를 준비했다(Design2Code는 16개만 생성 예산에 들어감). GT 대리 재생에서 DOM 이득은 WebSight −6.1%, S2W −3.6%다. DATASET.md "External data (prep)".
- 04:35 야간 작업 정리 (상세는 `POC_study.md` §12b–§15):
  - 공식 규칙으로 다시 돌린 결과, SD+DOM의 이득은 f1에서 0.5 [0.1, 0.9](세트별로는 유의하지 않음), f16에서 2.9 [2.3, 3.6], MAX_SPEC 64에서 1.0 [0.4, 1.5]이다. 그래서 주장을 "f16에서 1–3%"로 고쳤다.
  - 표현 변화(B 실험)에서 SD+DOM의 이득이 css·dom 조건에서 2–3배 커졌다. 다만 그 대부분은 재순위화 효과이고, DOM 특징 자체의 몫은 1.4–1.7이다.
  - 캐시 감사: 누수는 0이다. 서빙 순서로 돌려도 이득의 79%가 남고, 상용구를 지워도 82%가 남는다.
  - 오라클 비교: local+global 위치 오라클이 46.9로 가장 큰 잠재 이득을 보였다. 다음 연구 후보로 기록한다.
