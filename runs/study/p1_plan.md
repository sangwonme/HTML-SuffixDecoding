# P1 GPU 실행 계획: HTML 표현 변화 강건성

작성: 2026-11-08 01:1x, generativeui-study. 준비물은 CPU에서 만들었다. GPU 실행과 poc_rnsd.py 수정은 PoC 에이전트가 맡는다.

## 1. 준비물
- 변환 함수: `src/genui/perturb.py`. 모두 `f(html, seed, page=None) -> html` 형태이고, 실패하면 원본을 그대로 돌려준다.
  - `fmt`: 태그 사이에 줄바꿈과 들여쓰기를 넣는다(`<style>`, `<title>` 안은 제외). 속성 순서를 섞고, 일부 값은 작은따옴표로 바꾼다.
  - `css`: `<style>` 규칙을 해당 요소의 inline `style`로 펼친다.
    - 적용 순서는 특이도, 그다음 소스 순서이고, 기존 inline이 맨 마지막이다.
    - 의사 클래스·의사 요소 규칙과 매칭이 안 되는 규칙만 `<style>`에 남긴다.
    - 결과는 브라우저 DOM 직렬화로 만든다.
    - 반대 방향(inline → class)은 이번에 구현하지 않았다.
  - `dom`: body 안 요소의 약 40%를 `<div style="display:contents">`로 감싼다. 이번에는 wrapper 추가만 구현했고, 제거는 없다.
  - `cls`: 클래스 이름을 무작위로 바꾼다. class 속성과 `<style>` 선택자를 함께 바꿔서 서로 맞게 유지한다.
- 예시 파일: `runs/retrieval/pilot_exemplar_html_{fmt,css,dom,cls}.json`. 형식은 `{train_id: html}`이다.
  - 대상은 pilot 50의 TF-IDF(query_mid) top-3이고, 고유 예시는 150개다. poc_rnsd와 같은 검색을 썼다.
  - 전체 페이지 스크린샷이 원본과 픽셀 단위로 같은 것만 남겼다.
  - 통과율과 토큰 변화는 `runs/study/p1_perturb_report.json`에 있다.

## 2. 조건
- **(A) Reference와 Target 모두 변형**: 프롬프트 예시와 초안 원천 모두 변형된 HTML이다. 모델 출력도 새 형식을 따를 것으로 본다.
  - 현재 `--exemplar-html` 옵션 그대로 실행할 수 있다(예: `--exemplar-html runs/retrieval/pilot_exemplar_html_css.json`).
  - 이때 PLD/DOM/학습 drafter의 원천은 프롬프트 전체이므로 자동으로 변형본이 된다.
- **(B) Reference만 변형**: 프롬프트 예시는 원본이고, 초안 원천만 변형본이다. 모델 출력은 원본 조건과 같다.
  - 수치 차이로 갈라지는 경우는 있다.
  - 표현이 달라졌을 때 위치 선택이 버티는지를 직접 시험한다.

## 3. poc_rnsd.py에 필요한 인터페이스 (B용)
- 새 옵션 `--draft-source-html <json {train_id: html}>`. 기본값은 None이고, 이때는 지금과 같다.
- `build_prompts`가 두 메시지 세트를 만든다.
  - `msgs`: 모델 입력용. `--exemplar-html` 또는 원본 HTML을 쓴다.
  - `draft_msgs`: 초안 원천용. `--draft-source-html` 값으로 예시 HTML을 바꾼 것이다. 나머지(시스템, 질의, 순서)는 같다.
  - 둘 다 `apply_chat_template(add_generation_prompt=True)`로 토큰화한다.
- 디코딩 루프는 다음과 같다.
  - 모델 forward와 KV는 `msgs` 토큰(`enc`)만 쓴다.
  - 초안 문맥은 `draft_prompt_ids + gen`이다.
    - PLD: `ctx = concat(draft_prompt_ids, gen)`을 `ngram_draft`에 넘긴다.
    - DOM 규칙과 학습 모델: `RankerDrafter(draft_prompt_ids, dec, w_or_None).draft(gen, GAMMA)`
  - 초안 토큰은 그대로 verify에 넘긴다. 위치 인덱스는 draft 문맥 기준이지만 정확 검증이므로 문제없다.
- 기록 필드에 `draft_source_html` 경로와 `draft_prompt_tokens`를 추가한다.

## 4. 비교 정책
- prompt lookup(PLD, n=3→1)
- DOM 규칙: `RankerDrafter(..., weights=None)`
- 학습 순위 모델: `RankerDrafter(..., load_weights(runs/study/ranker_mlp_fold{fmap[id]}.npz))`
  - 원래 형식으로 학습한 cross-fit 가중치를 재학습 없이 그대로 쓴다. fold 매핑은 `runs/study/ranker_fold_map.json`이다.

## 5. 실행 행렬과 예상 GPU 시간 (E2B, pilot 50)
- 근거는 gate2 실측이다. 요청당 PLD 약 5.9 s, DOM 약 5.5 s, 학습 모델도 비슷할 것으로 본다. 한 조건(정책 1개, 요청 50개, rep 1회)이 약 4.6–5 분이다.
- 작업량: 4 변형 × 3 정책 × 2 조건(A, B) = 24 셀. 같은 세션의 원본 기준 3정책 3셀을 더하면 27 셀이다.

| 반복 | 예상 시간 |
|---|---|
| rep 1 | 27 × 약 5 분 ≈ **2.2 시간** |
| rep 3 (gate2와 같은 반복) | 약 6.5 시간 |

- 우선 B(출력이 원본과 같아 해석이 깨끗함) rep 1(약 1.1 시간)을 권한다. 다음으로 A rep 1(약 1.1 시간)이다.
- fmt, css 변형은 프롬프트 토큰이 늘어 prefill이 조금 길어진다. TTFT도 기록한다.
- 판정은 요청 단위 짝 차이로 한다(학습 − DOM 규칙, 학습 − PLD, 각 변형 대 원본). 부트스트랩 95% CI를 쓰고 end-to-end wall-clock 기준이다.

## 6. 시뮬레이터 사전 점검 (조건 B)
- 결과는 `runs/study/p1_perturb_report.json`의 `simB`에 있다.
- target은 gate2 PLD 출력(원본 프롬프트)이고, 초안 원천만 변형본으로 바꿨다.
- 요약은 아래 7절에 있다.

## 7. 사전 점검 결과

### 변형 생성
- 150개 예시 중 픽셀 동일(전체 페이지)을 통과한 비율:
  - fmt 150/150
  - css 150/150
  - dom 146/150. 4개는 감쌀 요소가 뽑히지 않아 변화가 없었다.
  - cls 150/150
- 토큰 수 변화(평균):
  - fmt +17%
  - css +61% (−18% ~ +143%). 브라우저 cssText가 장황하기 때문이다.
  - dom +10%
  - cls +4%

### 시뮬레이터, 조건 B (forward/요청, n=50)
- target은 gate2 PLD 출력이다. 짝 차이는 부트스트랩 95% CI다.

| 원천 | PLD | DOM 규칙 | 학습 MLP | 오라클 | 규칙 − MLP | PLD − 규칙 | PLD − MLP |
|---|---|---|---|---|---|---|---|
| 원본 | 96.9 | 90.4 | 86.3 | 54.8 | 4.1 [3.4, 4.9] | 6.5 [5.2, 7.8] | 10.6 [9.1, 12.3] |
| fmt | 141.3 | 132.0 | 128.2 | 70.4 | 3.8 [2.7, 4.9] | 9.3 [7.9, 10.8] | 13.1 [11.6, 14.8] |
| css | 202.3 | 191.5 | 189.5 | 100.5 | 2.0 [1.2, 2.7] | 10.8 [9.5, 12.2] | 12.8 [11.3, 14.3] |
| dom | 98.6 | 93.1 | 89.4 | 55.8 | 3.7 [3.0, 4.4] | 5.5 [4.4, 6.8] | 9.2 [7.8, 10.8] |
| cls | 107.4 | 100.5 | 96.5 | 57.3 | 4.0 [3.2, 4.8] | 6.9 [5.5, 8.4] | 10.9 [9.4, 12.5] |

### 해석
- 원천만 변형하면 모든 정책의 forward가 늘어난다.
  - fmt는 1.46배, css는 2.1배다. 초안이 출력 형식과 맞지 않게 되기 때문이다.
  - dom과 cls는 1.02–1.1배로 영향이 작다.
- 그래도 정책 순서는 다섯 조건 모두에서 PLD > 규칙 > MLP로 유지된다.
  - 학습 MLP는 원래 형식으로만 학습했고 재학습하지 않았다. 그래도 규칙보다 낫다(CI > 0).
  - css에서는 우위가 2.0으로 줄었다.
- PLD 대비 MLP 이득(절대 forward 수)은 fmt와 css에서 오히려 커진다(13.1, 12.8).
- 조건 B의 실측은 이 순서가 wall-clock에서도 유지되는지 확인하는 단계다.
- 조건 A는 모델 출력 형식이 바뀌므로 시뮬레이션할 수 없다. GPU로만 확인할 수 있다.
