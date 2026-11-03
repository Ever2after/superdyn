# Temporal Geometry QA: 카메라 운동과 장면 동역학을 함께 평가하는 벤치마크

이 프로젝트는 DynamicVerse의 영상을 카메라 이동·회전·장면 동역학으로 정량화하고,
추정한 3D geometry와 객체 궤적으로 temporal QA를 생성·검수·평가합니다.
현재 최종 세트는 **896 QA / 461개 영상 / 6개 task**이며, 정의와 생성 과정부터 아래 순서로 설명합니다.

## 1. Motivation

출발점은 기존 4D understanding benchmark의 영상에서 **충분한 camera motion과 scene dynamics가
함께 나타나는 사례가 부족할 수 있다**는 문제의식입니다. 카메라만 움직이거나 객체만 움직이는
영상에서는 영상상의 2D 변화만으로도 일부 질문을 해결할 수 있습니다. 이 프로젝트는 카메라와 객체의
운동을 분리하고, 시간에 따른 3D 방향·거리·이동량을 함께 추론하는 능력을 평가하려고 합니다.

이를 위해 영상의 겉보기 움직임 크기 하나로 고르지 않고 세 가지 metric을 각각 계산합니다.
QA도 단순히 움직임이 검출됐다는 이유로 만들지 않고, 답의 차이가 충분하고 추적과 geometry가
안정적인 구간에서만 생성합니다. 이후 질문과 표시된 타깃을 함께 보는 VLM 검수로
극단적인 부분 가시성이나 타깃 해석의 모호성을 걸러냅니다.

최종 선정은 **QA의 시각적 해석 가능성과 motion metric**으로 결정합니다.
Baseline 정오답이나 모델 성능은 선정 조건에 사용하지 않습니다.
다만 아래에서 설명하듯 현재 최종 조건은 영상 단위의 camera-high AND dynamics-high이며,
같은 시간 구간에서의 동시성을 추가로 요구하는 `joint_high`와는 구분됩니다.

## 2. 데이터 범위와 전체 파이프라인

데이터 소스는 **DAVIS, MOSE, MVS-Synth, SAV, VOST, uvo, youtube_vis**입니다.
Spring, Point Odyssey, Dynamic Replica, DynPose-100k는 현재 범위에서 제외합니다.
`DYNAMICVERSE_ROOT` 또는 `--data-root`로 원본 위치를 지정합니다.

```text
원본 영상 inventory / 필요한 ZIP 해제
  → released RGB 프레임 샘플링
  → VGGT-Omega pose/depth + GMOS-S foreground + RAFT-small audit
  → camera translation / rotation / scene dynamics 정량화 및 8개 그룹 분류
  → full GMOS multi-object masks + 33-frame RAFT-large 추적
  → 수치 기준을 통과한 6-task QA + 객체 설명용 Qwen annotation
  → QA별 F0–F8 프레임에 요청된 A/B 타깃 표시
  → 질문·선택지·표시 프레임 9장을 함께 보는 Qwen QA 검수
  → keep AND camera-high AND dynamics-high인 QA 확정
  → 동일 QA ID에서 baseline 채점·표·그래프 생성
```

현재 실행에서는 원본 2,387 QA의 baseline 예측이 이미 있으므로, 최종 평가에는 그 응답을 재사용해
선택된 896 QA만 다시 파싱·채점했습니다. 최종 세트 확정 과정에서 새 baseline 추론은 실행하지 않았습니다.

### 2.1 프레임과 시간의 정의

- 각 released sequence 전체에서 균일한 위치의 **33개 context frame**을 샘플링합니다.
- 이 중 context index `0, 4, 8, ..., 32`의 **9개 measurement frame**을 `F0–F8`로 사용합니다.
- 최소 9개의 서로 다른 measurement frame이 필요합니다. 원본이 33장보다 짧으면 context 위치는 중복될 수 있습니다.
- 시간은 `released_frame_index / (원본 프레임 수 - 1)`로 정규화한 `[0,1]`입니다.
  **물리적 초, 실제 FPS, 고정 4초 구간을 뜻하지 않습니다.**
- Geometry는 VGGT-Omega의 416 해상도 전처리 규약을 사용합니다. 객체 마스크와 RAFT 추적도 같은
  geometry grid에 맞춥니다. 극단적인 종횡비는 `[0.5,2]` 범위에 맞춘 중앙 crop이 적용될 수 있습니다.

샘플링 구현은 [frames.py](qa_pipeline/frames.py), geometry grid 정렬은
[inference.py](qa_pipeline/inference.py)와 [objects.py](qa_pipeline/objects.py)에 있습니다.

### 2.2 각 모델의 역할

| 단계 | 모델 | 입력과 산출물 |
|---|---|---|
| Geometry | VGGT-Omega | Measurement RGB 9장 → world-to-camera pose, intrinsics, depth, confidence |
| Motion audit | GMOS-S + RAFT-small | Foreground mask와 9장 사이 flow(12 updates) → 카메라 운동을 보상한 장면 동역학 |
| 객체 segmentation | **full GMOS** | Context 33장 → 여러 객체의 일관된 instance ID와 mask |
| QA 추적 | RAFT-large | Context 33장의 인접 flow(16 updates) → 같은 객체 표면점의 지속적인 궤적 |
| 객체 설명 | 로컬 Qwen3.8-27B | 표시된 2–3개 keyframe → category, description, identity confidence, single-entity 판정 |
| QA 검수 | 같은 로컬 Qwen3.8-27B | 질문·선택지 + 표시된 9장 전체 → keep / exclude / review |

GMOS-S는 foreground용이고, QA의 multi-object mask는 full GMOS에서 얻습니다.
GMOS 내부에는 SAM2 video propagator가 포함되어 있으나, 별도 SAM2 재분할 단계는 추가하지 않습니다.
BBox는 mask의 min/max 좌표로 계산하며 Qwen이 생성하지 않습니다. Qwen은 pose/depth/3D 궤적이나
기하 정답도 생성하지 않습니다. Annotation용 로컬 checkpoint 경로는
[config.py](qa_pipeline/config.py)의 `QWEN_MODEL`과 각 실행의 `config.json`에 기록합니다.

## 3. Metric 정의와 motion 그룹

**그룹 분류용 metric 임계값과 QA 생성용 정답 차이 임계값은 서로 다릅니다.**
그룹 metric은 영상 전체의 운동 조건을 정하고, QA 임계값은 특정 질문 구간의 답을 충분히 구별할 수 있는지 판정합니다.

### 3.1 공통 좌표계와 정규화

Camera extrinsics는 OpenCV world-to-camera convention인 `E_i = [R_i | t_i]`입니다.
카메라 중심은 다음과 같습니다.

$$
C_i = -R_i^T t_i.
$$

Monocular geometry의 절대 스케일에 의존하지 않도록, 해당 metric 경로에서 얻은 유효 배경 depth의
중앙값 `s_audit`로 translation을 정규화합니다. 배경은 GMOS-S foreground의 반대 영역입니다.
Scene dynamics의 영상 변위는 이미지 대각선 `d_img = sqrt(H²+W²)`로 나눕니다.
따라서 이 수치는 **미터나 m/s가 아닙니다.**

### 3.2 세 가지 영상 metric

| Metric | 계산 | High 조건 |
|---|---|---|
| Camera translation | 연속 카메라 중심 사이 거리의 누적합 / 배경 depth | `T >= 0.20` background depths |
| Camera rotation | 연속 `R` 사이 SO(3) 상대 회전각의 누적합 | `R_path >= 40°` |
| Scene dynamics | Foreground에서 observed flow와 camera-only rigid flow의 차이를 집계한 누적 residual | `D >= 0.12` image diagonals **AND** 시간 가중 foreground 면적 비율 `>= 0.01` |

수식으로는 다음과 같습니다. `θ_i`는 연속 pose의 상대 회전각이며 단위는 degree입니다.

$$
T = \sum_i \frac{\lVert C_{i+1}-C_i\rVert_2}{s_{audit}},
\qquad
R_{path} = \sum_i \theta_i.
$$

Scene dynamics는 VGGT depth·pose·intrinsics로 **정적인 장면이었다면 생길 rigid flow**를 만든 뒤,
RAFT-small observed flow에서 이를 뺍니다. Interval `i`의 foreground 유효 픽셀에서 residual 크기의
75th percentile을 구합니다.

$$
r_i = P_{75,\,p\in M_i\cap V_i}
\left(\frac{\lVert f^{obs}_i(p)-f^{rigid}_i(p)\rVert_2}{d_{img}}\right),
\qquad D=\sum_i r_i.
$$

`M_i`는 foreground mask, `V_i`는 계산 가능한 대응·projection 영역입니다.
구현은 먼저 `r_i / Δt_i`를 계산하고 `Δt_i`로 적분하므로 위 누적합과 같습니다.
카메라가 움직여 생기는 flow를 그대로 scene dynamics로 세지 않도록 한 정의입니다.

**High/low의 기준은 누적 path/residual입니다.** 함께 저장되는 median 값은 보고용 통계이며
그 자체가 최종 분류 기준은 아닙니다. 카메라가 갔다가 돌아와도 누적 path는 커질 수 있습니다.
Group rotation은 roll을 포함한 SO(3) 운동이고, 뒤의 T2 QA는 roll을 제외한 optical-axis turn입니다.

계산 구현과 임계값은 [metrics.py](metrics.py)의 `summarize_amplitudes()`와
`AMPLITUDE_THRESHOLDS`에 있습니다. 시간축 rate용 `THRESHOLDS`와 혼동하지 않아야 합니다.

### 3.3 8개 그룹과 4개 cohort

세 metric을 각각 high/low로 나누어 `2³ = 8`개 그룹을 만듭니다.
문자 순서는 **translation / rotation / scene dynamics**이며 `H=high`, `L=low`입니다.
사용자 표시의 `N`은 원본 데이터의 `L`과 같은 뜻입니다. `L`은 운동이 전혀 없다는 뜻은 아닙니다.

| 4개 cohort | Camera motion | Scene dynamics | 포함되는 3-bit 그룹 |
|---|---|---|---|
| LL | Low: translation과 rotation 모두 low | Low | LLL |
| LH | Low | High | LLH |
| HL | High: translation 또는 rotation high | Low | HLL + LHL + HHL |
| HH | High | High | **HLH + LHH + HHH** |

최종 세트는 HH cohort입니다. 즉 **HNH + NHH + HHH**에 해당합니다.

### 3.4 Raw group 분류, geometry quality, 동시성의 구분

현재 실행 설정은 `video_selection_quality_filter=false`입니다.
Group은 `unfiltered_amplitude_summary`에서 가져오며, 이 경로는 추가 confidence/forward-backward
pixel 품질 필터를 끄고 interval의 `quality_valid`를 모두 True로 놓아 누적합니다.
유한한 양의 depth와 계산 가능한 projection/대응 같은 기본 조건은 남아 있습니다.
따라서 아래 QA 구간 품질 검사를 영상 전체의 사전 제외 기준으로 적용한 실행은 아닙니다.

별도로 quality-valid interval의 시간 coverage가 75% 이상인지 나타내는 `eligible`도 기록합니다.
`--quality-filter`를 명시하면 이 영상 단위 조건을 추가할 수 있으나, **현재 최종 세트에는 적용하지 않았습니다.**
QA 생성에서는 이와 별개로 quality-valid 구간과 추적 조건을 항상 사용합니다.

`joint_high_duty`는 interval별 camera-high와 dynamics-high가 **같은 시간에** 나타나는 시간 비율입니다.
`joint_high`는 영상의 camera/dynamics high 조건과 함께 이 비율이 `>= 0.25`일 때 True가 됩니다.
그러나 **현재 최종 선택은 HH 그룹과 QA keep만 요구하며 `joint_high`를 필수 조건으로 추가하지 않습니다.**
현재 세트가 보장하는 조건은 영상 수준의 두 운동 크기이며, 모든 질문 구간에서 동시 운동이
충분하다고 보장하는 것은 아닙니다. 관련 경로는 [inventory.py](qa_pipeline/inventory.py)와
[finalize.py](qa_pipeline/finalize.py)입니다.

## 4. Temporal geometry QA의 정의와 생성 기준

### 4.1 객체 위치와 궤적

QA는 매 프레임 mask 중심을 독립적으로 찍지 않습니다. 객체 mask의 내부에서 시작한 표면점을
RAFT-large로 추적하고, **선택한 구간 전체에서 살아남은 동일한 점 집합**을 depth와 pose로 world 3D에
복원합니다. 그 점들의 산술평균을 객체 anchor `X_o(i)`로 사용합니다.

거리와 이동량은 QA 캐시의 `depth_scale = s_QA`로 정규화한 background-depth 단위입니다.
Group의 raw metric과 QA geometry는 유효 픽셀 선정 경로가 달라 배경 scale이 다를 수 있습니다.
Anchor는 보이는 표면의 대표 위치이며 **물체의 실제 무게중심이나 전체 물체 중심을 뜻하지 않습니다.**
좌표 방향을 묻는 QA는 질문 시작 프레임의 카메라 축을 고정하고,
`right/down/forward = +x/+y/+z`를 사용합니다.

### 4.2 6개 task와 효과 크기 임계값

| Task | 물어보는 값 | 선택지 | 주요 통과 조건 |
|---|---|---|---|
| T1 Camera translation direction | 시작 카메라 축에서 카메라 중심의 3D 변위 방향 | 좌/우/상/하/전/후, 6개 | 최대 성분 `>= 0.10`, 2위 대비 `>= 2배`, 성분 차이 `>= 0.05` |
| T2 Camera view direction change | 시작 카메라 축에서 optical axis가 향한 yaw/pitch 방향 | 좌/우/상/하, 4개 | 우세 각도 `>= 15°`, 다른 축 대비 `>= 2배`, 각도 차이 `>= 8°`; roll 제외 |
| T3 Object world motion direction | 카메라 운동 보상 후 객체 anchor의 3D 변위 방향 | 6개 | T1과 같은 방향 기준 + bootstrap 방향 일치율 `>= 95%` + signed CI margin |
| T4 Camera–object distance change | `norm(X_o(i)-C_i)`의 시작/끝 변화 | closer / farther, 2개 | 절대 변화 `>= 0.10` **AND** 초기 거리 대비 변화 `>= 20%` + signed CI margin |
| T5 Object–object distance change | `norm(X_A(i)-X_B(i))`의 시작/끝 변화 | increased / decreased, 2개 | 초기 separation `>= 0.08`, 절대 변화 `>= 0.10`, 상대 변화 `>= 20%` + signed CI margin |
| T6 Object travel distance comparison | 각 객체의 연속 anchor 변위 누적합 비교 | Object A / Object B, 2개 | 이동량 차이 `>= 0.10`, 큰 값/작은 값 `>= 1.5`, 큰 이동량 `>= 0.20` + signed CI margin |

각도 이외 수치는 background-depth 단위입니다. T1/T3의 최소값은 변위 norm이 아닌 **우세 축 성분의 절댓값**입니다.
T6의 이동량은 `L_o = sum_i norm(X_o(i+1)-X_o(i))`이며, 시작점과 끝점만의 직선 변위와 다릅니다.
T4는 카메라와 객체 사이의 상대 거리 변화이고, 객체 자체의 world motion을 묻는 T3와 다릅니다.

객체 궤적의 공간 block을 **128회 bootstrap**하여 95% 구간을 구합니다.
표의 signed CI margin은 변화량 또는 이동량 차이의 95% CI가 정답 방향으로 0에서 최소 `0.05`만큼
떨어져 있어야 한다는 뜻입니다. T3에는 우세 성분 CI 조건과 bootstrap 방향 일치율도 적용합니다.

정답은 이 수치 규칙으로 결정하고 선택지 순서만 고정 seed로 섞습니다.
정의는 [questions.py](qa_pipeline/questions.py), 추적·bootstrap은
[tracking.py](qa_pipeline/tracking.py), 고정 임계값은 [config.py](qa_pipeline/config.py)의 `TH`에 있습니다.

### 4.3 구간·mask·추적의 공통 gate

- 질문 구간은 measurement interval 2개 이상이며 정규화 시간 길이가 `>= 0.20`이어야 합니다.
  그 구간의 모든 interval이 geometry quality-valid여야 합니다.
- Interval quality는 충분한 static-background 대응(최소 `max(200 pixels, 이미지의 5%)`),
  작은 static residual(`median residual × 전체 window 길이 <= 0.08`), foreground 대응의 관측 가능성을 검사합니다.
  Depth confidence가 20th percentile 미만인 픽셀과 불안정한 forward/backward 대응도 이 경로에서 걸러냅니다.
  Foreground 면적이 이미지의 1% 이상이면 유효 foreground 대응 최소 20 pixels와 coverage 20% 이상을 요구합니다.
- 요청 객체의 구간 내 mask score는 GMOS 내부 stage-1 SAM2 predicted mask IoU `>= 0.80`이어야 합니다.
  이는 정답 mask와 측정한 실제 IoU가 아닙니다.
- 각 measurement frame에서 eroded 유효 mask가 최소 150 pixels이어야 합니다. 겹치는 instance 픽셀은 제외합니다.
- 초기 seed는 최대 256개, 최종 생존 track은 최소 32개 및 초기 대비 15% 이상이어야 합니다.
  시작 공간의 `4×4` block 중 최소 4개에 생존 점이 분포해야 합니다.
- Flow forward/backward 오차는 `<= min(2 px, 1 + 0.05 × flow magnitude)`를 요구하며,
  각 flow step의 생존 점은 직전 대비 25% 이상이어야 합니다.
- 객체 쌍의 mask overlap은 작은 mask 면적 대비 최대 5%입니다.

후보 구간은 긴 구간을 우선 검사합니다. 선택 record는 통과한 T5/T6 수, 구간 길이, track 수 순으로
우선순위를 정하고, 객체 쌍이 없으면 single-object 또는 camera-only record로 내려갑니다.
따라서 모든 영상에 모든 task를 강제로 생성하지 않습니다.

### 4.4 객체 설명 annotation

수치 QA가 살아남은 객체만 표시된 시작/중간/끝 keyframe 2–3장으로 Qwen에 보냅니다.
`category`, 짧은 식별 description, `identity_confidence`, `single_entity`를 받으며
**high confidence AND single_entity=true**인 요청 객체를 포함한 QA만 자동 통과시킵니다.
불명확한 객체를 사용하는 QA는 semantic review로 보류합니다. Camera-only QA는 객체 설명을 요구하지 않습니다.
이 단계는 `temperature=0`, `max_tokens=600`, thinking off, 고정 seed를 사용하며 geometry 정답은 입력·생성하지 않습니다.

## 5. Filtering 로직과 단계별 유지량

### 5.1 전체 흐름

| 단계 | 영상 수 | QA 수 | 적용 내용 |
|---|---|---|---|
| 범위 내 inventory | 4,493 | — | 제외 도메인 적용 후 released scene 목록 |
| RGB 사용 가능 | 4,457 | — | RGB 없는 1개, 9개 미만 measurement frame인 35개 제외 |
| Geometry/GMOS/RAFT 및 수치 QA 생성 | 4,457 처리 | 2,399 | 구간·추적·효과 크기를 만족한 후보 QA; geometry 실패 0 |
| 객체 설명 semantic gate 통과 | 1,275 | 2,387 | 12 QA는 semantic 보류; 3,182개 영상은 통과 QA 없음 |
| QA-conditioned visual audit keep | 1,247 | 2,262 | 전체 2,387 QA 검수: keep 2,262 / exclude 112 / review 13 |
| Keep 중 HH 그룹 확정 | **461** | **896** | `HLH + LHH + HHH`, 즉 `HNH + NHH + HHH` |

4,457개 중 QA가 없는 영상을 모두 영상 품질 필터 탈락으로 해석하면 안 됩니다.
현재 영상 단위 `--quality-filter`는 꺼져 있으며, 수치·추적·객체 설명 기준을 만족하는 QA가 없을 수도 있습니다.
단계별 실행 근거는 [run_status.json](outputs/temporal_geometry_fullscope/run_status.json)과
[QA config](outputs/temporal_geometry_fullscope/qa/config.json)에 있습니다.

### 5.2 QA-conditioned visual audit

검수 단위는 영상 하나가 아닌 **QA 하나**입니다. 같은 영상에서 다른 객체나 구간을 묻는 질문은
각각 검수합니다. 한 요청에 다음 자료를 함께 제공합니다.

1. 원본 질문과 선택지, 질문에 명시된 frame interval과 좌표 convention.
2. 요청된 객체 A/B 표시가 포함된 F0–F8 RGB 프레임 **9장 전체**.
3. A는 orange contour, B는 green contour라는 시각적 대응 규칙.

프레임을 한 장씩 따로 판정하거나 프레임별 다수결을 사용하지 않습니다.
정답·answer index·geometry 수치·기존 baseline 예측·이전 영상 필터 판정은 모델 입력에서 제외합니다.

| Decision | 판정 기준 | 최종 취급 |
|---|---|---|
| keep | 객체 정체성과 질문의 시간·공간 관계를 시각적으로 해석할 수 있음 | HH 조건을 추가로 확인 |
| exclude | 지나치게 잘린 일부만 보여 의도한 객체 운동을 판별하기 어려움, 타깃 라벨 누락/오류/ID 변경, 질문과 대상의 대응이 불명확함 | 제외 |
| review | 가시성 또는 질문 grounding이 경계 사례임 | 최종 세트에서 보류·제외 |

부분만 보여도 정체성과 필요한 움직임이 분명하면 keep합니다. 가까운 촬영, egocentric/selfie라는 형식,
짧은 occlusion, 질문 구간 밖의 프레임만으로 자동 제외하지 않습니다. 손이나 신체 일부를 사람 전체로
해석하는 오류는 잡되, **3D 추론이나 camera compensation이 어렵다는 이유 자체로 제외하지 않도록** 지시합니다.
이 검수는 QA를 풀거나 기하 정답을 검증하는 작업이 아닙니다.

모델 출력은 `{"decision":"keep"}` / `exclude` / `review` 중 하나뿐입니다.
추가 confidence, rationale, 프레임별 속성을 추론하거나 후처리 조건으로 요구하지 않습니다.
실행은 로컬 Qwen3.8-27B, GPUs 4–7, `temperature=0`, `max_tokens=32`, thinking off,
seed `20261027`을 사용했고 오류는 0개였습니다.

실제 [prompt.txt](outputs/temporal_geometry_fullscope/qa/qa_visual_filter_fullscope/prompt.txt),
[config.json](outputs/temporal_geometry_fullscope/qa/qa_visual_filter_fullscope/config.json),
[results.json](outputs/temporal_geometry_fullscope/qa/qa_visual_filter_fullscope/results.json)에 판정과 설정이 남아 있습니다.
`exclude/`, `review/`에는 해당 QA와 표시 프레임을 보관합니다. HTML은 생성하지 않습니다.

이전 **video-only framing filter**의 exclude/review는 진단 이력으로 보관합니다.
현재 최종 선택에서 그 영상 판정을 별도 AND 조건으로 다시 적용하지 않습니다.

### 5.3 최종 선택 규칙

```python
qa_keep = audit[qa_id]["decision"] == "keep"
camera_high = group[0] == "H" or group[1] == "H"
dynamics_high = group[2] == "H"
final = qa_keep and camera_high and dynamics_high
```

원본 HH cohort의 957 QA 중 QA 검수에서 56개 exclude, 5개 review가 되어 **896개**가 남았습니다.
전 그룹 keep 2,262개 중에서는 39.6%입니다. 그룹별 수나 task 비율을 맞추기 위한 추가 subsampling은 하지 않았습니다.
최종 QA ID와 원본 해시는 manifest로 고정하며 원본 질문·선택지·기하 정답·임계값은 수정하지 않습니다.

## 6. 최종 데이터 구성

### 6.1 Motion 그룹

| 표시 그룹 | 원본 그룹 | QA | 고유 영상 |
|---|---|---|---|
| HNH | HLH | 314 | 164 |
| NHH | LHH | 190 | 94 |
| HHH | HHH | 392 | 203 |
| 전체 | HH cohort | **896** | **461** |

### 6.2 Task 구성과 검수 전후

| Task | Semantic 통과 원본 QA | Visual keep, 전 그룹 | 최종 HH QA | 최종 고유 영상 | Random |
|---|---|---|---|---|---|
| T1 Camera translation | 448 | 448 | 271 | 271 | 1/6 |
| T2 Camera turn | 462 | 462 | 152 | 152 | 1/4 |
| T3 Object world direction | 594 | 549 | 203 | 170 | 1/6 |
| T4 Camera–object distance | 405 | 396 | 145 | 118 | 1/2 |
| T5 Object–object distance | 234 | 220 | 60 | 60 | 1/2 |
| T6 Object travel comparison | 244 | 187 | 65 | 65 | 1/2 |
| 전체 | **2,387** | **2,262** | **896** | **461** | **28.125%** |

한 영상에서 여러 task가 나올 수 있으므로 task별 영상 수를 더하면 전체 영상 수와 같지 않습니다.
T1–2는 총 423 QA, T3–6은 총 473 QA입니다.

### 6.3 도메인 구성

| 소스 | 전처리한 사용 가능 영상 | 최종 영상 | 최종 QA |
|---|---|---|---|
| DAVIS | 88 | 15 | 37 |
| MOSE | 1,327 | 90 | 188 |
| MVS-Synth | 120 | 0 | 0 |
| SAV | 288 | 38 | 98 |
| VOST | 641 | 129 | 203 |
| uvo | 1,244 | 122 | 237 |
| youtube_vis | 749 | 67 | 133 |
| 전체 | **4,457** | **461** | **896** |

집계 원본은 [qa_counts.csv](outputs/temporal_geometry_fullscope/qa/final/qa_counts.csv)입니다.

## 7. Baseline 구성과 평가 세팅

### 7.1 모델과 checkpoint

일반 VLM인 Qwen2.5-VL 3B/7B와 geometry·4D 특화 baseline을 함께 평가합니다.
모델명만 지정하지 않고 checkpoint revision과 native backend를 고정했습니다.

| 모델 | Checkpoint repository / variant | 입력 처리 | Transformers |
|---|---|---|---|
| Qwen2.5-VL 3B | `Qwen/Qwen2.5-VL-3B-Instruct` | Native HF multi-image | 4.50.0 |
| Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` | Native HF multi-image | 4.50.0 |
| VG-LLM QA 4B | `zd11024/vgllm-qa-vggt-4b` | 공식 RGB/geometry 전처리 + 자체 geometry encoder | 4.50.0 |
| VG-LLM QA 8B | `zd11024/vgllm-qa-vggt-8b` | 같은 VG-LLM backend | 4.50.0 |
| MLLM-4D SFT 8B | `flow666/MLLM-4D`, `MLLM-4D-SFT-1.0` | Native HF multi-image, Qwen3 backend | 5.19.0 |
| 4D-RGPT 8B | `nvidia/4D-RGPT-8B` | Native `generate_content` image sequence | 4.46.3 |
| DSR Suite | `TencentARC/DSR_Suite-Model` | Native video packaging + 32 GSM query tokens | 4.51.1 |

DSR은 Qwen2.5-VL **7B backbone**이며 spatial module을 포함한 전체 크기는 저장된 metadata상 약 9B입니다.
따라서 여기서 DSR의 7B 표기는 전체 parameter 수와 구분합니다.
Full commit hash, 로컬 checkpoint 경로와 Python 환경은
[models.json](outputs/temporal_vlm_eval_fullscope/final/models.json), 실제 로딩 설정은
[loading/](outputs/temporal_vlm_eval_fullscope/loading/)에 기록되어 있습니다.

### 7.2 공통 입력과 모델별 차이

- 모든 모델은 **같은 QA의 같은 F0–F8 표시 RGB 9장, 원본 질문, 같은 순서의 선택지**를 받습니다.
  요청된 객체 A/B에만 contour를 표시하며 camera-only QA는 객체 표시가 필요하지 않습니다.
- Reference answer, depth, camera pose, 수치 evidence와 threshold는 모델 adapter에 전달하지 않습니다.
  `inputs.json`과 `reference_answers.json`을 분리합니다.
- 답변은 선택지 위치에 해당하는 **옵션 문자 하나**를 요구합니다. 객체 A/B와 옵션 A/B를 혼동하지 않도록
  객체 선택지는 프롬프트에 `Object A` / `Object B`로 표기합니다.
- RGB 소스는 같지만 resize/crop, visual token 수 등은 각 native processor를 따릅니다.
  Qwen/MLLM/DSR processor의 `max_pixels=200704`, VG-LLM은 공식 geometry+RGB crop을 사용합니다.
- VG-LLM의 geometry encoder 입력은 **동일 RGB에서 계산하는 자체 feature**입니다.
  Annotation용 VGGT pose/depth를 모델에 넣지 않습니다.
- DSR은 동일 9장을 video로 포장하고 native temporal patch padding으로 마지막 프레임을 반복하며
  32개의 GSM query token을 추가합니다. `fps=2`는 temporal 위치 인코딩용이고 실제 영상 FPS가 아닙니다.
- 4D-RGPT는 inference에서 학습 전용 geometry teacher를 끕니다.
  MLLM의 patch projection은 동일 BF16 연산 결과를 내는 최적화를 적용하고 로딩 시 bit-exact probe로 확인합니다.

구현은 [backends.py](qa_pipeline/evaluation/backends.py), 공통 입력·파서는
[protocol.py](qa_pipeline/evaluation/protocol.py)에 있습니다. 이 실험은 **공통 9-frame 평가 프로토콜**이며,
각 논문의 원래 benchmark/frame 수/전처리 예산을 그대로 재현했다고 해석하지 않습니다.

### 7.3 추론·GPU 설정

공통 설정은 `do_sample=false`, `num_beams=1`, `max_new_tokens=64`, BF16, SDPA이며
Torch `2.9.1+cu130`을 사용했습니다. 추론 seed는 `20261020`입니다.
8개 NVIDIA B200에서 모델별 프로세스를 분리했습니다.

| GPU | 배치 |
|---|---|
| 0 | Qwen2.5-VL 3B |
| 1 | Qwen2.5-VL 7B |
| 2 | VG-LLM QA 4B |
| 3, 4 | VG-LLM QA 8B, QA 목록을 2개 shard로 분할 |
| 5 | MLLM-4D SFT 8B |
| 6 | 4D-RGPT 8B |
| 7 | DSR Suite |

정확한 배치는 [jobs_config.json](outputs/temporal_vlm_eval_fullscope/final/jobs_config.json),
입력·생성 규약은 [protocol.json](outputs/temporal_vlm_eval_fullscope/final/protocol.json)에 있습니다.

### 7.4 채점과 random baseline

원본 raw response를 결정적인 MCQ parser로 다시 파싱하여 reference index와 비교합니다.
옵션 문자나 명확한 동일 선택지 텍스트만 인정하고, 모호한 복수 답이나 파싱 불가 응답은 오답 처리합니다.
모든 모델의 분모는 같은 896 QA이며, 모델별로 어려운 QA나 invalid 응답을 분모에서 빼지 않습니다.

$$
Accuracy=\frac{\#correct}{896},\qquad
Random=\frac{1}{896}\sum_{q=1}^{896}\frac{1}{|options_q|}=28.125\%.
$$

T1/T3는 6지선다, T2는 4지선다, T4–6은 2지선다이므로 random은 task 구성에 따라 달라집니다.
`leaderboard.csv`의 전체 정확도는 **QA 수로 가중된 micro accuracy**이고,
`task_macro_percent`는 6개 task 정확도의 단순 평균으로 별도 저장합니다.

## 8. 최종 평가 결과

아래 결과는 **최종 896 QA** 기준입니다. 총 `896 × 7 = 6,272`개 예측을 기존 응답에서 재채점했고
runtime error는 0개였습니다. T1–2와 T3–6의 random 기대값은 각각 19.66%, 35.69%입니다.

| 모델 | Correct | 전체 Acc. | Random gap (pp) | T1–2 Acc. | T3–6 Acc. | Invalid |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-VL 3B | 357/896 | 39.84% | +11.72 | 37.83% | 41.65% | 6 |
| Qwen2.5-VL 7B | 369/896 | 41.18% | +13.06 | 37.83% | 44.19% | 0 |
| VG-LLM QA 4B | 355/896 | 39.62% | +11.50 | 37.12% | 41.86% | 0 |
| VG-LLM QA 8B | 414/896 | 46.21% | +18.08 | 43.50% | 48.63% | 0 |
| MLLM-4D SFT 8B | 443/896 | 49.44% | +21.32 | 43.50% | 54.76% | 0 |
| 4D-RGPT 8B | 290/896 | 32.37% | +4.24 | 22.93% | 40.80% | 1 |
| DSR Suite (7B backbone) | 383/896 | 42.75% | +14.62 | 41.13% | 44.19% | 0 |
| Uniform random | — | 28.125% | 0.00 | 19.66% | 35.69% | — |

MLLM-4D SFT 8B가 전체 49.44%로 가장 높고 random 대비 차이는 +21.32 percentage points입니다.
이 수치는 자동 geometry reference와의 일치율입니다.

### 8.1 세 motion 그룹별 성능

다음 표는 각 그룹의 실제 QA 구성 그대로 집계한 micro accuracy입니다.

| 모델 | HNH (314 QA) | NHH (190 QA) | HHH (392 QA) | 전체 Avg. |
| --- | --- | --- | --- | --- |
| Qwen2.5-VL 3B | 33.76% | 48.42% | 40.56% | 39.84% |
| Qwen2.5-VL 7B | 39.49% | 44.74% | 40.82% | 41.18% |
| VG-LLM QA 4B | 36.62% | 45.79% | 39.03% | 39.62% |
| VG-LLM QA 8B | 42.04% | 56.84% | 44.39% | 46.21% |
| MLLM-4D SFT 8B | 46.82% | 58.95% | 46.94% | 49.44% |
| 4D-RGPT 8B | 30.25% | 38.42% | 31.12% | 32.37% |
| DSR Suite (7B backbone) | 42.36% | 46.32% | 41.33% | 42.75% |
| Uniform random | 28.21% | 30.66% | 26.83% | 28.125% |

Task별 세부 결과는 [accuracy_by_task.csv](outputs/temporal_vlm_eval_fullscope/final/accuracy_by_task.csv),
T1–2 / T3–6 통계는 [per_task_family.csv](outputs/temporal_vlm_eval_fullscope/final/per_task_family.csv)에 있습니다.
전체·task·그룹 결과를 함께 그린 [그래프 PNG](outputs/temporal_vlm_eval_fullscope/final/results.png)와
[PDF](outputs/temporal_vlm_eval_fullscope/final/results.pdf)를 제공합니다.

### 8.2 그룹 성능을 비교할 때

그룹마다 task와 선택지 수의 비율이 달라 원시 accuracy 차이가 순수한 motion 난도 차이를 뜻하지는 않습니다.
Task 구성을 고정한 4-cohort 분석은
[기존 keep-only 보고서](outputs/temporal_vlm_eval_fullscope/subsets/qa_visual_keep/README.txt)와
[task-standardized 결과](outputs/temporal_vlm_eval_fullscope/subsets/qa_visual_keep/task_standardized_groups.csv)에
별도로 보존합니다. 이 분석은 전 그룹 keep 2,262 QA를 사용하고, 공통 task 가중치는 원본 2,387 QA의
구성비이며 95% CI는 video-cluster robust 방식입니다. 현재 896 QA만의 결과와 구분해야 합니다.

## 9. 해석 범위

현재 데이터의 질문과 정답은 자동 추정 geometry와 추적 결과에서 생성했습니다.
Qwen 검수는 타깃과 질문을 해석할 수 있는지를 검사하며, 기하 정답의 수동 검증을 대체하지 않습니다.
QA의 기존 `development` split과 자동 annotation 상태도 그대로 유지합니다.
Bootstrap은 추적 표면점 표본에 대한 안정성 검사이며 pose/depth 모델 오차 전체의 불확실성을 보장하지 않습니다.

최종 세트는 task나 도메인별 균형 표본이 아니며, 카메라·객체 운동이 모든 질문 구간에 동시에
존재한다는 조건도 별도로 강제하지 않습니다. 최종 accuracy만으로 다른 motion 그룹보다 더 어렵다는
일반적·인과적 결론을 내리려면 task 구성과 video 내 상관관계까지 고려한 별도의 비교가 필요합니다.

## 10. 최종 산출물과 원본 보관


프로젝트 기준 경로는 [benchmark.json](benchmark.json)에 고정했습니다.

| 자료 | 경로 |
|---|---|
| 최종 QA | [qa.json](outputs/temporal_geometry_fullscope/qa/final/qa.json), [qa.jsonl](outputs/temporal_geometry_fullscope/qa/final/qa.jsonl) |
| 고정 QA/video ID와 선택 조건 | [selection.json](outputs/temporal_geometry_fullscope/qa/final/selection.json), [manifest.json](outputs/temporal_geometry_fullscope/qa/final/manifest.json) |
| Task/그룹/도메인별 QA·영상 수 | [qa_counts.csv](outputs/temporal_geometry_fullscope/qa/final/qa_counts.csv) |
| 최종 평가 | [leaderboard.csv](outputs/temporal_vlm_eval_fullscope/final/leaderboard.csv), [summary.json](outputs/temporal_vlm_eval_fullscope/final/summary.json) |
| 평가 그래프 | [results.png](outputs/temporal_vlm_eval_fullscope/final/results.png), [PDF](outputs/temporal_vlm_eval_fullscope/final/results.pdf) |
| Task·그룹별 표, 전체 평균 포함 | [accuracy_by_task.csv](outputs/temporal_vlm_eval_fullscope/final/accuracy_by_task.csv), [accuracy_by_group.csv](outputs/temporal_vlm_eval_fullscope/final/accuracy_by_group.csv) |
| T1–2 / T3–6 분리 성능 | [per_task_family.csv](outputs/temporal_vlm_eval_fullscope/final/per_task_family.csv) |
| 검증 | [validation.json](outputs/temporal_vlm_eval_fullscope/final/validation.json) |

원본 2,387 QA, 전체 7개 모델 예측, geometry/GMOS/RAFT 근거와 표시 프레임은 두 `fullscope`
폴더에 보존합니다. 최종 `cases/`, `media/`, `predictions/`는 선택된 원본 자료를 상대 심볼릭 링크로
공유하므로 부모 폴더도 함께 보관해야 합니다. `qa/final/qa.json`이 최종 QA이며,
상위 `qa/qa.json`은 필터링 전 원본입니다.

## 11. 실행과 재현


아래 명령은 `motion_audit/`에서 실행합니다. `qa.py`가 단일 진입점입니다.

```bash
.venv/bin/python qa.py --help
.venv/bin/python qa.py status       # 기본: 최종 평가 상태
.venv/bin/python qa.py summarize    # 기본: 최종 QA 수 통계

# 원본 검수 결과와 기존 예측으로 최종 세트·표·PNG/PDF 재생성
.venv/bin/python qa.py finalize

# 고정된 최종 QA ID로 기존 예측 재채점 및 그래프 재생성
.venv/bin/python qa.py report

# 로그/PID/lock/temp/bytecode 정리: 기본은 목록 확인
.venv/bin/python qa.py cleanup
.venv/bin/python qa.py cleanup --apply

# CPU 계약·회귀 테스트
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests -v
```

`finalize`는 기본적으로 896개를 요구하고, 전체 검수 coverage와 source/input/reference의 일치,
프롬프트·픽셀 해시, 원본 예측 재채점, 프레임 경로를 검증합니다. 같은 최종 경로에서는
선택된 QA ID나 원본 해시가 바뀌면 중단합니다. 다른 실험은 별도 `--out`, `--eval-out`을 사용합니다.

새 모델 추론이 필요할 때는 평가 폴더를 별도로 만듭니다. `--models`는 고정 revision과 로컬 경로를,
`--jobs-config`는 모델/GPU/shard 배치를 지정합니다. 기존 평가 재사용은 프롬프트, 픽셀, 선택지,
정답, revision과 backend가 일치할 때만 허용합니다.

```bash
.venv/bin/python qa.py evaluate \
  --source outputs/temporal_geometry_fullscope/qa/final \
  --models outputs/temporal_vlm_eval_fullscope/models.json \
  --jobs-config outputs/temporal_vlm_eval_fullscope/jobs_config.json \
  --out outputs/temporal_vlm_eval_fullscope/runs/final896 \
  --reuse-evaluation outputs/temporal_vlm_eval_fullscope --prepare-only

.venv/bin/python qa.py evaluate \
  --out outputs/temporal_vlm_eval_fullscope/runs/final896 --background
```

QA visual filter 전체 검수를 재실행할 경우 새 출력 경로를 사용합니다.

```bash
.venv/bin/python qa.py filter-qa-visual \
  --eval-out outputs/temporal_vlm_eval_fullscope \
  --out outputs/temporal_geometry_fullscope/qa/audits/new_visual_audit \
  --all --gpus 4,5,6,7 --background
```

9개 표시 프레임을 QA와 함께 **한 요청**으로 입력하고 `keep/exclude/review` decision만 받습니다.
정답, geometry 수치, 기존 baseline 예측은 입력하지 않습니다. `--all`은 같은 영상의 여러 QA도
각각 처리합니다. 파일럿은 `--all` 대신 `--count 100`을 사용합니다.
`exclude/`, `review/`에 해당 QA와 표시 프레임을 내보내며 HTML은 생성하지 않습니다.

### 11.1 원본부터 다시 생성할 때

```bash
.venv/bin/python qa.py inventory \
  --out outputs/temporal_geometry_fullscope/manifests/scenes.json

.venv/bin/python qa.py run \
  --manifest outputs/temporal_geometry_fullscope/manifests/scenes.json \
  --cache-root outputs/temporal_geometry_fullscope/geometry \
  --out outputs/temporal_geometry_fullscope/runs/new_run \
  --gpus 0,1,2,3,4,5,6,7 --background
```

`run`은 없는 geometry 캐시를 생성한 뒤 QA를 만듭니다. `generate`는 기존 캐시만 사용합니다.
`--reuse-run`으로 완료된 annotation을 재사용할 수 있습니다. ZIP 전처리는 `extract`,
geometry만 생성할 때는 `preprocess`를 사용하며 세부 옵션은 각 명령의 `--help`에 있습니다.
새 QA의 visual audit에는 해당 QA에서 준비한 평가 `inputs.json`이 필요합니다.
현재 `finalize`는 완료된 원본 평가 응답을 사용하므로 새 데이터 재실행 시에도 이 입력/예측 경로를 명시합니다.

재현 검증은 프롬프트·선택지·reference·픽셀·모델 revision 일치를 검사합니다.
기존 리팩토링 검증에서는 CPU 테스트 52개와 최종 6,272개 예측의 점수 일치,
기존 2,262-QA 표준화 결과의 일치를 확인했습니다. 기록은 [refactor_validation.json](refactor_validation.json)입니다.

## 12. 코드와 환경


```text
qa.py                       단일 CLI
benchmark.json              최종 세트 경로와 QA/video 수
qa_pipeline/
  config.py                 경로, task, 그룹, 임계값
  inventory.py, extraction.py
  inference.py, preprocess.py, frames.py
  storage.py, tracking.py, questions.py
  objects.py, flow.py, generation.py, pipeline.py
  framing_filter.py         프레이밍 audit 및 공통 Qwen 서버/I/O
  qa_visual_filter.py       QA-conditioned 공동 9-frame audit
  finalize.py               최종 선택·내보내기·무결성 검증
  maintenance.py            비활성 로그·PID·임시 파일 정리
  evaluation/
    protocol.py             프롬프트와 MCQ 파서/채점 계약
    runner.py               입력 준비, 병렬 추론, 재사용, 집계
    backends.py             모델별 native adapter
    reports.py              재채점, task/그룹 표, PNG/PDF
    filtered_report.py      4개 cohort의 keep-only 비교·task 표준화
metrics.py                  synthetic 프로젝트도 사용하는 공용 수치 코드
release_hf.py               Hugging Face 공개본(2,262 QA·1,247 영상) 생성
tests/                      geometry·추적·입력·채점·최종 선택 계약
outputs/                    최종 산출물과 재현용 원본 (git에는 요약·최종 QA·평가 표만 포함)
```

`.venv`는 QA/CPU 집계/그래프 환경이고 [requirements.txt](requirements.txt)에 고정했습니다.
Qwen annotation 서버는 `.venv_vllm`, baseline별 추론 환경은 `.venv_eval_vg`,
`.venv_qwen`, `.venv_eval_rgpt`, `.venv_eval_dsr`를 사용합니다.
`vendor/`, `checkpoints/`, `.venv*`, 원본 영상과 캐시는 로컬에 보존하며 git에는 포함하지 않습니다.
`vendor/`는 baseline 공식 저장소(4D-RGPT, DSR_Suite, MLLM-4D, VG-LLM)와 GMOS·VGGT-Omega 코드이고,
`checkpoints/`는 GMOS·VGGT-Omega 가중치입니다. 초기 기존-벤치마크 motion audit의 자료
(`videos/`, `cache/`, `metadata/`, `results/`, `dynsuper_*.tar`, `*_manifest.json`)도 로컬에만 있습니다.

## 13. 공개 데이터 (Hugging Face)

[`Ever2after/superdyn-real`](https://huggingface.co/datasets/Ever2after/superdyn-real)에는 visual audit keep
**2,262 QA**와 그 QA가 사용하는 **1,247개 원본 영상**(DynamicVerse released RGB 프레임)을 올렸습니다.
최종 896 QA는 `subsets/final_hh_896.txt`로 표시합니다. QA 레코드의 `cases/...` 경로는 데이터셋 루트 기준으로 그대로 풀립니다.
생성 스크립트는 [release_hf.py](release_hf.py)입니다.

```bash
.venv/bin/python release_hf.py --out /path/to/empty/folder
```

`cleanup`은 실행 중인 작업이 있으면 중단하며 모델 환경·vendor·원본 영상·cache·cases·media·
predictions·필터의 exclude/review 근거를 지우지 않습니다. 과거 worker/pilot 결과는 보존합니다.
정리 내역은 `maintenance.json`, 코드 이동과 원본 해시는 `refactor_manifest.json`에서 확인합니다.
