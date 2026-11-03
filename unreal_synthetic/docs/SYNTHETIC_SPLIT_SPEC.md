# Synthetic split 사양 초안 (v0, 파일럿용)

이 문서는 feasibility 확인용 **작은 초안**이다. 파일럿 결과를 보고 수치와 범위를 다시 정한다.

## 1. 역할

Real split은 실제 영상과 자동 추정 정답(VGGT·RAFT·GMOS)으로 생태적 타당성을 맡는다.
Synthetic split은 real이 할 수 없는 것을 맡는다.

| 목적 | 근거 |
|---|---|
| 정확한 GT | 카메라·객체 pose와 depth·mask를 렌더러에서 직접 얻는다. |
| 정답 분포 설계 | real의 T1 up/down 5%, T3 up/down 6%, T1 forward 38%, T4 closer 61% 편향을 설계 단계에서 균형 맞춘다. |
| 질문 구간의 동시 운동 보장 | real은 영상 단위 HH만 보장하고 질문 구간의 `joint_high`는 보장하지 않는다. |
| 2D 단서와 3D 정답의 분리 진단 | 화면상 운동은 같고 3D 정답만 다른 counterfactual 묶음을 만든다. |
| real 파이프라인 검증 | 같은 영상에 real QA 생성기를 돌려 자동 정답의 정확도를 GT로 측정한다(후속). |

## 2. 구성

| Sub-split | 내용 | 채점 |
|---|---|---|
| **S-Matched** | real과 같은 T1–T6, 같은 9-frame 프로토콜·임계값. 정답 분포는 균형. | QA 단위 accuracy |
| **S-Counterfactual** | 같은 환경·배우·카메라 높이에서, 타깃의 화면 궤적은 비슷하지만 3D 정답이 다른 클립 2개 묶음 | 묶음 단위(둘 다 맞아야 정답) + QA 단위 |

모든 QA에 `cue` 플래그를 붙인다. 타깃의 화면상 이동 방향이 3D 정답과 같은 쪽이면 `consistent`,
반대면 `conflict`다(T3, T6 대상). 정확도를 플래그별로 나눠 보고한다.

## 3. 영상 규격과 프로토콜

| 항목 | 값 |
|---|---|
| 해상도·fps | 1280×720, 24fps |
| 길이 | 129 frame(5.375초) |
| Context 33 frame | 렌더 frame `0, 4, …, 128` |
| Measurement F0–F8 | 렌더 frame `0, 16, …, 128` |
| 모델 입력 | real과 동일: F0–F8 9장, 타깃 A 주황·B 초록 contour, 같은 프롬프트 |
| 수평 FOV | 클립별 설계값(파일럿 65–80°) |

## 4. GT 정의

- **World**: Unreal 좌표(cm, 왼손계)를 `(x, −y, z)/100`으로 바꾼 오른손계 미터 좌표.
- **Camera**: OpenCV world-to-camera `E=[R|t]`(x 오른쪽, y 아래, z 앞). Intrinsics는 FOV와 해상도에서 계산한다.
- **Depth / instance**: GT 전용 렌더 job(spatial sample 1, 32-bit EXR)에서 scene depth와
  "가장 가까운 표면일 때만" 기록한 custom stencil을 함께 얻는다. 가려진 부분은 mask에 들어가지 않는다.
- **객체 anchor**: 두 정의를 모두 계산한다.
  - `center`: 객체 bounding box 중심(사람·동물은 매 frame 평가된 skinned bounds).
  - `visible`: 해당 frame의 mask 픽셀을 depth로 역투영한 world 점의 평균(real의 "보이는 표면" 정의와 대응).
  두 정의가 같은 정답을 낼 때만 QA를 채택한다.
- **정규화 단위**: 9개 measurement frame에서 타깃이 아닌 픽셀의 depth 중앙값 `s`(하늘·200m 초과 제외).
  임계값은 real의 `TH`를 그대로 `s` 단위에 적용하고, 미터 값도 evidence로 함께 저장한다.

## 5. 생성 방식: answer-first

1. 클립마다 **설계 의도**를 먼저 적는다: `(task, window, answer, cue)` 목록.
2. 카메라 primitive(정지·pan·tilt·truck·dolly·crane·orbit·follow·handheld)와 배우 경로를 그 의도에 맞게 정한다.
3. 렌더 후 GT로 T1–T6을 real과 같은 규칙으로 다시 계산한다.
4. 아래 gate를 모두 통과하고 **설계 정답과 GT 정답이 일치**하는 QA만 채택한다.

| Gate | 기준 |
|---|---|
| 효과 크기 | real `TH` 그대로(`s` 단위) |
| 가시성 | 요청 타깃이 구간의 모든 measurement frame에서 erosion 후 mask ≥ 150px |
| 쌍 겹침 | 두 타깃 mask 겹침 ≤ 작은 mask의 5% |
| anchor 일치 | `center`와 `visible` anchor가 같은 정답 |
| 설계 일치 | GT 정답 = 설계 정답 |

## 6. 파일럿 범위와 합격 기준

**범위**: 완성 환경 맵 2–3개, 클립 10개 안팎(S-Matched 6–8, S-Counterfactual 2쌍).
사람(Rocketbox 1·Paragon 3)과 동물(늑대·사슴·돼지·여우·까마귀)을 쓴다.

**Feasibility 합격 기준**

| 항목 | 기준 |
|---|---|
| 파이프라인 | 클립 설계 → 렌더(RGB+GT) → QA·평가 입력까지 수동 개입 없이 실행 |
| GT 정합 | 객체 3D anchor 투영점이 해당 mask 안에 들어가는 비율 ≥ 95% |
| 설계 적중률 | 설계한 QA 중 gate 통과·정답 일치 ≥ 70% |
| 정답 균형 | T1·T3에서 up/down 포함 6방향 중 5개 이상 등장 |
| 비용 | 클립당 렌더 시간 기록(RGB·GT 분리) |
| 외관 | F0–F8 contact sheet로 확인: 검은 무텍스처 벽·빈 지평선 프레임 없음 |

**파일럿에서 다루지 않는 것**: 조명·재질 랜덤화, 실내 환경, 차량 바퀴 동역학, real 파이프라인 교차 검증,
대규모 baseline 평가. 결과를 보고 v1 사양에서 정한다.
