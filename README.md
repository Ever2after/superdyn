# SuperDyn: Temporal Geometry QA

카메라 운동과 장면 동역학이 **함께 큰** 영상에서 시간에 따른 3D 방향·거리·이동량을 추론하는 능력을 평가하는 4D 이해 벤치마크다.
카메라만 움직이거나 객체만 움직이는 영상에서는 2D 변화만으로 풀리는 질문이 많다. 이 벤치마크는 카메라 운동과 객체 운동을 분리해서 묻는다.

| Split | 폴더 | 정답 출처 | 현재 규모 |
|---|---|---|---|
| **Real** | [motion_audit/](motion_audit/README.md) | DynamicVerse 실제 영상(DAVIS, MOSE, MVS-Synth, SA-V, VOST, UVO, YouTube-VIS)에서 VGGT-Omega pose/depth, GMOS mask, RAFT 추적으로 자동 생성. Qwen이 QA 단위로 시각 검수 | 최종(HH) **896 QA / 461 영상**. 검수 통과 전체 2,262 QA / 1,247 영상. baseline 7종 평가 |
| **Synthetic** | [unreal_synthetic/](unreal_synthetic/README.md) | Unreal Engine 5.8 렌더러 GT(depth, instance, pose). 카메라·객체 운동을 설계하고 counterfactual 쌍을 구성 | 파일럿: 클립 12개, QA 35개 |

## 과제

모든 QA는 F0–F8의 measurement frame 9장을 보고 답한다. 객체 A와 B는 각각 주황·초록 윤곽으로 표시한다.

| Task | 질문 | 선택지 |
|---|---|---|
| T1 Camera translation direction | F0 카메라 축 기준 카메라 이동 방향 | 6 |
| T2 Camera view direction change | 시선(yaw/pitch) 회전 방향 | 4 |
| T3 Object world motion direction | 카메라 운동을 보상한 객체의 3D 이동 방향 | 6 |
| T4 Camera–object distance change | 카메라와 객체 사이 거리 변화 | 2 |
| T5 Object–object distance change | 두 객체 사이 거리 변화 | 2 |
| T6 Object travel distance comparison | 두 객체의 누적 이동량 비교 | 2 |

정의, 임계값, 필터링, 평가 결과는 [real split README](motion_audit/README.md)에 있다.

## 데이터

**Hugging Face: [`Ever2after/superdyn-real`](https://huggingface.co/datasets/Ever2after/superdyn-real)**

- **규모:** real split의 검수 통과 2,262 QA와 원본 영상 1,247개
- **포함 내용:**
  - QA 레코드
  - 객체 설명, 마스크, 3D anchor 근거
  - 모델 입력 프레임
  - 원본 RGB 프레임
- **최종 896 QA:** 하위 집합 목록으로 함께 제공한다.

## 저장소 구조

```text
motion_audit/        real split: inventory → geometry → QA 생성 → 시각 검수 → 최종 선택 → baseline 평가 (qa.py 단일 CLI)
unreal_synthetic/    synthetic split: 환경 survey·클립 설계·렌더·GT/QA (pilot.py), 지형 생성기(terrain/), 장면군·walkthrough
```

각 폴더 README에 설치 방법과 실행 명령이 있다. Python 환경은 폴더별 `.venv`에 있으며, 의존성 목록은 각 폴더의 `requirements.txt`다.

## git에 포함하지 않는 것

다음은 로컬에만 있다. 정확한 목록은 [.gitignore](.gitignore)에 있다.

- **real split 원본과 중간 산출물:** 원본 영상, geometry/flow 캐시, 표시 프레임, 모델 예측(공개 데이터는 Hugging Face 사용)
- **모델:** 가중치(`checkpoints/`)와 baseline 공식 저장소(`vendor/`)
- **Unreal 엔진과 Fab 에셋:** Fab 에셋은 라이선스상 재배포할 수 없어 각자 취득해야 한다.
- **렌더 결과:** RGB, EXR, MP4
