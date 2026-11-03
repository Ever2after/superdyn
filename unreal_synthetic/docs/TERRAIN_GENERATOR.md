# 연결된 단차 지형 생성기

`terrain/` 패키지는 시드로 직사각형 지면을 여러 평지 구획으로 나누고, 높이가 다른 구획을 경사로로 잇는다.
모든 형상은 코드로 만들며 외부 모델은 쓰지 않는다. 순수 NumPy 코드이고 렌더러를 호출하지 않는다.
Unreal 장면 가족(`scene_families.py`)은 이 레이아웃을 `terrain.layout.terrain_mesh()`로 메시화해 사용한다.

과거 Blender 툴킷(`synthetic/`)에서 지형 로직만 옮겨 왔다. Blender 렌더·갤러리·드론 경로 코드는 삭제했다.

| 모듈 | 역할 |
|---|---|
| [sampling.py](../terrain/sampling.py) | 구획 분할, 높이 배정, 경사로 배치, 거절 샘플링 (`generate`) |
| [height_patterns.py](../terrain/height_patterns.py) | 높은 지대·낮은 지대를 분산하는 높이 패턴 탐색 |
| [layout.py](../terrain/layout.py) | 높이 조회, 평지·경사면·단차 벽 메시 (`height`, `terrain_mesh`) |
| [validation.py](../terrain/validation.py) | 면적·경사로 접합·보행 연결성 검사 (`validate`, `raster`, `clearance_mask`) |
| [paths.py](../terrain/paths.py) | 보행 가능 격자 위 경로 탐색·단축·재샘플링 |
| [cli.py](../terrain/cli.py) | `manage.py terrain` 하위 명령 |
| [configs/terrain/](../configs/terrain/) | 샘플링 범위(`terraced_sampling.json`)와 초안 레이아웃(`terraced_draft.json`) |

## 명령

워크스페이스 루트에서 실행한다.

```bash
# 레이아웃 하나를 JSON으로 샘플링
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py terrain sample --seed 42 --output /tmp/terrain42.json

# 저장된 레이아웃 검증
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py terrain validate --layout /tmp/terrain42.json

# 시드 범위 스트레스 테스트
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py terrain audit --seed 0 --count 500 --output /tmp/terrain_audit.json
```

`--settings PATH`로 다른 샘플링 설정을 쓸 수 있다. 같은 시드와 설정은 레이아웃, 인물 위치, 주변 객체까지 똑같이 재현한다.

## 생성 과정

1. 지면 크기와 구획 수를 샘플링한다. 유효한 절단 위치, 최소 변 길이, 종횡비 제한을 지키며 직사각형을 재귀적으로 나눈다.
2. 경계를 공유하는 구획으로 인접 그래프를 만든다.
   - 서로 인접하지 않은 높은 구획 2개와 낮은 구획 2개를 고른다.
   - 나머지 구획에는 중간 높이를 샘플링한다.
   - 경사로로 그래프 전체를 이을 수 있을 때까지 중간 높이 배정을 바꿔 가며 탐색한다(`bsp_graph_terraces_v3`).
   - 면적 가중 높이를 X/Y 평면으로 회귀한 R²가 0.5를 넘으면 버린다. 한 방향으로만 높아지는 지형을 막기 위해서다.
3. 높이가 같은 이웃은 그대로 잇는다. 높이가 다른 구획 사이에는 무작위 신장 그래프를 만들고 연결을 2–5개 더 시도한다.
   - 경사로는 XY 면적의 30% 이상을 차지해야 한다.
   - 경사로 크기·측면 위치·구획 침투 깊이는 샘플링한다.
   - 착지 여유 공간은 다른 연결과 겹치지 않게 예약한다.
4. 면적, 폭, 높이, 경사, 연결성 조건을 어기면 버린다. 통과한 구획에는 충분히 넓은 평지 직사각형이 남는다.
5. 평지에는 축척용 인물을, 경사로에는 로버를, 여유 공간에는 주변 블록을 배치한다.
   - 수락된 레이아웃 전체, 시드, 설정, 분할 이력, 그래프, 거절 사유별 횟수를 저장한다.

실패한 시드를 고정 레이아웃으로 대체하지 않는다. 재시도는 `maximum_attempts`로 제한하고, 조건이 서로 맞지 않으면 거절 사유별 횟수와 함께 실패를 반환한다.

## 검사

- 구획이 지면을 정확히 한 번씩 덮는다.
- 공간적으로 분리된 최고 지대 2곳 이상, 최저 지대 2곳 이상이 있고 각각 지면의 12% 이상을 차지한다.
  - 높이가 같은 인접 구획은 한 지대로 센다.
- 평면 추세 R² ≤ 0.5. 검증은 저장된 메타데이터가 아니라 형상에서 지표를 다시 계산한다.
- 크기는 길이 20–24m, 폭 18–22m, 구획 7–10개.
- 평지 비율은 50–70%. `maximum_flat_fraction`은 연결성과 별개로 평지만 넓은 레이아웃을 거절한다.
- 모든 구획에 최소 변 2.25m, 면적 8m² 이상인 평지 직사각형이 남는다.
- 경사로 끝은 전체 폭에서 이웃 바닥과 맞닿고, 착지 여유 공간은 다른 경사로를 넣은 뒤에도 비어 있다.
- 경사로 최대 경사는 기본 0.6.
- 보행 격자의 모든 셀에 도달할 수 있다.
- 구획 중심과 경사로 중심은 절벽·외곽에서 0.75m 여유를 둔 상태로 연결돼 있다.

## 설정

`configs/terrain/terraced_sampling.json`의 주요 항목은 다음과 같다.

- 크기와 구획: 지면 길이·폭 범위, 구획 수와 최소 크기
- 높이: 고도 단계 수와 높이 증분
- 경사로와 이동: 경사로 폭·길이, 최대 경사, 평지 요건, 착지 여유, 보행 여유 반경

`height_assignment`은 극값 최소 면적, 평면 추세 상한, 탐색 제한을 정한다.
저장된 레이아웃은 생성 당시의 설정 전체를 함께 담고 있어서, 기본 설정이 바뀌어도 검증 결과가 변하지 않는다.
단위는 world 미터이고 Z가 위쪽이다.
