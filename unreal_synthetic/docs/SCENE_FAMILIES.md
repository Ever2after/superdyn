# Synthetic main 장면군

`configs/main_scene_families.json`은 산업 야드·도시 광장·공원을 서로 다른 지형 seed와
공간 구성·에셋 조합으로 정의한다. 현재 단계는 세 장면군의 생성과 시각 검증이다.
카메라/객체 운동 조합 96개와 T1–T6 QA는 후속 단계다.

## 지형과 공간 구성

[단차 지형 생성기](TERRAIN_GENERATOR.md)(`terrain/`)를 사용한다. 단일 지형을 재질만 바꾸지 않고 세 seed를
독립적으로 샘플링한다. 수평 배율을 적용할 때 분석 격자·메시 격자·착지부·최소 평지
크기와 검증 설정도 함께 변환한다. 높이 값은 유지하므로 배율이 클수록 경사는 완만해진다.

| 장면군 | 기본 seed | 수평 배율 | 구성 |
|---|---:|---:|---|
| 산업 야드 | 101 | 1.25 | 아스팔트 단차, 창고·하역문, 컨테이너·차량·묶음 적재물 |
| 도시 광장 | 202 | 1.35 | 포장 광장, 상점·벽돌 외벽의 건물 체적, 벤치·가로등·차량 |
| 공원 | 303 | 1.50 | 잔디 단차, 포장 산책로, 나무·관목·바위·목조 쉼터·동물 |

세 장면 모두 기본 생성기의 평지 비율 50–70%, 여러 고지대·저지대, 연결된 경사로,
최소 고저차 3m 조건을 검사한다. 기존 사방 콘크리트 경계벽은 사용하지 않는다.
바깥에는 핵심 보행면과 겹치지 않는 지형 skirt와 장면에 맞는 구조물을 둔다.

서로 다른 높이를 연결하는 A* 경로를 먼저 선택한다. 높이 조회는 원래 메시의 삼각형
대각선과 동일하며, 경로를 보간한 뒤에도 실제 지면 높이를 다시 계산한다. 이 경로의
여유 공간을 예약한 다음 실제 Unreal 메시 바운드를 사용해 객체를 배치한다.
배치 후보는 지형 외곽·단차·다른 객체·예약 통로를 피해야 한다. 나무는 줄기 반경
40cm로 검사하므로 수관 전체의 충돌 인증은 아니다.

## 실행

워크스페이스 루트에서 실행한다. 기존 walkthrough 입력·출력은 수정하지 않는다.

```bash
# CPU로 지형·메시 패킷·배치 recipe·검사 경로 생성
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes plan

# 준비된 장면을 Unreal에서 구축·검증·렌더
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes render --gpu 5

# 저장 맵·소스 스냅샷·RGB·MP4·배치·지형 검증
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes verify

# 새 seed와 별도 네임스페이스. 출력 디렉터리는 비어 있어야 한다.
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes render \
  --output unreal_synthetic/outputs/main_scene_families_seed10 \
  --seed-offset 10 --gpu 5

# 한 장면군만 생성 가능
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes render \
  --family park --output unreal_synthetic/outputs/park_seed20 --seed-offset 20 --gpu 5
```

GPU 번호는 현재 공유 장비의 가용 상태에 맞춰 지정한다. 이미 완료되었거나 native 시도를
시작한 출력 디렉터리에 새 렌더를 덮어쓰지 않는다. 실패한 시도의 로그도 보존하고 새
출력 네임스페이스로 다시 생성한다. 같은 장면 seed를 사용할 수 있다.

## 검사와 산출물

`outputs/main_scene_families_v3/REPORT.md`가 최종 확인 문서다. 조명·외벽 방향·도시 창문·
외곽 연결을 점검한 뒤 보정한 버전이며, 이전 렌더는 `outputs/main_scene_families/`와
`outputs/main_scene_families_v2/`에 보존한다.
각 장면 폴더는 다음을 담는다.

| 파일 | 의미 |
|---|---|
| `layout.json`, `terrain_validation.json` | 실제 샘플링 설정·지형·연결성·경사·면적 검증 |
| `recipe.json`, `scene.json.gz` | 장면군·에셋 요청과 독립적인 Unreal 메시/카메라 입력 |
| `route.json`, `layout.png` | 단차를 통과하는 예약 경로와 배치도 |
| `scene_manifest.json` | 실제 선택된 배치·크기·소스·최종 보정 후 객체 행렬 |
| `native_validation.json` | 실제 지형 collision ray query와 전 프레임 카메라 검증 |
| `evaluated_scene.json` | 실제 평가된 객체 행렬과 192개 카메라 pose |
| `walk/` | 8초·720p·24fps RGB와 MP4 |
| `overview/`, `contact_sheet.jpg` | 네 방향 전체 조망과 지면 시점 검사 이미지 |

Unreal 저장 위치는 `/Game/SuperDyn/SyntheticMain/<출력 폴더명>/`이다.
원본 팩 메시·재질·애니메이션은 직접 수정하지 않고, idle 클립은 생성 네임스페이스에 복사한다.
저장된 맵·메시·시퀀스·복사 클립과 생성 근거의 해시를 최종 manifest에 기록한다.

이 단계의 차량은 정차하고 사람·동물은 idle을 재생한다. 현재 평가 상태 출력은 dense
surface-point GT나 instance/depth 패스 완성을 뜻하지 않는다. 장면군 확인 후 객체별
이동 경로, 독립 카메라 조건, 표면점·픽셀 GT와 QA를 추가한다.
