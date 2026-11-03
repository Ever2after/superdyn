# 작업 흐름

모든 명령은 워크스페이스 루트(`superdyn/`)에서 실행한다. 호스트 Python은 `unreal_synthetic/.venv/bin/python`이고, 에디터 내부 Python은 Unreal 설치본을 쓴다.

## 환경 확인

```bash
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py doctor
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py test
```

- **엔진 찾기:** `.engine` 링크, `SUPERDYN_UE_ENGINE`, `UE_ROOT`, `--engine` 중 하나로 찾는다.
  - 엔진 zip은 `.downloads/`에 보존한다.
  - 설치는 `manage.py install-engine --archive <zip>`로 한다.
- **인증:** `.auth/`는 로컬 Fab 인증 파일이다. 문서, 출력, git에 포함하지 않는다.
- **에셋:** Fab 팩은 저장소에 없다. [다운로드 안내](../FAB_DOWNLOAD.md)대로 각자 계정으로 취득해야 렌더할 수 있다.
- **테스트:** `test`의 일부 검사는 로컬 Fab 에셋과 보존된 출력이 있어야 통과한다.

## Synthetic split 파일럿 (현재 방향)

큐레이션한 환경 맵 위에 설계된 카메라와 객체 운동을 배치한다. 그다음 RGB와 GT(depth, instance id)를 렌더하고, real split과 같은 규칙으로 QA를 만든다.
[사양 초안](SYNTHETIC_SPLIT_SPEC.md) · [파일럿 보고서](../outputs/pilot_v2/REPORT.md)

```bash
cd unreal_synthetic
.venv/bin/python pilot.py survey --output outputs/<run>   # 맵 표면 래스터 + 개요 렌더
.venv/bin/python pilot.py detail --output outputs/<run>   # 후보 창의 0.5m 정밀 래스터
.venv/bin/python pilot.py plan   --output outputs/<run>   # 클립 설계 배치 (plan.json, plan_*.png)
.venv/bin/python pilot.py render --output outputs/<run> --gpu <id>
.venv/bin/python pilot.py qa     --output outputs/<run>   # GT 판독, gate, qa.json, 평가 입력
.venv/bin/python pilot.py report --output outputs/<run>   # REPORT.md, 컨택트 시트
```

- **원본 맵 보호:** 원본 맵은 저장하지 않는다. 생성된 persistent level이 원본 맵을 서브레벨로 스트리밍한다.
- **세션 분리:** 빌드와 survey는 `-nullrhi` 세션에서 돌리고, GPU 세션은 Movie Render Queue만 실행한다.
- **재실행 방지:** 각 시도는 `native/<attempt>/`에 기록한다. 내용이 있는 시도 폴더에는 다시 실행하지 않는다.

## 장면군 (main_scene_families_v3)

산업 야드, 도시 광장, 공원을 [단차 지형 생성기](TERRAIN_GENERATOR.md)와 에셋 조합으로 만든다. [장면군 상세](SCENE_FAMILIES.md)

```bash
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes plan --output unreal_synthetic/outputs/<run>
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes render --output unreal_synthetic/outputs/<run> --gpu 5
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py scenes verify --output unreal_synthetic/outputs/<run>
```

## 지형만 생성

```bash
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py terrain sample --seed 42 --output /tmp/terrain42.json
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py terrain audit --seed 0 --count 100 --output /tmp/terrain_audit.json
```

## A/B walkthrough 데모

[상세 문서](AB_WALKTHROUGHS.md)

```bash
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py tours verify
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py tours preview --gpu 7 --output unreal_synthetic/outputs/experiment_01
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py tours render  --gpu 7 --output unreal_synthetic/outputs/experiment_01
unreal_synthetic/.venv/bin/python unreal_synthetic/manage.py tours report  --output unreal_synthetic/outputs/experiment_01
```

- **검증(`verify`):** MP4를 끝까지 디코딩하고 RGB, 경로, 장면, 스냅샷 해시를 검사한다.
- **새 실험:** 별도 출력 폴더를 쓰고, 맵·시퀀스·대기 클립은 `/Game/SuperDyn/ABWalkthroughs/Experiments/<출력 폴더명>/`에 저장한다.
- **덮어쓰기:** 부분 렌더는 덮어쓰지 않는다.
