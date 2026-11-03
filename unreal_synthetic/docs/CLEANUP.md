# 정리 기록

## 2026-11-03: synthetic 통합과 업로드 준비

- **`synthetic/`(Blender 툴킷) 삭제.** 삭제 범위는 Blender 5.2.2 바이너리, 렌더·갤러리·드론 코드, HDRI·캐릭터 에셋, 테스트다.
  - 코드·설정·문서·에셋은 저장소 밖 `../superdyn_archive/synthetic_blender_20261103.tar.gz`에 백업했다.
- **지형 생성 로직 이전.** [terrain/](../terrain/)(`sampling`, `height_patterns`, `layout`, `validation`, `paths`, `cli`)과 [configs/terrain/](../configs/terrain/)로 옮겼다.
  - [문서](TERRAIN_GENERATOR.md)와 `tests/test_terrain.py`(12개)를 함께 옮겼다.
  - `scene_families.py`가 새 패키지를 사용한다.
- **Blender 변환 스크립트 삭제.** `scripts/prepare.py`, `scripts/export_scene.py`와 `manage.py prepare`를 삭제했다.
  - 이미 변환한 `inputs/0*/scene.json.gz`는 그대로 사용한다.
- **호스트 venv 이동.** `synthetic/.venv`를 `unreal_synthetic/.venv`로 옮기고 경로를 고쳤다. 의존성은 [requirements.txt](../requirements.txt)에 있다.
- **로그·캐시 삭제.**
  - 출력과 보고서의 `*.log` 130개(54MB)
  - `Saved/Logs`, `Saved/Crashes`, `Saved/ShaderDebugInfo`
  - Python 바이트코드
- **재실행 방지 조건 변경.** `pilot.py`가 `editor.log` 대신 시도 폴더에 내용이 있는지로 판단한다.
- **README 정리.** README를 개요로 줄이고, walkthrough 상세 내용은 [AB_WALKTHROUGHS.md](AB_WALKTHROUGHS.md)로 옮겼다.
- **검사.** 정리 후 오프라인 테스트 49개(기존 37개 + 지형 12개)가 통과했다.

## 2026-11-02: 최종 산출물 기준 정리

당시 최종 산출물은 [A·B 지형 1인칭 영상 3개](../outputs/ab_walkthroughs/REPORT.md)다.

삭제 전 집계한 **18,885개 파일·논리 크기 27.15GB**를 정리했다. 블록 할당량 합계는 27.28GB다. 공유 디스크 전체 여유 공간의 변화를 뜻하지 않는다.

| 정리 대상 | 경로 수 | 파일 수 | 논리 GB |
| --- | ---: | ---: | ---: |
| 이전 Unreal 출력 | 7 | 9,800 | 12.927 |
| 이전 Blender 출력 | 6 | 7,522 | 11.920 |
| 최종 폴더 안의 개발 시도·프리뷰 | 23 | 1,552 | 2.299 |
| 통합·폐기한 Markdown | 11 | 11 | 0.000 |

### 보존한 것

- `outputs/ab_walkthroughs/`의 최종 MP4 3개·RGB 2,880개·경로·검증 영수증·배치 기록·보고서·렌더 당시 소스 스냅샷.
- `/Game/SuperDyn/ABWalkthroughs/`의 맵·시퀀스·대기 클립과 이 장면에 필요한 원본 `Content/` 에셋. 에셋 의존성을 임의로 삭제하지 않았다.
- 지형 입력 네 개, 다운로드 원본·출처·라이선스·SHA 기록, 엔진·Blender·Python 환경·로컬 도구·인증 파일.
- 과거 소스 27개와 진행 기록은 [legacy](../legacy/README.md)로 이동해 현재 실행 코드에서 분리했다. 소스 내용·SHA는 유지했다.
- `motion_audit/`는 이번 작업 범위에서 제외했다.

프로젝트 소스의 Python 바이트코드 캐시 154개·1.53MB도 삭제했다. 엔진·환경 캐시는 보존한다.

### 리팩토링

- `inputs/ab_walkthroughs/placements.json`과 `height_grid.json`을 독립 입력으로 만들었다. 삭제된 Fab 리뷰의 `render_request.json`을 읽던 의존성을 제거했다.
- `manage.py tours verify|report|preview|retry_preview|render`로 현재 명령을 모았다. 완료된 결과에서 프리뷰를 재생성하지 않으며, 새 실험의 맵·시퀀스·모션은 별도 네임스페이스에 저장한다.
- 이미 검증 영수증이 있어도 영상·프레임 해시를 확인하고 나서 재사용한다. 인코딩은 실제 경로 ID와 명시적 기반 지형 입력을 사용한다.
- 문서 재생성과 렌더 당시 증거를 분리했다. 원본 매니페스트와 소스 스냅샷은 덮어쓰지 않는다.
- 기존 `prepare`의 삭제된 출력 기본 경로를 없애고 `--source`를 필수로 바꿨다.
- 원래 배치 레시피와 최종 입력의 일치, 설치 에셋 존재, 손상 영상 거부, 최종 결과 덮어쓰기 방지를 회귀 검사한다.

### 확인 결과

- 최종 영상 전체 2,880프레임 디코딩, RGB 2,880개 해시, 경로·배치·렌더 당시 스냅샷과 저장 Unreal 에셋 해시를 확인했다.
- 삭제 전 보호 목록 26개 파일의 해시가 정리 후에도 같다. [보호 목록](../reports/protected_deliverables.json).
- Unreal 오프라인 회귀 검사 24개·Blender 지형 검사 55개 통과. 현재 Markdown 10개, 로컬 링크 50개, HTTP 링크 6개를 확인했다.
- 리팩토링 뒤 새 GPU 렌더는 실행하지 않았다. 최종 영상은 이전에 검증된 원본을 그대로 보존했다.
- 자동 승인 검토에서 소스 코드까지 삭제하는 넓은 정리 범위가 거절되어, 소스 삭제 없이 이전 출력·개발 프리뷰·폐기 문서만 삭제했다.

[삭제 목록·크기](../reports/cleanup_manifest.json) · [최종 미디어 검사](../reports/walkthrough_verification.json) · [정리 후 종합 검사](../reports/cleanup_verification.json) · [실행 방법](WORKFLOW.md)
