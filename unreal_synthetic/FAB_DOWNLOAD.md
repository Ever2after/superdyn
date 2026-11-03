# Fab 에셋 다운로드

현재 선택한 무료 팩 15개는 사용자 계정에서 라이브러리에 추가한 뒤 다운로드했다. 최종 장면에는 A·B 팩 14개를 사용하고 C 등급 용도는 제외한다. [에셋 개수·종류·이슈 표](docs/ASSETS.md).

다운로드 원본은 `inputs/source_demos/fab_downloads/`, 현재 선택·취득 기록은 `inputs/fab_selected_review/selection.json`에 보존한다. 원본·프로젝트 에셋은 삭제된 과거 데모 출력과 구분한다.

워크스페이스 루트의 대화형 터미널에서 실행한다.

```bash
bash unreal_synthetic/scripts/fab.sh status
bash unreal_synthetic/scripts/fab.sh library
bash unreal_synthetic/scripts/fab.sh download <asset-id>
```

라이브러리에 추가된 팩을 내려받는 도구다. 새 팩을 취득하려면 Fab 웹의 사용자 계정에서 먼저 라이브러리에 추가한다. `asset-id`는 `library`의 ID이며 웹 listing ID와 다를 수 있다. 도구는 UE 5.8 배포본을 요청한다. 배포본 호환성은 팩별로 확인한다.

로컬 로그인이 실제로 만료된 경우에만 다음을 실행한다.

```bash
bash unreal_synthetic/scripts/fab.sh login
```

인증 코드는 터미널에 입력한다. 인증 파일은 `.auth/fab/`에 보관하고 출력·문서·버전 관리에 포함하지 않는다. 도구는 프로젝트 내부 `.tools/bun/`과 비공식 `epic-fab` 구현을 사용한다. 실제 취득 계약의 범위는 계정에서 취득한 약관을 따른다.
