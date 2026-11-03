# 에셋 목록과 채택 기준

현재의 **벽·HDRI가 있는 산업 지형에서 비생성형 검출·추적용 영상을 만드는 목적**으로 평가했다. 개수는 받은 배포본의 Unreal 에셋 레지스트리 집계이며, 외형·동작 판단은 대표 모델 48개 비교에 근거한다.

**SK** = 스켈레탈 메시(뼈대 모델), **SM** = 정적 메시, **A** = AnimSequence 동작 클립 파일 수. 메시 수에는 부품·스킨·표시용 모델이 포함되며 서로 다른 사람·차종 수와 일치하지 않는다. 클립 수도 고유 행동 종류 수와 다르다.

**현재 A·B만 채택하고 C는 제외한다.** A는 우선 채택, B는 용도에 맞게 보완 채택한다. 아래 C 항목은 제외 사유를 남긴 기록이다. 이는 기술적 추천도다.

| 분야 · 소스 | 모델·동작 수 | 주요 종류 | 확인된 이슈·한계 | 최종 추천도 |
| --- | --- | --- | --- | --- |
| 사람 · [Game Animation Sample](https://www.fab.com/listings/880e319a-a59e-4ed2-b268-b32dac7fa016) | SK 20 / SM 12 / A 2,008 | UEFN·Manny·Quinn 마네킹, 보행 등 사람 동작 | 실사 얼굴·의상 다양성 부족; 일부 실험용 데이터 호환 경고; 전체 Motion Matching 미검증 | **A · 사람 동작 기반** |
| 사람 · [Paragon: Twinblast](https://www.fab.com/listings/9fa88852-5711-42e1-94fa-2491498a64da) | SK 10 / SM 34 / A 255 | 남성 캐릭터; 기본·ActionHero·SummerTime 등 스킨 | 무장·조준 자세 중심; ActionHero 코트·머리카락 점상 노이즈; 빈 미사용 슬롯 보정 | **B · 인물 외형 보강** |
| 사람 · [Paragon: Sparrow](https://www.fab.com/listings/7d76ddf0-d9ce-4d00-939e-d72793534d01) | SK 8 / SM 18 / A 152 | 여성 궁수; 기본·Rogue·Monarch 등 스킨 | 활·판타지 복장 중심으로 일상 보행자 분포와 차이 | **B · 의상·실루엣 보강** |
| 사람 · [Paragon: Gadget](https://www.fab.com/listings/52a621fd-28bb-4898-a2e7-93229a40e3f4) | SK 9 / SM 28 / A 103 | 여성 SF 캐릭터; 기본·Guerilla·Winter 등 스킨 | 기계 장비·전투 동작 중심; 일상 사람 외형 보완 필요 | **B · 특수 복장 보강** |
| 차량 · [City Sample Vehicles](https://www.fab.com/listings/2909157b-ddfa-4cef-a925-69dc2467021f) | SK 12 / SM 118 / A 0 | 승용차·밴·트럭·버스·트레일러 | 표본 소형차 표시 메시 LOD0 약 78.7만 정점; 주행 물리 미검증 | **A · 차량 기본 소스** |
| 차량 · [Vehicle Variety Pack](https://www.fab.com/listings/dc1ada50-2523-44b1-b0e2-a72d14076fb4) | SK 6 / SM 7 / A 0 | 해치백·픽업·SUV·스포츠카·박스 트럭·트럭 섀시 | 픽업 표본 LOD 1개; 바퀴 회전·주행 물리 미검증 | **A · 차종 다양화** |
| 차량 · [Vehicle Variety Pack Volume 2](https://www.fab.com/listings/591e3b3f-9d49-4cd2-8e28-d471c1a10cab) | SK 4 / SM 4 / A 0 | 캠핑카·세단·SUV·박스 트럭 | 캠핑카 표본 LOD 1개; 주행 물리 미검증 | **A · 특수 차종 보강** |
| 동물 · [ANIMAL VARIETY PACK](https://www.fab.com/listings/2dd7964c-a601-4264-a53d-465dcae1644c) | SK 6 / SM 0 / A 146 | 늑대·여우·사슴 암수·돼지·까마귀: 5종류·6모델 | 근접 털 표현은 게임 모델 수준; 발 접지 보정 필요 | **A · 동물 기본 소스** |
| 조형물 · [Construction Site VOL. 1](https://www.fab.com/listings/ba44a508-bfa5-444c-bbf4-69e8b5dee530) | SK 0 / SM 73 / A 0 | 벽돌 팔레트·목재 상자·케이블 릴 등 건설 소품 | 검사 표본에서 큰 외관 문제 없음; 전체 충돌·모든 소품 미검증 | **A · 현재 야드 우선** |
| 조형물 · [Industry Props Pack 6](https://www.fab.com/listings/b5603e44-e1b0-4346-9c3d-04887aa9f87d) | SK 0 / SM 30 / A 0 | 드럼통·나무 상자·호스 등 산업 소품 | 검사 표본에서 큰 외관 문제 없음; 전체 충돌 미검증 | **A · 가벼운 소품 보강** |
| 조형물 · [Unfinished Building](https://www.fab.com/listings/25f2e7e5-5cca-48a5-99a3-35c38b8240ac) | SK 0 / SM 95 / A 0 | 벽·기둥·계단·공사 중 구조물·고밀도 Megascans 소품 | 원본 맵의 재질 오버라이드 필요; 기둥 복구 확인; 벽·계단 보정 렌더 미완료; 첫 캐시 생성 부담 | **C · 재질 설정 후 채택** |
| 조형물 · [Modular Building Set](https://www.fab.com/listings/474a0598-ed86-40b6-baa1-c801d96ef4ab) | SK 0 / SM 137 / A 0 | 상점 문·벽돌 외벽·금속 기둥 등 건물 모듈 | 벽 표본은 단면이라 뒤에서 보이지 않음; 면 방향·두께 구성 필요 | **B · 외벽·문 조립용** |
| 배경 · [City Sample Buildings](https://www.fab.com/listings/008fe959-5511-428e-93bd-f99b1179f6d5) | SK 0 / SM 2,390 / A 0 | 고층 건물 외벽·창문·코너·장식·조립용 BP | 원본 15.35GB; 모듈 수가 완성 건물 수는 아님; 건물 조립 작업 필요 | **B · 도시 테마에서는 A** |
| 배경 · [City Park Environment Collection](https://www.fab.com/listings/11cc2abb-126c-4452-9fe4-6f2381d96544) | SK 0 / SM 524 / A 0 | 나무·식생·벤치·카페·도로·공원 시설 | 개별 소품 정상; 원본 맵은 현재 리뷰 설정에서 과노출·식생 노이즈 | **B · 개별 소품 / C · 전체 맵** |
| 배경 · [Modular Rural Cabins](https://www.fab.com/listings/508fe84a-4976-4cfe-9a40-c2b9533da601) | SK 0 / SM 160 / A 0 | 통나무 벽·문·창·오두막·소나무·시골 소품 | 개별 모듈 정상; 원본 맵은 현재 리뷰 카메라에서 심한 과노출 | **B · 개별 모듈 / C · 전체 맵** |

## 해석할 때 구분할 점

- 차량의 A 0은 AnimSequence 파일이 없다는 뜻이다. 차량 리그·AnimBlueprint·차량 Blueprint는 별도로 있으며, 실제 주행 제어의 검증 결과를 뜻하지 않는다.
- 사람·동물에는 공통으로 발 IK·접지 보정과 이동 속도 동기화가 필요하다. 이번 비교는 크기를 정규화한 외형·클립 검사다.
- 공원·오두막 원본 맵은 원본 카메라의 위치·화각을 가져왔고 후처리 전체는 이식하지 않았다. 과노출이 관찰돼 완성 장면의 외관 평가는 보류했다. 개별 모듈 평가는 현재 산업 지형의 정상 출력 결과를 근거로 한다.
- 공유 GPU의 다른 계산 작업 때문에 팩 단독의 VRAM·렌더 속도는 확정하지 않았다.
- 추천도는 AI 이용 허가 확정과 구분한다. 실제 취득 약관 관련 범위는 [취득 기록](#취득-기록)에 기록했다.

## 전체 등록 에셋 수와 보관 용량

전체 등록 수는 메시 외에 재질·텍스처·사운드·Blueprint·레벨 등도 포함한다. README·설정 파일 등을 포함하는 다운로드 원본 파일 수와는 다른 집계다. GB는 10진 단위의 매니페스트상 원본 파일 크기 합계이며 프로젝트 복사·캐시는 별도다.

| 소스 | 등록 에셋 전체 | Blueprint | 레벨(World) | 원본 GB |
| --- | ---: | ---: | ---: | ---: |
| Game Animation Sample | 3,883 | 127 | 5 | 7.46 |
| Paragon: Twinblast | 1,984 | 1 | 3 | 2.48 |
| Paragon: Sparrow | 1,723 | 1 | 2 | 3.21 |
| Paragon: Gadget | 1,454 | 1 | 3 | 1.80 |
| City Sample Vehicles | 1,405 | 47 | 1 | 7.94 |
| Vehicle Variety Pack | 156 | 20 | 2 | 1.47 |
| Vehicle Variety Pack Volume 2 | 154 | 16 | 1 | 1.41 |
| ANIMAL VARIETY PACK | 220 | 0 | 5 | 0.52 |
| Construction Site VOL. 1 | 239 | 20 | 2 | 0.62 |
| Industry Props Pack 6 | 94 | 0 | 2 | 0.19 |
| Unfinished Building | 382 | 4 | 1 | 3.28 |
| Modular Building Set | 317 | 0 | 2 | 0.77 |
| City Sample Buildings | 10,779 | 76 | 97 | 15.35 |
| City Park Environment Collection | 1,117 | 1 | 3 | 4.37 |
| Modular Rural Cabins | 586 | 32 | 5 | 1.66 |

**먼저 묶어서 쓸 조합:** GASP 동작 + Paragon 외형 보강 / City Sample Vehicles + Vehicle Variety 1·2 / Animal Variety / Construction Site + Industry Props. 도시 건물·공원·오두막은 필요한 모듈을 선택해 추가한다.

[최종 배치 영상](../outputs/ab_walkthroughs/REPORT.md) · [원본 집계](../inputs/asset_catalog/inventory_summary.json) · [팩별 레지스트리](../inputs/asset_catalog/inventory)

이전 소스별 비교 영상·프리뷰는 정리했다. 위 표는 당시 평가 기록이며, 현재 최종 영상에 실제 배치된 메시와 팩별 전체 개수는 구분한다.

## 취득 기록

15개 팩은 사용자 계정 라이브러리에서 취득해 다운로드했다. 다운로드 API에는 실제 취득 계약 전문이 포함되지 않으므로 이 표는 AI 이용 허가를 확정하는 문서가 아니다. MetaHuman/Crowd는 최종 배치 모델에 포함하지 않았다. HDRI의 출처·CC0·해시는 `inputs/lighting/`에 보존했다. [다운로드 도구](../FAB_DOWNLOAD.md).
