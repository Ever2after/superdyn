# Temporal geometry data

최종 세트: **896 QA / 461개 영상**.

- 사용할 QA: [qa/final/qa.json](qa/final/qa.json), [qa/final/qa.jsonl](qa/final/qa.jsonl)
- 고정 선택과 출처: [qa/final/manifest.json](qa/final/manifest.json)
- 통계: [qa/final/qa_counts.csv](qa/final/qa_counts.csv)
- 평가: [../temporal_vlm_eval_fullscope/final](../temporal_vlm_eval_fullscope/final)
- 코드·명령어·기준: [프로젝트 README](../../README.md)

`qa/qa.json`은 필터 전 원본 2,387 QA입니다. `qa/qa_visual_filter_fullscope/`는
전체 검수 결과이며, decision=keep 중 HLH/LHH/HHH만 `qa/final/`에 포함했습니다.
N은 메타데이터의 L과 같은 표기입니다.

`geometry/`, `qa/cases/`, `qa/source/`는 생성 근거와 공유 캐시입니다.
최종 세트는 원본 사례와 평가 프레임을 링크로 공유하므로 부모 자료도 보관해야 합니다.
기존 pilot/worker 이력과 exclude/review 근거는 보존합니다.

기하 정답은 VGGT-Omega/RAFT 기반 추정치이며 Qwen은 QA의 시각적 해석 가능성을 검수했습니다.
