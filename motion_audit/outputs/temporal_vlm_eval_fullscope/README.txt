Full-scope evaluation archive and shared inference assets

Original evaluation: 2,387 QA / 1,275 videos / 7 models.
The selected final benchmark is final/: 896 QA / 461 videos.
Final questions: ../temporal_geometry_fullscope/qa/final/qa.json
Project documentation and supported commands: ../../README.md

final/leaderboard.csv and final/results.png/pdf are the final results.
Root leaderboard.csv, summary.json, protocol.json, inputs.json and
reference_answers.json describe the original, unfiltered 2,387-QA evaluation.
Root predictions/ and media/ are shared source assets; retain them because the
final benchmark and audit outputs use these files. Root media_sha256.json and
code_versions.json preserve the original input/code provenance.

subsets/qa_visual_keep/: keep-only 2,262-QA comparison across four cohorts.
subsets/HNH_NHH_HHH/: historical 957-QA group-only report before QA filtering.
statistical_analysis/: historical full-scope group and task-mix analysis.
These older reports remain distinct from final/.

Active inference/adapters/reporting: ../../qa_pipeline/evaluation/.
Use ../../qa.py finalize to rebuild the frozen final export and graphs, or
../../qa.py report to rescore the fixed final IDs without inference.
For fresh model inference, create a separate output and use qa.py evaluate;
see the root README for pinned model and GPU/shard configuration examples.

Each model uses the same nine marked RGB frames, original question and ordered
options. References and geometry evidence are separate from adapter inputs.
Invalid answers count as wrong. Reference geometry is automatically estimated;
QA audit decisions check visual interpretability, not human-verified correctness.
Original source QA, references, model revisions and raw predictions are unchanged.
