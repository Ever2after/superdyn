QA-filtered evaluation: keep only

2262 QA / 1247 videos; 125 exclude/review QA omitted.
Source audit: /home/jovyan/data-1/jusang/superdyn/motion_audit/outputs/temporal_geometry_fullscope/qa/qa_visual_filter_fullscope/results.json
Source predictions: /home/jovyan/data-1/jusang/superdyn/motion_audit/outputs/temporal_vlm_eval_fullscope/predictions
Existing responses are reparsed and rescored against the same automatic reference
answers. No new inference. Original source QA, predictions and full reports are preserved.
Invalid model answers remain wrong; all models use exactly the same retained QA.

results.png: overall accuracy, task accuracy, and raw LL/LH/HL/HH accuracy.
before_after.png/pdf: observed and fixed-task-mix comparisons before/after filtering.
task_standardized_groups.png/pdf: common task mix across four cohorts, with
video-cluster-robust normal 95% confidence intervals. Common weights remain the
original 2,387-QA task distribution, matching the previous unfiltered graph.
HL contains only 3 T5 and 4 T6 questions; interpret sparse cells cautiously.

leaderboard.csv, per_task.csv, per_group4.csv, per_group8.csv,
per_task_family.csv (tasks 1-2 versus 3-6), accuracy_by_group.csv (overall at right),
accuracy_by_task.csv, before_after.csv and task_standardized_groups.csv.
Raw random expectation changes from 30.6102% to 30.2019%.
References are model-estimated geometry labels. Filtering is an automatic Qwen
quality audit, not a human verification of reference correctness. No HTML generated.
