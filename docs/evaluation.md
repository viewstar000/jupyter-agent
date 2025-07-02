# 评估数据

- 评估用例：<../examples/data_loader_eval.ipynb>
- 生成模型：
  - 推理：qwen3-30b-a3b
  - 编码：devstral-small-2505-mlx
- 评估模型：qwen3-30b-a3b
- 模型部署：本地 lmstudio
- 测试环境：MacBook Pro 2024 M4 Max 64G

## 测试结果

### 执行成功率

#### 整体执行成功率

| eval_type | success | total |     rate |
| :-------- | ------: | ----: | -------: |
| NOTEBOOK  |       7 |    12 | 0.583333 |
| STAGE     |     329 |   358 | 0.918994 |
| FLOW      |      64 |    69 | 0.927536 |

- NOTEBOOK: 是否完成全局目标，不考虑完成质量，只要完成全局目标即可
- FLOW: 子任务是否成功执行，不考虑子任务完成质量，只要完成子任务是否完成所有步骤无异常即可
- STAGE: 子任务中的单个步骤是否成功执行，不考虑子任务完成质量，只要该步骤成功执行无异常即可

#### 分Stage执行成功率

| flow               | stage                        | success | total |     rate |
| :----------------- | :--------------------------- | ------: | ----: | -------: |
| TaskExecutorFlowV3 | TaskStage.EXECUTING          |      48 |    73 | 0.657534 |
| TaskExecutorFlowV3 | TaskStage.REASONING          |       4 |     5 |      0.8 |
| TaskExecutorFlowV3 | planning_paused              |      12 |    14 | 0.857143 |
| TaskExecutorFlowV3 | TaskStage.SUMMARY            |      48 |    49 | 0.979592 |
| MasterPlannerFlow  | start                        |      12 |    12 |        1 |
| TaskExecutorFlowV3 | TaskStage.CODING             |      53 |    53 |        1 |
| TaskExecutorFlowV3 | TaskStage.DEBUGGING          |      20 |    20 |        1 |
| TaskExecutorFlowV3 | TaskStage.PLANNING           |      64 |    64 |        1 |
| TaskExecutorFlowV3 | TaskStage.PLANNING_PAUSED    |       2 |     2 |        1 |
| TaskExecutorFlowV3 | TaskStage.PREPARE_NEXT       |      13 |    13 |        1 |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_ABOVE |      14 |    14 |        1 |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_BELOW |      39 |    39 |        1 |

### 执行时长

| eval_type | duration_avg | duration_std | duration_min | duration_max |
| :-------- | -----------: | -----------: | -----------: | -----------: |
| STAGE     |      29.4403 |      26.7122 |      1.00433 |      158.829 |
| FLOW      |      99.3356 |      101.221 |            0 |      500.206 |
| NOTEBOOK  |      1030.71 |      290.688 |      617.438 |      1697.06 |

### 生成质量

| flow               | score_type  |      avg |       std |
| :----------------- | :---------- | -------: | --------: |
| MasterPlannerFlow  | correct     | 0.838889 | 0.0600925 |
| TaskExecutorFlowV3 | correct     |     0.93 | 0.0453689 |
| TaskExecutorFlowV3 | planning    | 0.829423 | 0.0647594 |
| TaskExecutorFlowV3 | reasoning   | 0.859615 | 0.0955061 |
| TaskExecutorFlowV3 | coding      | 0.734902 | 0.0445588 |
| TaskExecutorFlowV3 | important   | 0.881373 | 0.0663029 |
| TaskExecutorFlowV3 | user_supply | 0.785577 | 0.0628853 |

- 质量评分使用评估模型（qwen3-30b-a3b）对生成的结果自动评分得到，评分越高质量越好
  - correct: 正确性评估，生成的结果是否符合当前规划的要求
  - planning: 规划质量，规划的具体步骤是否符合全局目标
  - reasoning: 推理质量，推理的结果是否准确、合理
  - coding: 代码生成质量，代码是否符合规划的要求，是否无语法错误、逻辑错误、冗余等
  - important: 重要性质量，生成的结果是否充分的考虑了每个子任务生成的“Important infos”
  - user_supply: 用户补充质量，生成的结果是否充分的考虑了用户的补充信息
