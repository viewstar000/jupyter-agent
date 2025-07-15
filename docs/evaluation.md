# 评估数据

- 评估用例：<../examples/data_loader_eval.ipynb>
- 生成模型：
  - 推理：qwen3-30b-a3b with mlx@4bit
  - 编码：devstral-small-2505 with mlx@4bit
- 评估模型：qwen3-30b-a3b
- 模型部署：本地 lmstudio
- 测试环境：MacBook Pro 2024 M4 Max 64G

## 测试结果

### 执行成功率

#### 整体执行成功率

| eval_type   |   success |   total |     rate |
|:------------|----------:|--------:|---------:|
| NOTEBOOK    |         7 |      10 | 0.7      |
| STAGE       |       247 |     262 | 0.942748 |
| FLOW        |        55 |      58 | 0.948276 |

- NOTEBOOK: 是否完成全局目标，不考虑完成质量，只要完成全局目标即可
- FLOW: 子任务是否成功执行，不考虑子任务完成质量，只要完成子任务是否完成所有步骤无异常即可
- STAGE: 子任务中的单个步骤是否成功执行，不考虑子任务完成质量，只要该步骤成功执行无异常即可

#### 分 Stage 执行成功率

| flow               | stage                        |   success |   total |     rate |
|:-------------------|:-----------------------------|----------:|--------:|---------:|
| TaskExecutorFlowV3 | TaskStage.EXECUTING          |        39 |      53 | 0.735849 |
| TaskExecutorFlowV3 | TaskStage.SUMMARY            |        39 |      40 | 0.975    |
| MasterPlannerFlow  | start                        |        10 |      10 | 1        |
| TaskExecutorFlowV3 | TaskStage.CODING             |        42 |      42 | 1        |
| TaskExecutorFlowV3 | TaskStage.DEBUGGING          |        11 |      11 | 1        |
| TaskExecutorFlowV3 | TaskStage.PLANNING           |        55 |      55 | 1        |
| TaskExecutorFlowV3 | TaskStage.PREPARE_NEXT       |        12 |      12 | 1        |
| TaskExecutorFlowV3 | TaskStage.REASONING          |         6 |       6 | 1        |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_BELOW |        33 |      33 | 1        |

### 执行时长

| eval_type   |   duration_avg |   duration_std |   duration_min |   duration_max |
|:------------|---------------:|---------------:|---------------:|---------------:|
| STAGE       |        16.1172 |        16.8909 |        1.00558 |        89.1442 |
| FLOW        |        60.1578 |        40.9805 |        0       |       193.046  |
| NOTEBOOK    |       505.32   |        82.9813 |      350.897   |       638.146  |

### 生成质量

| flow               | score_type        |      avg |       std |   median |   lower |   upper |
|:-------------------|:------------------|---------:|----------:|---------:|--------:|--------:|
| MasterPlannerFlow  | correct_score     | 0.835    | 0.0579751 |     0.85 |  0.8125 |  0.85   |
| TaskExecutorFlowV3 | correct_score     | 0.862273 | 0.0548089 |     0.85 |  0.85   |  0.8625 |
| TaskExecutorFlowV3 | planning_score    | 0.818667 | 0.0826163 |     0.82 |  0.75   |  0.88   |
| TaskExecutorFlowV3 | reasoning_score   | 0.818    | 0.0922841 |     0.8  |  0.75   |  0.9    |
| TaskExecutorFlowV3 | coding_score      | 0.828    | 0.0706978 |     0.85 |  0.78   |  0.85   |
| TaskExecutorFlowV3 | important_score   | 0.778889 | 0.122219  |     0.8  |  0.7    |  0.85   |
| TaskExecutorFlowV3 | user_supply_score | 0.757333 | 0.136288  |     0.8  |  0.65   |  0.85   |

- 质量评分使用评估模型（qwen3-30b-a3b）对生成的结果自动评分得到，评分越高质量越好
  - correct: 正确性评估，生成的结果是否符合当前规划的要求
  - planning: 规划质量，规划的具体步骤是否符合全局目标
  - reasoning: 推理质量，推理的结果是否准确、合理
  - coding: 代码生成质量，代码是否符合规划的要求，是否无语法错误、逻辑错误、冗余等
  - important: 重要性质量，生成的结果是否充分的考虑了每个子任务生成的“Important infos”
  - user_supply: 用户补充质量，生成的结果是否充分的考虑了用户的补充信息
