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

| eval_type   |   success |   total |     rate |
|:------------|----------:|--------:|---------:|
| NOTEBOOK    |        11 |      15 | 0.733333 |
| STAGE       |       390 |     417 | 0.935252 |
| FLOW        |        85 |      89 | 0.955056 |

- NOTEBOOK: 是否完成全局目标，不考虑完成质量，只要完成全局目标即可
- FLOW: 子任务是否成功执行，不考虑子任务完成质量，只要完成子任务是否完成所有步骤无异常即可
- STAGE: 子任务中的单个步骤是否成功执行，不考虑子任务完成质量，只要该步骤成功执行无异常即可

#### 分 Stage 执行成功率

| flow               | stage                        |   success |   total |     rate |
|:-------------------|:-----------------------------|----------:|--------:|---------:|
| TaskExecutorFlowV3 | TaskStage.EXECUTING          |        60 |      86 | 0.697674 |
| TaskExecutorFlowV3 | TaskStage.PLANNING           |        85 |      86 | 0.988372 |
| MasterPlannerFlow  | start                        |        15 |      15 | 1        |
| TaskExecutorFlowV3 | TaskStage.CODING             |        64 |      64 | 1        |
| TaskExecutorFlowV3 | TaskStage.DEBUGGING          |        22 |      22 | 1        |
| TaskExecutorFlowV3 | TaskStage.PREPARE_NEXT       |        17 |      17 | 1        |
| TaskExecutorFlowV3 | TaskStage.REASONING          |        10 |      10 | 1        |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_ABOVE |         2 |       2 | 1        |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_BELOW |        53 |      53 | 1        |
| TaskExecutorFlowV3 | TaskStage.SUMMARY            |        60 |      60 | 1        |
| TaskExecutorFlowV3 | planning_paused              |         2 |       2 | 1        |

### 执行时长

| eval_type   |   duration_avg |   duration_std |   duration_min |   duration_max |
|:------------|---------------:|---------------:|---------------:|---------------:|
| STAGE       |         27.806 |        25.523  |         1.0017 |        173.278 |
| FLOW        |        110.354 |        89.0118 |         0      |        608.013 |
| NOTEBOOK    |        929.158 |       181.709  |       691.275  |       1444.67  |

### 生成质量

| flow               | score_type        |      avg |       std |   median |   lower |   upper |
|:-------------------|:------------------|---------:|----------:|---------:|--------:|--------:|
| MasterPlannerFlow  | correct_score     | 0.823077 | 0.0483709 |     0.85 |    0.8  |    0.85 |
| TaskExecutorFlowV3 | correct_score     | 0.926563 | 0.0575207 |     0.95 |    0.95 |    0.95 |
| TaskExecutorFlowV3 | planning_score    | 0.836143 | 0.0618622 |     0.85 |    0.85 |    0.85 |
| TaskExecutorFlowV3 | reasoning_score   | 0.857246 | 0.0796478 |     0.9  |    0.85 |    0.9  |
| TaskExecutorFlowV3 | coding_score      | 0.728551 | 0.0731492 |     0.75 |    0.75 |    0.75 |
| TaskExecutorFlowV3 | important_score   | 0.88403  | 0.0464843 |     0.9  |    0.9  |    0.9  |
| TaskExecutorFlowV3 | user_supply_score | 0.788676 | 0.0645051 |     0.8  |    0.8  |    0.8  |

- 质量评分使用评估模型（qwen3-30b-a3b）对生成的结果自动评分得到，评分越高质量越好
  - correct: 正确性评估，生成的结果是否符合当前规划的要求
  - planning: 规划质量，规划的具体步骤是否符合全局目标
  - reasoning: 推理质量，推理的结果是否准确、合理
  - coding: 代码生成质量，代码是否符合规划的要求，是否无语法错误、逻辑错误、冗余等
  - important: 重要性质量，生成的结果是否充分的考虑了每个子任务生成的“Important infos”
  - user_supply: 用户补充质量，生成的结果是否充分的考虑了用户的补充信息
