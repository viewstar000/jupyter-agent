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
| NOTEBOOK  |       8 |    15 | 0.533333 |
| STAGE     |     382 |   415 | 0.920482 |
| FLOW      |      83 |    90 | 0.922222 |

- NOTEBOOK: 是否完成全局目标，不考虑完成质量，只要完成全局目标即可
- FLOW: 子任务是否成功执行，不考虑子任务完成质量，只要完成子任务是否完成所有步骤无异常即可
- STAGE: 子任务中的单个步骤是否成功执行，不考虑子任务完成质量，只要该步骤成功执行无异常即可

#### 分 Stage 执行成功率

| flow               | stage                        | success | total |     rate |
| :----------------- | :--------------------------- | ------: | ----: | -------: |
| TaskExecutorFlowV3 | TaskStage.EXECUTING          |      59 |    87 | 0.678161 |
| TaskExecutorFlowV3 | TaskStage.PLANNING           |      83 |    87 | 0.954023 |
| TaskExecutorFlowV3 | TaskStage.SUMMARY            |      59 |    60 | 0.983333 |
| MasterPlannerFlow  | start                        |      15 |    15 |        1 |
| TaskExecutorFlowV3 | TaskStage.CODING             |      66 |    66 |        1 |
| TaskExecutorFlowV3 | TaskStage.DEBUGGING          |      21 |    21 |        1 |
| TaskExecutorFlowV3 | TaskStage.PREPARE_NEXT       |      28 |    28 |        1 |
| TaskExecutorFlowV3 | TaskStage.REASONING          |       9 |     9 |        1 |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_ABOVE |       1 |     1 |        1 |
| TaskExecutorFlowV3 | TaskStage.REQUEST_INFO_BELOW |      40 |    40 |        1 |
| TaskExecutorFlowV3 | planning_paused              |       1 |     1 |        1 |

### 执行时长

| eval_type | duration_avg | duration_std | duration_min | duration_max |
| :-------- | -----------: | -----------: | -----------: | -----------: |
| STAGE     |      28.1425 |       27.442 |      1.00412 |      167.552 |
| FLOW      |      102.972 |       76.965 |            0 |      373.909 |
| NOTEBOOK  |      935.799 |       192.55 |       636.48 |      1223.69 |

### 生成质量

| flow               | score_type  |      avg |       std |
| :----------------- | :---------- | -------: | --------: |
| MasterPlannerFlow  | correct     | 0.854545 | 0.0350325 |
| TaskExecutorFlowV3 | correct     | 0.921061 | 0.0773629 |
| TaskExecutorFlowV3 | planning    | 0.836618 | 0.0582739 |
| TaskExecutorFlowV3 | reasoning   | 0.874627 | 0.0624822 |
| TaskExecutorFlowV3 | coding      | 0.739394 | 0.0410208 |
| TaskExecutorFlowV3 | important   | 0.881667 | 0.0611325 |
| TaskExecutorFlowV3 | user_supply | 0.784478 | 0.0558212 |

- 质量评分使用评估模型（qwen3-30b-a3b）对生成的结果自动评分得到，评分越高质量越好
  - correct: 正确性评估，生成的结果是否符合当前规划的要求
  - planning: 规划质量，规划的具体步骤是否符合全局目标
  - reasoning: 推理质量，推理的结果是否准确、合理
  - coding: 代码生成质量，代码是否符合规划的要求，是否无语法错误、逻辑错误、冗余等
  - important: 重要性质量，生成的结果是否充分的考虑了每个子任务生成的“Important infos”
  - user_supply: 用户补充质量，生成的结果是否充分的考虑了用户的补充信息
