# Model Card: xgboost

- Run ID: `20260614T165819291939Z_xgboost_54b26577`
- Objective: fraud-risk triage for financial transactions.
- Recommended use: prioritization and manual investigation.
- Not recommended: automatic blocking without additional controls.
- Dataset version: `be2887cff0ab0b3356894cde802c44cc62186e6ae07ad0d85e13a798b0b9c381`
- Feature set version: `v1`
- Code version: `e091182c135888f7c01c29ff7ca835f1e9735612-dirty`
- Model selection engine: `optuna`
- Model selection trials: 15
- Train rows: 201117
- Validation rows: 50280
- Test rows: 50279
- Out-of-time rows: 33520
- Selected threshold: 0.100000
- Threshold strategy: `business_cost`
- Leakage audit: `warning`
- Baseline decision: `reject`

## Metrics

| Split | PR-AUC | ROC-AUC | Precision | Recall | F-beta | Alert rate |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.586165 | 0.933667 | 0.233333 | 0.717949 | 0.507246 | 0.004773 |
| test | 0.006956 | 0.654357 | 0.005882 | 0.018519 | 0.012953 | 0.003381 |
| out_of_time | 0.051128 | 0.778494 | 0.046296 | 0.090909 | 0.076220 | 0.003222 |

## Warnings

- O treino usa uma amostra limitada de transacoes ao longo do horizonte.
- Features geograficas dominam pelo menos duas das cinco maiores importancias.

## Decision

- Recall out-of-time abaixo do minimo operacional.
- PR-AUC de teste inferior ao baseline atual.
- Custo por registro superior ao limite do baseline.
- Warnings da auditoria ainda nao possuem justificativa.

## Limitations

- Performance is dataset- and time-window-specific.
- Feature coefficients express association, not causality.
- Operational recall requires delayed fraud feedback.
