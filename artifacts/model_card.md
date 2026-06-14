# Model Card: logistic_regression

- Run ID: `20260614T054157873737Z_logistic_regression_fa0dbb98`
- Objective: fraud-risk triage for financial transactions.
- Recommended use: prioritization and manual investigation.
- Not recommended: automatic blocking without additional controls.
- Dataset version: `be2887cff0ab0b3356894cde802c44cc62186e6ae07ad0d85e13a798b0b9c381`
- Feature set version: `v1`
- Code version: `0e622acbec676d9635caa5c1dc5aa629d34c96ba-dirty`
- Train rows: 361887
- Validation rows: 90471
- Test rows: 90472
- Out-of-time rows: 60315
- Selected threshold: 0.800000
- Threshold strategy: `business_cost`
- Leakage audit: `warning`
- Baseline decision: `reject`

## Metrics

| Split | PR-AUC | ROC-AUC | Precision | Recall | F-beta | Alert rate |
|---|---:|---:|---:|---:|---:|---:|
| validation | 0.179728 | 0.940423 | 0.053269 | 0.723684 | 0.205761 | 0.022825 |
| test | 0.001315 | 0.519165 | 0.000520 | 0.009174 | 0.002120 | 0.021244 |
| out_of_time | 0.005244 | 0.561154 | 0.003693 | 0.046296 | 0.013998 | 0.022449 |

## Warnings

- O threshold selecionado esta no limite da faixa analisada.
- O treino usa uma amostra limitada de transacoes ao longo do horizonte.
- Features geograficas dominam pelo menos duas das cinco maiores importancias.

## Decision

- Recall out-of-time abaixo do minimo operacional.
- PR-AUC de teste inferior ao baseline atual.
- Custo por registro superior ao limite do baseline.
- Threshold selecionado no limite da faixa analisada.
- Warnings da auditoria ainda nao possuem justificativa.

## Limitations

- Performance is dataset- and time-window-specific.
- Feature coefficients express association, not causality.
- Operational recall requires delayed fraud feedback.
