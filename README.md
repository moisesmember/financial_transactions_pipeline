# Detecção de Fraude em Transações Financeiras

Projeto Python modular para treino, avaliação, ajuste de threshold e serving de um modelo de detecção de fraude usando o dataset do Kaggle "Financial Transactions Dataset: Analytics".

A seleção principal usa Optuna para comparar famílias e hiperparâmetros sklearn
somente com treino e validação temporal. AutoGluon, H2O AutoML e FLAML podem ser
executados como benchmarks externos nos mesmos splits e custos operacionais.

Fonte de referência: https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets

## Estrutura

```text
data/raw/                         # CSVs e JSONs originais do Kaggle
artifacts/                        # Pipeline treinada e metadados
src/config/settings.py            # Configurações centralizadas
src/data/                         # Carga, merge e split temporal
src/features/                     # Limpeza, engenharia e preprocessing
src/models/                       # Factory, treino, avaliação e threshold
src/pipelines/                    # Orquestração de treino e predição
src/storage/                      # Storage local e MinIO/S3
src/api/                          # FastAPI e schemas
tests/                            # Testes unitários
```

## Importação dos dados do Kaggle

O projeto baixa automaticamente o dataset pela CLI oficial do Kaggle e grava os
arquivos diretamente no storage configurado, local ou MinIO.

Os arquivos importados em `data/raw` não devem ser versionados no Git. Esse
diretório está no `.gitignore`; somente o código de importação e as instruções
para reproduzir o download fazem parte do repositório.

Gere um token em `https://www.kaggle.com/settings/api` e configure no `.env`:

```bash
KAGGLE_API_TOKEN=seu_token
KAGGLE_DATASET=computingvictor/transactions-fraud-datasets
KAGGLE_OVERWRITE=false
```

Com `STORAGE_BACKEND=local`, execute:

```bash
python -m scripts.import_kaggle_data
```

Com `STORAGE_BACKEND=minio`, inicie o MinIO e execute o mesmo comando:

```bash
docker compose up -d minio
python -m scripts.import_kaggle_data
```

Por padrão, arquivos existentes não são alterados. Para substituí-los apenas
nessa execução:

```bash
python -m scripts.import_kaggle_data --overwrite
```

Também é possível definir `KAGGLE_OVERWRITE=true`. O argumento
`--no-overwrite` desativa a substituição para uma execução.

Para importar automaticamente antes de cada treinamento, configure:

```bash
KAGGLE_AUTO_IMPORT=true
```

Quando todos os arquivos esperados já existem e `KAGGLE_OVERWRITE=false`, o
download é ignorado. Os arquivos esperados são:

- `transactions_data.csv`
- `cards_data.csv`
- `users_data.csv`
- `mcc_codes.json`
- `train_fraud_labels.json`

O código é tolerante a pequenas variações de nomes de colunas, mas espera uma coluna temporal nas transações e labels binárias de fraude.

## Storage local ou MinIO

O projeto usa o MinIO como armazenamento definitivo para dados raw, modelos,
históricos, baselines, auditorias e resultados dos benchmarks. Os diretórios
locais são usados somente como staging durante a execução e removidos após o
upload ser confirmado.

Para usar MinIO como object store:

1. Copie `.env.example` para `.env` ou exporte as variáveis no terminal.
2. Configure:

```bash
STORAGE_BACKEND=minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=fraud-detection
MINIO_SECURE=false
RAW_DATA_PREFIX=data/raw
ARTIFACTS_PREFIX=artifacts
KEEP_LOCAL_RAW_DATA=false
KEEP_LOCAL_ARTIFACTS=false
```

3. Suba o MinIO local:

```bash
docker compose up -d minio
```

Console web do MinIO:

```text
http://localhost:9001
```

4. Importe o dataset diretamente para o bucket:

```bash
python -m scripts.import_kaggle_data
```

Para migrar dados e artefatos que já existam localmente, execute uma vez:

```bash
python -m scripts.migrate_local_storage_to_minio
```

A limpeza local só ocorre após todos os objetos serem confirmados no bucket.
Se qualquer upload falhar, os arquivos locais são preservados.

5. Treine lendo os dados do MinIO e salvando os artefatos no bucket:

```bash
python main.py
```

Os objetos ficam organizados pelos prefixos:

- `data/raw/`: dataset original.
- `artifacts/`: artefatos da execução mais recente.
- `artifacts/history/<run_id>/`: histórico imutável de cada treino.
- `artifacts/baseline/`: baseline oficial.
- `artifacts/external_benchmarks/<run_id>/`: saídas de AutoGluon, H2O e FLAML.

Quando `KEEP_LOCAL_ARTIFACTS=false`, a API baixa o pipeline e os metadados,
carrega-os em memória e remove imediatamente as cópias locais.

## MLflow

O projeto pode registrar cada treinamento governado no MLflow, mantendo o
historico proprio em `artifacts/history` e, quando habilitado, tambem o schema
analitico `fraud_tracking` no PostgreSQL. A integracao e nao bloqueante: se o
servidor MLflow estiver indisponivel, o treino continua e os artefatos locais
sao preservados.

Variaveis principais:

```bash
MLFLOW_TRACKING_ENABLED=true
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=fraud-detection
MLFLOW_ARTIFACT_LOCATION=
MLFLOW_ARTIFACT_ROOT=s3://fraud-detection/mlflow
MLFLOW_LOG_MODEL=true
MLFLOW_REGISTER_MODEL=false
MLFLOW_REGISTERED_MODEL_NAME=fraud-detection-model
```

Suba o tracking server local com PostgreSQL como backend store e MinIO como
artifact store:

```bash
docker compose up -d postgres minio mlflow
```

UI do MLflow:

```text
http://localhost:5000
```

No Compose, `MLFLOW_ARTIFACT_ROOT=s3://fraud-detection/mlflow` fica no servidor
MLflow. Deixe `MLFLOW_ARTIFACT_LOCATION` vazio no cliente para usar o proxy de
artefatos do tracking server.

Ao final do treino, o MLflow recebe:

- parametros do modelo, selecao, threshold e versoes;
- metricas de validacao, teste, out-of-time, custos e volumes dos splits;
- tags com `run_id`, status, auditoria e decisao de baseline;
- todos os artefatos governados em `governance/`;
- opcionalmente o modelo sklearn em formato MLflow em `model/`.

Para usar apenas os artefatos/historico locais, configure:

```bash
MLFLOW_TRACKING_ENABLED=false
```

## PostgreSQL

O Docker Compose inclui um PostgreSQL preparado para armazenar futuramente os
metadados estruturados do projeto e o backend store do MLflow. Configure no
`.env`:

```bash
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_PORT=5432
```

Suba o banco:

```bash
docker compose up -d postgres
```

Confira o healthcheck:

```bash
docker compose ps postgres
```

String de conexão a partir da máquina local:

```text
postgresql://mlflow:mlflow@localhost:5432/mlflow
```

Dentro da rede do Docker Compose, use o host `postgres` no lugar de `localhost`.
Em ambientes compartilhados ou de produção, substitua a senha padrão.

### Migrations

As migrations usam Alembic e criam um schema isolado chamado
`fraud_tracking`, sem interferir nas tabelas internas que o MLflow poderá criar
no mesmo banco.

Instale as dependências e suba o PostgreSQL:

```bash
pip install -U -r requirements.txt
docker compose up -d postgres
```

Aplicar todas as migrations:

```bash
python -m scripts.migrate_database upgrade
```

Verificar a revisão aplicada:

```bash
python -m scripts.migrate_database current
```

Reverter a última migration:

```bash
python -m scripts.migrate_database downgrade
```

O schema contém:

- `training_runs`: configuração, período, volumes, auditoria e metadata completa.
- `run_metrics`: métricas de validação, teste e out-of-time.
- `threshold_evaluations`: TP, FP, TN, FN e custo por threshold/cenário.
- `run_artifacts`: localização e hash dos artefatos.
- `baseline_promotions`: histórico de modelos promovidos.
- `leakage_audit_checks`: checks individualizados e suas severidades.
- `model_features`: coeficientes/importâncias e classificação das features.
- `robustness_experiments`: resultados das ablações geográficas A-D.
- `model_search_trials`: trials, parâmetros e métricas da seleção Optuna.
- `external_benchmark_results`: leaderboard normalizado de AutoGluon, H2O,
  FLAML e do vencedor sklearn.
- `model_predictions`, `operational_feedback` e `drift_metrics`: base para
  monitoramento pós-produção.
- `fact_model_runs`: view fato com uma linha por treino, métricas de validação
  teste e OOT, custos, auditoria, versões, thresholds, baseline e rankings.

Por padrão, a conexão utilizada pelas migrations é:

```text
postgresql+psycopg://mlflow:mlflow@localhost:5432/mlflow
```

Defina `DATABASE_URL` para sobrescrever a conexão.

### Persistência após o treino

Após salvar o histórico local, o pipeline tenta gravar automaticamente no
PostgreSQL:

- Uma linha em `training_runs`.
- Métricas de validação, teste e out-of-time em `run_metrics`.
- Toda a grade de thresholds em `threshold_evaluations`.
- Caminhos, tamanhos e hashes em `run_artifacts`.
- Checks da auditoria e importâncias das features em tabelas próprias.

A operação é transacional e idempotente pelo `run_id`. Se o PostgreSQL estiver
indisponível ou as migrations ainda não tiverem sido aplicadas, o treinamento
não falha: os arquivos continuam preservados em `artifacts/history` e um aviso
é registrado no log.

Configuração:

```bash
DATABASE_TRACKING_ENABLED=true
DATABASE_CONNECT_TIMEOUT_SECONDS=3
DATABASE_URL=postgresql+psycopg://mlflow:mlflow@localhost:5432/mlflow
```

Use `DATABASE_TRACKING_ENABLED=false` para trabalhar somente com o histórico
local.

Consultar a view fato:

```sql
SELECT
    run_id,
    model_name,
    test_pr_auc,
    test_fbeta,
    test_business_cost,
    audit_status,
    is_active_baseline,
    test_pr_auc_rank
FROM fraud_tracking.fact_model_runs
ORDER BY test_pr_auc_rank;
```

As coleções completas de thresholds, artefatos e promoções ficam disponíveis
nas colunas JSONB `threshold_evaluations`, `artifacts` e
`baseline_promotions`.

## Princípios de modelagem

- Split temporal em treino, validação, teste e out-of-time.
- Nenhum `fit` em validação, teste ou out-of-time.
- Nenhum ID cru usado como feature.
- Features históricas calculadas com `shift`, sem usar a própria transação nem dados futuros.
- Métrica principal orientada a fraude: PR-AUC, recall, precision, F1 e F-beta.
- Threshold escolhido na validação e aplicado sem ajuste no teste e OOT.
- Pipeline completa salva com limpeza, engenharia de features, `ColumnTransformer` e modelo.

## Como executar

Crie e ative um ambiente virtual:

Windows PowerShell:

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências em ordem. Primeiro instale ou atualize a base do
projeto; esse arquivo contém os limites de versão compatíveis com o pipeline e
com os benchmarks:

```bash
pip install -U -r requirements.txt
```

Depois, instale os modelos opcionais usados pelo Optuna (`xgboost`, `lightgbm`
e `catboost`) quando quiser incluí-los na busca:

```bash
pip install -r requirements-models.txt
```

Por fim, instale os benchmarks externos (`AutoGluon`, `H2O` e `FLAML`). Deixe
esta etapa por último porque esses pacotes têm restrições próprias de versões:

```bash
pip install -U -r requirements-benchmarks.txt
```

Execute o projeto:

```bash
python main.py
pytest
uvicorn src.api.app:app --reload
```

AutoGluon e H2O são recomendados em Linux/WSL; H2O também requer uma JVM
compatível. Uma dependência ausente é registrada como `unavailable` e não
interrompe o treino principal.

Por padrão, o carregamento bruto é completo e o limite local é aplicado apenas
ao split de treino, depois do `inner join` com labels e do split temporal. Todos
os positivos conhecidos são preservados; somente negativos são amostrados de
forma estratificada por mês:

```bash
RAW_DATA_MAX_ROWS=0
TRAINING_MAX_ROWS=600000
PRESERVE_ALL_POSITIVES=true
NEGATIVE_SAMPLING_ENABLED=true
NEGATIVE_SAMPLING_STRATEGY=temporal_stratified
NEGATIVE_SAMPLING_BY=month
NEGATIVE_TO_POSITIVE_RATIO=100
```

Para treinar com o dataset supervisionado completo, sem amostragem:

```bash
RAW_DATA_MAX_ROWS=0
TRAINING_MAX_ROWS=0
```

`RAW_DATA_MAX_ROWS` também aceita um limite explícito para diagnóstico, mas uma
rodada assim não representa o range bruto completo. Validation, test e
out-of-time nunca são reduzidos por `TRAINING_MAX_ROWS`. Se o número de
positivos do treino for maior que o limite, a pipeline falha em vez de remover
fraudes conhecidas.

### Comparar a correção de amostragem

Use o mesmo código, seed e configuração de modelo em todas as novas rodadas.
A rodada A deve ser um `run_id` histórico produzido antes da correção; a
pipeline atual não oferece modo sequencial porque essa estratégia é inválida.

| Rodada | Configuração |
|---|---|
| A — legado sequencial | run histórico com o antigo corte cronológico |
| B — supervisionado completo | `TRAINING_MAX_ROWS=0` |
| C — positivos + negativos por mês | `TRAINING_MAX_ROWS=600000`, `NEGATIVE_TO_POSITIVE_RATIO=0` |
| D — mensal 1:100 | `TRAINING_MAX_ROWS=600000`, `NEGATIVE_TO_POSITIVE_RATIO=100` |
| E — mensal 1:200 | `TRAINING_MAX_ROWS=600000`, `NEGATIVE_TO_POSITIVE_RATIO=200` |

Em todas as rodadas novas mantenha `RAW_DATA_MAX_ROWS=0`,
`PRESERVE_ALL_POSITIVES=true`, `NEGATIVE_SAMPLING_ENABLED=true` e
`NEGATIVE_SAMPLING_BY=month`. Execute `python main.py` após cada alteração e
compare o histórico com:

```bash
python scripts/list_training_history.py --limit 10
```

`NEGATIVE_TO_POSITIVE_RATIO=0` desativa somente o teto por razão; o orçamento
de `TRAINING_MAX_ROWS` continua sendo distribuído entre os meses.

### Baseline após a correção e robustez temporal

As novas execuções são identificadas separadamente das rodadas que sofreram o
corte cronológico sequencial:

```env
BASELINE_NAME=baseline_after_sampling_fix
SAMPLING_FIX_APPLIED=true
PREVIOUS_BASELINES_INVALIDATED_BY_SAMPLING_BIAS=true
MODEL_SELECTION_OBJECTIVE=temporal_robustness
WALK_FORWARD_ENABLED=true
RUN_GEO_ABLATION=true
PROMOTE_BASELINE=false
HUMAN_APPROVAL_CONFIRMED=false
```

`baseline_after_sampling_fix` é um baseline analítico para comparação, não uma
aprovação de produção. Uma decisão `approved` exige todos os gates temporais e
`HUMAN_APPROVAL_CONFIRMED=true`. O Optuna usa folds internos e o último holdout
de seleção; test e out-of-time permanecem independentes para rejeição.

Além dos relatórios principais, cada execução gera:

- `performance_by_year.csv`, `performance_by_month.csv` e
  `performance_by_period.md`;
- `error_attribution_report.md` e `error_attribution_by_group.csv`;
- `feature_stability_by_period.csv`;
- `top_k_analysis.csv` e `top_k_analysis.md`;
- `baseline_after_sampling_fix.json`.

Top-K mede a capacidade operacional de investigação, mas nunca transforma
sozinho uma decisão `reject` em `candidate` ou `approved`.

## Avaliação, threshold e baseline

Cada treinamento gera:

- `artifacts/target_audit.json`, `target_audit.md`,
  `target_audit_by_split.csv` e `target_audit_by_period.csv`: auditoria de
  labels, cobertura temporal e duplicidades. Transacoes sem label sao removidas
  do treino supervisionado via `inner join`; nenhuma e convertida para classe 0,
  e labels invalidos geram erro antes do treino.
- `artifacts/sampling_audit.json`, `sampling_audit.md`,
  `sampling_by_period.csv`, `sampling_by_split.csv` e
  `sampling_positive_coverage.csv`: comprovam o estágio da amostragem, a
  cobertura positiva, o range temporal e que os splits de avaliação ficaram
  intactos.
- `artifacts/data_drift_report.json`, `data_drift_report.md`,
  `data_drift_numeric.csv` e `data_drift_categorical.csv`: comparação de
  distribuição entre treino, validação, teste e out-of-time, com PSI/KS quando
  aplicável.
- `artifacts/feature_stability_report.json` e `feature_stability_report.md`:
  ranking de PSI por feature/split, features com drift alto e recomendacao de
  manter, transformar ou remover em nova rodada.
- `artifacts/robustness_report.json`, `robustness_report.md`,
  `geo_ablation_report.json` e `geo_ablation_report.md`: experimentos de
  robustez e ablação geográfica, incluindo variantes sem coordenadas, sem
  cidade/estado, sem todas as features geográficas e baseline interpretável.
- `artifacts/walk_forward_report.json` e `walk_forward_report.md`: validação
  temporal expansiva quando `WALK_FORWARD_ENABLED=true`; caso contrário o
  relatório registra explicitamente `disabled`.
- `artifacts/model_review_report.md`: relatório consolidado para revisão humana
  com decisão, métricas, auditorias, drift, threshold e próximos passos.
- `artifacts/threshold_recommendations.json`: recomendações complementares de
  threshold, mantendo a seleção operacional baseada apenas na validação.
- `artifacts/optuna_trials.csv`: parâmetros, PR-AUC, threshold, custo e estado
  de todos os trials.
- `artifacts/optuna_study.json`: vencedor, parâmetros, seed e objetivo da busca.
- `artifacts/external_benchmark_results.csv`: comparação do vencedor sklearn
  com AutoGluon, H2O e FLAML em validação, teste e OOT.
- `artifacts/external_benchmark_summary.json`: status, duração e modelo líder
  de cada framework.
- `artifacts/threshold_analysis.csv`: grade principal entre `0.05` e `0.80`
  para validação, teste e OOT.
- `artifacts/threshold_cost_scenarios.csv`: melhores pontos para os cenários
  `1:10`, `1:25`, `1:50` e `5:25`.
- `artifacts/leakage_audit.json`: checks temporais, features de risco e alerta
  para ROC-AUC anormalmente alta.
- `artifacts/feature_importance.csv`: coeficientes, odds ratio, direção e grupo.
- `artifacts/calibration_report.csv`, `score_deciles.csv`,
  `calibration_metrics.json` e `calibration_curve.png`: calibração dos scores.
- `artifacts/out_of_time_metrics.json`: avaliação da janela futura intocada.
- `artifacts/model_card.md`: documentação automática da execução.
- `artifacts/baseline_decision.json`: decisao `approved`, `candidate`,
  `pending_review` ou `reject`, com os motivos.
- `artifacts/manifest.json`: hashes e tamanhos dos artefatos obrigatórios.
- `artifacts/model_metadata.joblib`: métricas e configuração operacional.
- `artifacts/history/<run_id>/`: cópia imutável dos metadados, auditoria,
  thresholds e pipeline daquela execução.
- `artifacts/history/runs.csv`: índice consolidado para comparar todos os
  treinamentos.

### Política de decisão

A decisão final agora é uma das opções:

- `reject`: falhou em gates bloqueantes, como recall out-of-time mínimo,
  queda relativa de PR-AUC/recall, custo contra baseline, falha crítica de
  auditoria ou evidências fortes de instabilidade.
- `candidate`: passou nos gates automáticos, mas ainda não foi solicitado
  promover para produção.
- `pending_review`: não há bloqueio automático, mas existem warnings sem
  justificativa, drift, auditoria pendente ou revisão humana obrigatória.
- `approved`: aprovado pela política e elegível para promoção quando
  `PROMOTE_BASELINE=true`.

A pipeline pode rejeitar automaticamente. Promoção para produção continua
dependendo de política explícita e não ocorre quando há warnings não
justificados, instabilidade temporal relevante ou queda ruim em teste/OOT.

O threshold é escolhido somente na validação. Teste e OOT confirmam desempenho,
estabilidade temporal, custo e capacidade operacional. Um threshold no limite
da grade bloqueia a promoção automática.

### Seleção de modelo com Optuna

Por padrão, Optuna compara `logistic_regression`, `random_forest` e
`hist_gradient_boosting`. Também há suporte opcional a `xgboost`, `lightgbm`
e `catboost`. Instale `requirements-models.txt`, conforme a sequência da seção
`Como executar`, antes de incluí-los na busca.

Com `OPTUNA_SELECTION_OBJECTIVE=temporal_stability`, o objetivo prioriza
robustez temporal: pior PR-AUC entre janelas, pior recall, penalidade por
instabilidade e penalidade quando a ultima janela desaba. Teste e OOT nunca
participam da busca.

```bash
MODEL_SELECTION_ENGINE=optuna
OPTUNA_MODEL_CANDIDATES=logistic_regression,random_forest,hist_gradient_boosting,lightgbm,catboost,xgboost
OPTUNA_TRIALS=15
OPTUNA_TIMEOUT_SECONDS=900
OPTUNA_N_JOBS=1
OPTUNA_SELECTION_OBJECTIVE=temporal_stability
OPTUNA_TEMPORAL_HOLDOUT_FRACTION=0.20
OPTUNA_PR_AUC_STABILITY_PENALTY=0.50
OPTUNA_RECALL_STABILITY_PENALTY=0.50
OPTUNA_LAST_WINDOW_PENALTY=1.00
```

Com `OPTUNA_SELECTION_OBJECTIVE=temporal_stability`, cada trial tambem e
avaliado em janelas temporais internas e na validacao oficial. O pior periodo
pesa mais do que a media, e a ultima janela recebe penalidade propria quando
recall ou PR-AUC caem. Isso reduz a chance de escolher um modelo que apenas
explora a validacao tradicional.

Use `OPTUNA_N_JOBS=1` para maior reprodutibilidade e controle de memória. Uma
dependência opcional ausente faz somente aquele candidato ser ignorado, com um
aviso nos logs e registro no resumo do estudo. Para treinar somente o modelo
configurado:

```bash
MODEL_SELECTION_ENGINE=fixed
MODEL_NAME=logistic_regression
```

### Benchmarks externos

```bash
RUN_EXTERNAL_BENCHMARKS=false
RUN_AUTOGLUON_BENCHMARK=false
RUN_H2O_BENCHMARK=false
RUN_FLAML_BENCHMARK=false
EXTERNAL_BENCHMARK_BACKENDS=autogluon,h2o,flaml
EXTERNAL_BENCHMARK_TIME_LIMIT_SECONDS=300
EXTERNAL_BENCHMARK_MAX_MODELS=10
```

Os benchmarks recebem as mesmas features governadas e usam somente
treino/validação para ajuste. O threshold é selecionado na validação e aplicado
em teste/OOT. Os resultados são comparativos: um benchmark externo não é
promovido automaticamente nem substitui `fraud_pipeline.joblib`.

Por padrão eles ficam desativados para preservar o treinamento principal. Para
rodar somente H2O, por exemplo:

```bash
RUN_EXTERNAL_BENCHMARKS=true
RUN_AUTOGLUON_BENCHMARK=false
RUN_H2O_BENCHMARK=true
RUN_FLAML_BENCHMARK=false
```

Falhas, dependências ausentes e interrupções de benchmarks externos são
registradas como `WARNING` e no `external_benchmark_summary.json`; os artefatos
governados, a persistência no PostgreSQL e a sincronização com MinIO continuam.

Configure os custos relativos no `.env`:

```bash
THRESHOLD_SELECTION_STRATEGY=business_cost
FALSE_POSITIVE_COST=1
FALSE_NEGATIVE_COST=25
THRESHOLD_ANALYSIS_START=0.05
THRESHOLD_ANALYSIS_STOP=0.80
THRESHOLD_ANALYSIS_STEP=0.01
THRESHOLD_COST_SCENARIOS=1:10,1:25,1:50,5:25
OUT_OF_TIME_SIZE=0.10
```

Os valores representam custos relativos e devem refletir o negócio. Por
exemplo, `FALSE_NEGATIVE_COST=25` considera uma fraude não detectada 25 vezes
mais custosa que uma análise desnecessária.

Para promover o modelo já treinado como baseline oficial:

```bash
python -m scripts.promote_baseline
```

O baseline é salvo em `artifacts/baseline` com hash SHA-256, métricas e relatórios.
Ele não é sobrescrito automaticamente. Para substituir deliberadamente:

```bash
python -m scripts.promote_baseline --overwrite
```

Para listar os melhores runs históricos por PR-AUC de teste:

```bash
python -m scripts.list_training_history --sort-by test_pr_auc --limit 10
```

Depois de escolher um `run_id`, promova exatamente aquela execução:

```bash
python -m scripts.promote_baseline --run-id RUN_ID --overwrite
```

Por padrão, cada run mantém uma cópia do pipeline para reprodução e promoção.
Use `TRAINING_HISTORY_SAVE_PIPELINE=false` apenas para economizar storage,
aceitando que o modelo histórico não poderá ser restaurado diretamente.

Também é possível solicitar promoção ao final do treino com
`PROMOTE_BASELINE=true`. A promoção só ocorre quando a política retorna
`promote`; `keep_candidate` e `reject` são bloqueados. Se o PostgreSQL falhar, a
alteração local é revertida para evitar registries divergentes.

Gates configuráveis:

```bash
PROMOTION_MIN_RECALL=0.90
PROMOTION_MAX_ALERT_RATE=0.025
PROMOTION_MAX_OOT_PR_AUC_DROP=0.15
PROMOTION_MAX_COST_INCREASE=0.05
PROMOTION_MIN_OOT_PR_AUC_LIFT=1.0
PROMOTION_MIN_WALK_FORWARD_RECALL=0.05
PROMOTION_MIN_WALK_FORWARD_PR_AUC_LIFT=1.0
PROMOTION_MAX_WALK_FORWARD_RECALL_DROP=0.50
FEATURE_STABILITY_PSI_THRESHOLD=0.25
BASELINE_WARNING_JUSTIFICATION=
```

Para executar as ablações geográficas e os experimentos de robustez, habilite
`RUN_GEO_ABLATION=true`. Esse modo treina variantes adicionais e aumenta
consideravelmente tempo e memória.

Para excluir colunas do modelo principal, use:

```bash
FEATURE_EXCLUSIONS=merchant_city,merchant_state,zip,latitude,longitude
```

Ou, para remover o conjunto geografico padrao:

```bash
EXCLUDE_GEOGRAPHIC_FEATURES=true
```

Deixe essas opcoes vazias/desativadas quando quiser que o modelo principal seja
o completo e que a comparacao sem geografia seja feita apenas pelos experimentos
de ablation.

Para executar validação temporal walk-forward:

```bash
WALK_FORWARD_ENABLED=true
WALK_FORWARD_FOLDS=4
```

Uma auditoria com status `warning` não comprova vazamento, mas exige revisão.
Neste dataset, atributos de snapshot como `card_on_dark_web`, `credit_score` e
`current_age`, além de campos como `errors`, precisam ser confirmados como
disponíveis no instante da transação.

Por padrão, `STRICT_LEAKAGE_PREVENTION=true` remove do modelo identificadores,
PII, atributos financeiros sem histórico temporal e campos potencialmente
posteriores à autorização. Desative apenas após comprovar a disponibilidade
dessas features no momento real da predição.

## Jupyter

Com o ambiente virtual ativado, instale as dependências do projeto:

```bash
pip install -U -r requirements.txt
python -m pip install -U -r requirements.txt
```

Se o ambiente já tiver o pacote descontinuado `ydata-profiling`, substitua-o
pelo `fg-data-profiling` e instale o suporte a widgets do Jupyter:

```bash
pip uninstall ydata-profiling
pip install ipywidgets
pip install fg-data-profiling
```

Para gerar relatórios de perfilamento, use o novo namespace do pacote:

```python
from data_profiling import ProfileReport
```

Inicie o JupyterLab na raiz do projeto:

```bash
python -m jupyter lab
```

O navegador abrirá a interface do JupyterLab. Ao criar ou abrir um notebook,
selecione o kernel Python do ambiente virtual `.venv`.

Para encerrar o servidor, pressione `Ctrl+C` no terminal.

## Predição em batch

```python
from src.config.settings import Settings
from src.pipelines.prediction_pipeline import FraudPredictionService

service = FraudPredictionService(Settings())
predictions = service.predict_csv("data/raw/transactions_data.csv")
print(predictions.head())
```

## API

Depois de treinar:

```bash
uvicorn src.api.app:app --reload
```

A documentação Swagger fica disponível em `http://localhost:8000/docs`.

Para iniciar um treinamento governado pela API:

```bash
curl -X POST "http://localhost:8000/training-runs" \
  -H "Content-Type: application/json" \
  -d '{
    "THRESHOLD_SELECTION_STRATEGY": "business_cost",
    "THRESHOLD_ANALYSIS_START": 0.05,
    "THRESHOLD_ANALYSIS_STOP": 0.80,
    "THRESHOLD_ANALYSIS_STEP": 0.01,
    "FALSE_POSITIVE_COST": 1,
    "FALSE_NEGATIVE_COST": 25,
    "THRESHOLD_COST_SCENARIOS": "1:10,1:25,1:50,5:25",
    "OUT_OF_TIME_SIZE": 0.10,
    "LEAKAGE_ROC_AUC_WARNING": 0.99,
    "STRICT_LEAKAGE_PREVENTION": true,
    "PROMOTE_BASELINE": false,
    "BASELINE_OVERWRITE": false,
    "RUN_GEO_ABLATION": false,
    "TRAINING_HISTORY_SAVE_PIPELINE": true,
    "RAW_DATA_MAX_ROWS": 0,
    "TRAINING_MAX_ROWS": 600000,
    "PRESERVE_ALL_POSITIVES": true,
    "NEGATIVE_SAMPLING_ENABLED": true,
    "NEGATIVE_SAMPLING_STRATEGY": "temporal_stratified",
    "NEGATIVE_SAMPLING_BY": "month",
    "NEGATIVE_TO_POSITIVE_RATIO": 100,
    "BASELINE_WARNING_JUSTIFICATION": "",
    "PROMOTION_MIN_RECALL": 0.90,
    "PROMOTION_MAX_ALERT_RATE": 0.025,
    "PROMOTION_MAX_OOT_PR_AUC_DROP": 0.15,
    "PROMOTION_MAX_COST_INCREASE": 0.05
  }'
```

A rota retorna `202 Accepted` e um `job_id`. Consulte o andamento:

```bash
curl "http://localhost:8000/training-runs/JOB_ID"
```

Todos os campos do body são opcionais. Campos ausentes ou `null` mantêm os
valores resolvidos do `.env`; `TRAINING_MAX_ROWS=0` processa o dataset completo.
Quando existe limite, ele e aplicado somente depois do inner join e do split
temporal: todos os positivos do treino sao preservados e apenas negativos sao
amostrados por mes. Validacao, teste e out-of-time nao sao reduzidos. Use
`NEGATIVE_TO_POSITIVE_RATIO=100` para controlar a razão máxima global de
negativos por positivo; o orçamento resultante é distribuído entre os períodos.
Se os positivos sozinhos excederem `TRAINING_MAX_ROWS`, a execução falha e
nenhum positivo é removido.
Os nomes também podem ser enviados em `snake_case`. Apenas um treinamento pode
executar por processo da API. Cada job usa staging isolado em
`.runtime/training/<job_id>` para não disputar arquivos com a inferência. Em
produção, proteja essa rota com autenticação e execute a API com um único worker
ou substitua o gerenciador em memória por uma fila distribuída.

Para consultar um relatório transparente de um treinamento concluído:

```bash
curl "http://localhost:8000/training-runs/RUN_ID/report?feature_limit=50"
```

O report consolida:

- resumo executivo e decisão de baseline;
- modelo selecionado, parâmetros, ranking e famílias consideradas;
- tamanho e taxa positiva dos splits temporais;
- precision, recall, F1, F-beta, PR-AUC, ROC-AUC, matriz de confusão,
  alert rate e custo de negócio;
- degradação entre validação, teste e out-of-time;
- features utilizadas, pesos/importâncias, direção e grupo;
- features desconsideradas, política de exclusão e colunas de risco;
- checks, warnings, falhas e recomendações da auditoria de leakage;
- trials do Optuna, benchmarks externos e experimentos de robustez;
- análise de thresholds, artefatos e histórico de promoção.

Use `feature_limit` entre 1 e 500 para controlar quantas features ordenadas por
importância são retornadas. A importância é interna ao modelo e não representa
causalidade. Runs antigos podem não possuir a lista exata de colunas excluídas;
nesse caso, a resposta marca `legacy_run_policy_only` e apresenta a política que
foi aplicada.

Para exportar em JSON todas as colunas da view
`fraud_tracking.fact_model_runs`:

```bash
curl "http://localhost:8000/model-runs/export?limit=100&offset=0"
```

A resposta contém `total`, `count`, os dados de paginação e `items`. Cada item
representa um treino e preserva os blocos JSON de thresholds, artefatos,
auditoria e promoções de baseline. O limite máximo por requisição é 1000.

Exemplo de request:

```json
{
  "records": [
    {
      "date": "2019-01-01 10:30:00",
      "amount": "$42.10",
      "card_id": 123,
      "client_id": 10,
      "merchant_id": 999,
      "merchant_city": "Austin",
      "merchant_state": "TX",
      "mcc": 5812
    }
  ]
}
```

## Selecao temporal com elegibilidade (v2)

A selecao nao interpreta estabilidade como qualidade. A formula anterior
somava componentes de recall/PR-AUC e uma penalidade pequena de dispersao; por
isso um modelo com desempenho futuro uniformemente ruim ainda podia obter uma
pontuacao alta. A formula v2 executa duas etapas:

1. aplica gates minimos ao numero de folds validos, pior/ultimo recall, lift de
   PR-AUC sobre a prevalencia de cada fold, alert rate e quedas temporais;
2. somente entre trials elegiveis calcula
   `quality - instability - alert_penalty - cost_penalty`.

O termo de qualidade e:

```text
0.30 * normalized_min_fold_pr_auc_lift
+ 0.20 * min_fold_recall
+ 0.15 * last_fold_recall
+ 0.15 * normalized_last_fold_pr_auc_lift
+ 0.10 * median_fold_recall
+ 0.10 * normalized_median_fold_pr_auc_lift
```

O lift usa `fold_pr_auc / fold_positive_rate`. Trials inelegiveis recebem o
piso `-1e9` e nao podem ser vencedores de familia. Os valores abaixo sao
criterios experimentais iniciais, nao uma politica definitiva de producao:

```dotenv
OPTUNA_TRIALS_PER_MODEL=10
OPTUNA_TIMEOUT_PER_MODEL_SECONDS=3600
OPTUNA_ENABLE_PRUNING=true
MIN_VALID_TEMPORAL_FOLDS=3
MIN_FOLD_RECALL_CANDIDATE=0.05
MIN_LAST_FOLD_RECALL_CANDIDATE=0.05
MIN_PR_AUC_LIFT_OVER_RANDOM=1.25
MAX_TEMPORAL_ALERT_RATE=0.025
MAX_PR_AUC_TEMPORAL_DROP=0.80
MAX_RECALL_TEMPORAL_DROP=0.80
SAMPLING_ENFORCE_FIXED_RATIO=true
NEGATIVE_TO_POSITIVE_RATIO=50
IMBALANCE_STRATEGY=negative_sampling
THRESHOLD_MIN=0.01
THRESHOLD_MAX=0.99
THRESHOLD_STEP=0.01
```

`IMBALANCE_STRATEGY` aceita `negative_sampling`, `class_weight`,
`sample_weight` e `negative_sampling_plus_class_weight`. A opcao padrao evita
aplicar class weight junto com negative sampling sem um experimento explicito.
Razoes 25:1, 50:1, 100:1 e dataset completo devem ser comparadas apenas nos
folds internos.

Ao executar `python main.py`, sao criados estudos independentes
`fraud_logistic_regression_search`, `fraud_hist_gradient_boosting_search` e
`fraud_random_forest_search`. O melhor trial elegivel de cada familia e
comparado nos mesmos folds; test e out-of-time sao usados uma unica vez depois
da escolha. O threshold tambem e escolhido depois do modelo, somente na
validacao, e uma escolha em `0.01` ou `0.99` gera warning explicito.

Meses sem fraude recebem `evaluation_status=no_positive_class`, com `recall`,
`pr_auc` e `fbeta` nulos. Meses somente com fraude recebem
`evaluation_status=no_negative_class` e `roc_auc` nulo. Esses valores nao
entram em medias nem em premios de estabilidade; contagens de folds validos e
single-class ficam nos relatorios.

Use `objective_score_breakdown.csv/.md` para reproduzir cada score e
`optuna_study.json` para comparar os vencedores. O arquivo
`baseline_challenger_comparison.json/.md` registra retrospectivamente o
HistGradientBoosting como `best_rejected_baseline` e uma regressao logistica
pior como `rejected_challenger`; um challenger pior em test ou OOT nunca
substitui esse baseline. Essa comparacao final nao retroalimenta o Optuna.
