# Contexto do projeto atual

Este arquivo especializa o `AGENTS.md` para a pipeline deste repositório. Atualize-o
sempre que um contrato central mudar.

## Identidade e objetivo

- Projeto: detecção de fraude em transações financeiras.
- Tipo de problema: classificação binária supervisionada e desbalanceada.
- Unidade de predição: uma transação.
- Target canônico: `is_fraud`.
- Decisão operacional: score de fraude e classe binária derivada de threshold.
- Risco principal: falsos negativos têm custo configurável superior ao de falsos
  positivos; limites concretos são definidos no `.env`.

## Dados e validação

- Fontes raw: transações, cartões, usuários, MCC e labels de fraude.
- Chave supervisionada canônica: `transaction_id`.
- Labels inválidos bloqueiam o treino; transações sem label são removidas por `inner
  join` e nunca convertidas em negativas.
- A coluna temporal é resolvida entre os candidatos validados em
  `src/config/settings.py`.
- Split: temporal, sem shuffle, em treino, validação, teste e out-of-time.
- Proporções padrão: validação `0.15`, teste `0.15` e OOT configurável, padrão `0.10`.
- Amostragem: somente no treino, depois do merge supervisionado e do split. Preserva
  positivos e amostra negativos de forma estratificada por período quando habilitada.
- Validação e teste confirmam desempenho; OOT mede generalização futura. Teste e OOT
  não participam da seleção nem da escolha do threshold.

## Modelagem e decisão

- Pipeline sklearn completa: limpeza, engenharia de features, `ColumnTransformer` e
  estimador.
- Seleção padrão: Optuna com objetivo de robustez temporal; modo fixo permanece
  disponível.
- Modelos opcionais ficam em `requirements-models.txt`; benchmarks externos ficam em
  `requirements-benchmarks.txt` e não participam de promoção automática.
- Métricas centrais: PR-AUC, recall, precision, F1/F-beta, alert rate e custo de
  negócio. ROC-AUC é diagnóstica e também aciona auditoria de leakage.
- Threshold: selecionado apenas na validação por custo de negócio ou estratégia
  configurada; aplicado sem ajuste em teste, OOT e inferência.
- Reprodutibilidade: `random_state=42`; para comparações controladas, Optuna usa um
  único job por padrão.

## MLflow, Optuna e AutoML neste projeto

### MLflow

- Habilitação: `MLFLOW_TRACKING_ENABLED`.
- Tracking URI padrão: `http://localhost:5000`.
- Experimento padrão: `fraud-detection`.
- Backend do servidor local: PostgreSQL; artifact store: MinIO/S3.
- O run pai representa um treinamento governado. Configuração, métricas por split,
  artefatos de governança e, opcionalmente, o modelo sklearn são registrados nele.
- `MLFLOW_LOG_MODEL` controla o log do modelo e `MLFLOW_REGISTER_MODEL` controla a
  solicitação de registro. Registro não equivale a promoção e deve respeitar os gates.
- O tracking é resiliente: indisponibilidade do MLflow não pode apagar os artefatos
  locais nem impedir a persistência governada restante; deve ficar registrada como
  warning.

### Optuna

- Habilitação: `MODEL_SELECTION_ENGINE=optuna`; `fixed` mantém o caminho sem busca.
- Candidatos e orçamento são controlados por `OPTUNA_MODEL_CANDIDATES`,
  `OPTUNA_TRIALS`, `OPTUNA_TIMEOUT_SECONDS`, `OPTUNA_TRIALS_PER_MODEL` e
  `OPTUNA_TIMEOUT_PER_MODEL_SECONDS`.
- Objetivo padrão: `MODEL_SELECTION_OBJECTIVE=temporal_robustness`, usando apenas folds
  temporais internos e validação permitida. Teste e OOT permanecem intocados.
- Pruning é controlado por `OPTUNA_ENABLE_PRUNING`. `OPTUNA_N_JOBS=1` é o padrão para
  reduzir consumo e variabilidade.
- Cada run deve preservar `optuna_trials.csv`, `optuna_study.json` e o breakdown do
  objective. Trials falhos ou candidatos sem dependência instalada devem aparecer no
  resumo, não desaparecer silenciosamente.

### AutoML

- AutoGluon, H2O AutoML e FLAML são benchmarks opcionais e ficam desativados por padrão.
- A habilitação global usa `RUN_EXTERNAL_BENCHMARKS`; cada backend possui uma flag
  própria e a lista permitida fica em `EXTERNAL_BENCHMARK_BACKENDS`.
- Orçamento: `EXTERNAL_BENCHMARK_TIME_LIMIT_SECONDS` e
  `EXTERNAL_BENCHMARK_MAX_MODELS`.
- Os backends recebem as mesmas features e somente treino/validação. O threshold é
  selecionado na validação e aplicado sem ajuste em teste/OOT.
- Leaderboards e resumos devem ser gravados em
  `external_benchmark_results.csv` e `external_benchmark_summary.json`.
- Dependência ausente ou falha de backend gera `WARNING`/`unavailable` e não interrompe
  o treino governado principal. Nenhum backend pode promover automaticamente seu líder.

## Leakage e features

- IDs crus, PII, datetimes crus e campos pós-evento são removidos do modelo.
- `STRICT_LEAKAGE_PREVENTION=true` também remove snapshots cuja disponibilidade
  point-in-time não está comprovada.
- Features históricas devem usar somente informação anterior à própria transação.
- Ablation geográfica, walk-forward e estabilidade temporal podem bloquear promoção.

## Persistência e governança

- Object storage: adaptadores local e MinIO/S3 por `src/storage`.
- Tracking: MLflow opcional e PostgreSQL estruturado, com migrations Alembic.
- Artefatos principais: pipeline, metadados, métricas, threshold, auditorias, model
  card, relatório de revisão, manifesto e histórico imutável por `run_id`.
- Estados de decisão: `reject`, `candidate`, `pending_review` e `approved`.
- Promoção é uma operação explícita. Aprovação automática não substitui
  `HUMAN_APPROVAL_CONFIRMED=true` quando a política exigir confirmação humana.
- Baselines anteriores à correção de viés de amostragem não são comparáveis aos runs
  atuais sem ressalva explícita.

## Mapa do código

- Configuração: `src/config/settings.py` e `.env.example`.
- Ingestão: `src/ingestion` e scripts de importação.
- Carga, merge e split: `src/data`.
- Limpeza e features: `src/features`.
- Seleção, avaliação e governança: `src/models`.
- Orquestração: `src/pipelines/training_pipeline.py` e
  `src/pipelines/prediction_pipeline.py`.
- Persistência: `src/storage`, `migrations` e scripts operacionais.
- API: `src/api`.
- Testes: `tests`.

## Comandos de trabalho

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m scripts.migrate_database upgrade
docker compose up -d postgres minio mlflow
uvicorn src.api.app:app --reload
```

Fallback com ambiente ativado:

```bash
python -m pytest -q
python main.py
python -m scripts.migrate_database upgrade
```

Para mudanças pequenas, execute primeiro o arquivo de teste diretamente relacionado.
Treinos completos, benchmarks, serviços Docker e testes que exigem infraestrutura são
validações caras; execute-os quando o risco da mudança justificar e informe claramente
quando não forem executados.

## Arquivos que não devem ser tratados como fonte editável

- `.env`: pode conter segredos e valores locais.
- `data/raw`, `artifacts/history` e objetos remotos: são dados ou histórico imutável.
- relatórios HTML e notebooks gerados: são resultados, não a implementação canônica.
