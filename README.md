# Detecção de Fraude em Transações Financeiras

Projeto Python modular para treino, avaliação, ajuste de threshold e serving de um modelo de detecção de fraude usando o dataset do Kaggle "Financial Transactions Dataset: Analytics".

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

Por padrão, o projeto usa arquivos locais em `data/raw` e salva artefatos em `artifacts`.

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

5. Treine lendo os dados do MinIO e salvando os artefatos também no bucket:

```bash
python main.py
```

Quando `STORAGE_BACKEND=minio`, a API e o serviço de predição tentam baixar `artifacts/fraud_pipeline.joblib` e `artifacts/model_metadata.joblib` do bucket se eles ainda não existirem localmente.

## PostgreSQL

O Docker Compose inclui um PostgreSQL preparado para armazenar futuramente os
metadados de experimentos do MLflow. Configure no `.env`:

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
pip install -r requirements.txt
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
- `run_metrics`: métricas consolidadas de validação e teste.
- `threshold_evaluations`: TP, FP, TN, FN e custo por threshold.
- `run_artifacts`: localização e hash dos artefatos.
- `baseline_promotions`: histórico de modelos promovidos.
- `fact_model_runs`: view fato com uma linha por treino, métricas de validação
  e teste, custos, auditoria, thresholds, artefatos, baseline e rankings.

Por padrão, a conexão utilizada pelas migrations é:

```text
postgresql+psycopg://mlflow:mlflow@localhost:5432/mlflow
```

Defina `DATABASE_URL` para sobrescrever a conexão.

### Persistência após o treino

Após salvar o histórico local, o pipeline tenta gravar automaticamente no
PostgreSQL:

- Uma linha em `training_runs`.
- Métricas de validação e teste em `run_metrics`.
- Toda a grade de thresholds em `threshold_evaluations`.
- Caminhos, tamanhos e hashes em `run_artifacts`.

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

- Split temporal em treino, validação e teste.
- Nenhum `fit` em validação ou teste.
- Nenhum ID cru usado como feature.
- Features históricas calculadas com `shift`, sem usar a própria transação nem dados futuros.
- Métrica principal orientada a fraude: PR-AUC, recall, precision, F1 e F-beta.
- Threshold escolhido na validação e aplicado no teste.
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

Instale as dependências e execute o projeto:

```bash
pip install -r requirements.txt
python main.py
pytest
uvicorn src.api.app:app --reload
```

Por padrão, `python main.py` limita a `500000` as transações usadas no merge e
no treino, evitando esgotar a memória nas etapas mais pesadas em ambientes
locais. Ajuste no `.env` conforme a RAM disponível:

```bash
TRAINING_MAX_ROWS=200000
```

Para tentar processar o dataset completo, sem limite:

```bash
TRAINING_MAX_ROWS=0
```

O dataset completo exige significativamente mais memória e pode não ser
adequado para treinamento local com scikit-learn.

## Avaliação, threshold e baseline

Cada treinamento gera:

- `artifacts/threshold_analysis.csv`: TP, FP, TN, FN, precision, recall,
  F-beta e custo para thresholds entre `0.08` e `0.30`, nos splits de validação
  e teste.
- `artifacts/leakage_audit.json`: checks temporais, features de risco e alerta
  para ROC-AUC anormalmente alta.
- `artifacts/model_metadata.joblib`: métricas e configuração operacional.
- `artifacts/history/<run_id>/`: cópia imutável dos metadados, auditoria,
  thresholds e pipeline daquela execução.
- `artifacts/history/runs.csv`: índice consolidado para comparar todos os
  treinamentos.

O threshold é escolhido somente na validação. O split de teste é usado para
confirmar o resultado e não deve ser usado para escolher o threshold.
Quando o menor custo estiver em `0.08` ou `0.30`, a auditoria recomenda ampliar
a faixa antes de aprovar o ponto operacional.

Configure os custos relativos no `.env`:

```bash
THRESHOLD_SELECTION_STRATEGY=business_cost
FALSE_POSITIVE_COST=1
FALSE_NEGATIVE_COST=25
THRESHOLD_ANALYSIS_START=0.08
THRESHOLD_ANALYSIS_STOP=0.30
THRESHOLD_ANALYSIS_STEP=0.01
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

Também é possível promover ao final do treino com `PROMOTE_BASELINE=true`.
Use `BASELINE_OVERWRITE=true` somente quando a substituição tiver sido aprovada.

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
pip install -r requirements.txt
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
