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
