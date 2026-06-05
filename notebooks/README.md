# Notebooks

Notebooks organizados para aprendizado, inspeção do dataset e avaliação do modelo.

Ordem recomendada:

1. `01_data_understanding.ipynb`: leitura, merge, limpeza inicial e qualidade dos dados.
2. `02_features_and_split.ipynb`: split temporal, engenharia de features e validação contra vazamento.
3. `03_training_evaluation.ipynb`: treino, comparação de modelos, métricas, curvas e threshold.
4. `04_prediction_workflow.ipynb`: predição batch, inspeção de outputs e teste simples do contrato da API.

Execute o Jupyter a partir da raiz do projeto:

```bash
jupyter lab
```

Os notebooks assumem que os arquivos do Kaggle estão em `data/raw`.
