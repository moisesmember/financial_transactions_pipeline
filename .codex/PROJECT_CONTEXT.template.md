# Contexto da nova pipeline

Substitua todos os marcadores `<PREENCHER>` antes de pedir a geração integral da
pipeline. Use `não aplicável` quando uma seção realmente não fizer parte do produto.
Não copie premissas de fraude para outro domínio sem validação.

## Identidade e objetivo

- Projeto: `<PREENCHER>`
- Problema de negócio: `<PREENCHER>`
- Tipo de ML: `<classificação binária/multiclasse, regressão, ranking, forecasting...>`
- Unidade de predição: `<PREENCHER>`
- Target e significado: `<PREENCHER>`
- Momento em que a predição ocorre: `<PREENCHER>`
- Consumidor da predição: `<PREENCHER>`
- Ação tomada a partir do resultado: `<PREENCHER>`
- Custos de erro ou restrições operacionais: `<PREENCHER>`

## Contrato dos dados

- Fontes: `<PREENCHER>`
- Formatos e localização: `<PREENCHER>`
- Chave da entidade/evento: `<PREENCHER>`
- Coluna temporal e timezone: `<PREENCHER>`
- Como o label é obtido e em quanto tempo amadurece: `<PREENCHER>`
- Colunas obrigatórias e tipos: `<PREENCHER>`
- Política de duplicidade: `<PREENCHER>`
- Política para target ausente/ambíguo: `<PREENCHER>`
- PII ou dados sensíveis: `<PREENCHER>`
- Volume atual, crescimento e restrição de memória: `<PREENCHER>`
- Identificação/versionamento do dataset: `<PREENCHER>`

## Disponibilidade point-in-time

- Features disponíveis no instante da predição: `<PREENCHER>`
- Features conhecidas somente depois do evento: `<PREENCHER>`
- Agregações históricas e janela: `<PREENCHER>`
- Entidades que não podem cruzar partições: `<PREENCHER>`
- Hipóteses que precisam de auditoria de leakage: `<PREENCHER>`

## Split e validação

- Estratégia: `<temporal, group, stratified ou combinação>`
- Justificativa em relação ao uso real: `<PREENCHER>`
- Partições e proporções/janelas: `<PREENCHER>`
- Out-of-time ou backtesting: `<PREENCHER>`
- Cross-validation interna: `<PREENCHER>`
- Política de amostragem ou pesos, aplicada em qual partição: `<PREENCHER>`
- Seed e tolerância a não determinismo: `<PREENCHER>`

## Modelagem e avaliação

- Baseline simples obrigatório: `<PREENCHER>`
- Famílias de modelos permitidas: `<PREENCHER>`
- Métrica principal de seleção: `<PREENCHER>`
- Métricas de confirmação: `<PREENCHER>`
- Métricas operacionais/custo: `<PREENCHER>`
- Segmentos ou grupos para avaliação: `<PREENCHER>`
- Necessidade de calibração: `<PREENCHER>`
- Política de threshold ou decisão: `<PREENCHER>`
- Restrições de latência, tamanho e explicabilidade: `<PREENCHER>`

## MLflow e rastreamento de experimentos

- MLflow é obrigatório, opcional ou não aplicável: `<PREENCHER>`
- Tracking URI por ambiente: `<PREENCHER>`
- Nome/padrão de experimentos: `<PREENCHER>`
- Backend store e artifact store: `<PREENCHER>`
- Convenção de run pai e runs filhos: `<PREENCHER>`
- Tags obrigatórias: `<dataset_version, code_version, feature_set_version...>`
- Parâmetros e métricas obrigatórios por split: `<PREENCHER>`
- Artefatos a enviar versus registrar somente por URI/hash: `<PREENCHER>`
- Autologging permitido e para quais bibliotecas: `<PREENCHER>`
- Log de modelo, signature e input example sanitizado: `<PREENCHER>`
- Model Registry, aliases/estágios e responsável pela promoção: `<PREENCHER>`
- Política quando o tracking server estiver indisponível: `<falhar ou continuar>`
- Retenção, acesso e tratamento de PII/secrets: `<PREENCHER>`

## Optuna e busca de hiperparâmetros

- Optuna habilitado e caminho alternativo sem busca: `<PREENCHER>`
- Objective, direção e fórmula de penalidades: `<PREENCHER>`
- Partições/folds permitidos no objective: `<PREENCHER>`
- Sampler, pruner e seed: `<PREENCHER>`
- Espaço de busca por família de modelo: `<PREENCHER>`
- Orçamento de desenvolvimento (`n_trials`, timeout, jobs): `<PREENCHER>`
- Orçamento oficial (`n_trials`, timeout, jobs): `<PREENCHER>`
- Storage do estudo, `study_name` e política de retomada: `<PREENCHER>`
- Métricas intermediárias válidas para pruning: `<PREENCHER>`
- Integração com MLflow e granularidade de runs filhos: `<PREENCHER>`
- Artefatos do estudo e campos obrigatórios por trial: `<PREENCHER>`
- Política de refit do vencedor: `<PREENCHER>`
- Tratamento de trial falho e dependência opcional ausente: `<PREENCHER>`

## AutoML e challengers

- AutoML habilitado e finalidade: `<benchmark, seleção ou não aplicável>`
- Frameworks e versões permitidas: `<AutoGluon, H2O, FLAML, outro>`
- Métrica explícita de leaderboard: `<PREENCHER>`
- Como fornecer split temporal/group sem divisão aleatória interna: `<PREENCHER>`
- Limite por backend de tempo, modelos/iterações, CPU, GPU, memória e disco:
  `<PREENCHER>`
- Presets/algoritmos permitidos e proibidos: `<PREENCHER>`
- Diretório isolado de trabalho e retenção: `<PREENCHER>`
- Política de ensembles e calibração interna: `<PREENCHER>`
- Avaliação externa do líder em teste/OOT: `<PREENCHER>`
- Threshold governado externamente ou pelo framework: `<PREENCHER>`
- Artefatos: leaderboard, configuração, versões e modelo exportado: `<PREENCHER>`
- Critérios de portabilidade, latência, tamanho e explicabilidade: `<PREENCHER>`
- Política de falha ou backend indisponível: `<PREENCHER>`
- Confirmação de que AutoML não promove automaticamente: `<PREENCHER>`

## Governança e promoção

- Artefatos obrigatórios por run: `<PREENCHER>`
- Tracking de experimentos alternativo ao MLflow, se houver: `<PREENCHER>`
- Object storage: `<local/S3/MinIO/outro>`
- Banco de metadados: `<PREENCHER>`
- Gates automáticos de rejeição: `<PREENCHER>`
- Critérios para candidato e aprovação: `<PREENCHER>`
- Aprovação humana necessária: `<sim/não e responsável>`
- Estratégia de baseline, registry e rollback: `<PREENCHER>`
- Retenção e auditoria: `<PREENCHER>`

## Inferência e operação

- Modo: `<batch, API, streaming ou combinação>`
- Contrato de entrada: `<PREENCHER>`
- Contrato de saída: `<PREENCHER>`
- Frequência/latência/SLA: `<PREENCHER>`
- Tratamento de schema inválido e artefato indisponível: `<PREENCHER>`
- Monitoramento de dados, drift e performance: `<PREENCHER>`
- Feedback/labels em produção: `<PREENCHER>`

## Stack e estrutura

- Versão de Python: `<PREENCHER>`
- Bibliotecas aprovadas e limites de versão: `<PREENCHER>`
- Serviços de infraestrutura: `<PREENCHER>`
- Estrutura de módulos desejada: `<PREENCHER ou usar a sugerida em AGENTS.md>`
- Estratégia de configuração e segredos: `<PREENCHER>`
- Ambientes suportados: `<Windows/Linux/containers/cloud>`

## Comandos verificáveis

```bash
# criar ambiente
<PREENCHER>

# instalar dependências
<PREENCHER>

# testes focados e suíte completa
<PREENCHER>

# lint, type-check e formatação
<PREENCHER>

# executar treino
<PREENCHER>

# executar inferência/API
<PREENCHER>

# migrations/infra, se aplicável
<PREENCHER>
```

## Critérios de aceite iniciais

- `<PREENCHER: comportamento funcional>`
- `<PREENCHER: qualidade mínima validada fora do treino>`
- `<PREENCHER: governança e reprodutibilidade>`
- `<PREENCHER: operação e documentação>`

## Decisões em aberto

- `<PREENCHER ou nenhuma>`
