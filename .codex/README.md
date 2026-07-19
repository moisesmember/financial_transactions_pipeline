# Kit Codex para replicar a pipeline em outro domínio

## Por que existem vários arquivos

O Codex carrega automaticamente `AGENTS.md` na raiz do repositório. A pasta
`.codex/` deste kit guarda o contexto variável e o guia de adaptação:

- `../AGENTS.md`: práticas duráveis de Python, ML e MLOps;
- `PROJECT_CONTEXT.md`: contrato preenchido desta pipeline de fraude;
- `PROJECT_CONTEXT.template.md`: questionário neutro para a nova pipeline;
- este arquivo: procedimento de reutilização.

Um arquivo chamado apenas `.codex` não é uma fonte padrão de instruções. Também não
foi criado `.codex/config.toml`, pois esse arquivo serve para configurações do agente
(modelo, sandbox, MCP etc.), não para registrar o desenho da pipeline. A descoberta de
instruções por `AGENTS.md` é descrita na
[documentação oficial do Codex](https://learn.chatgpt.com/docs/agent-configuration/agents-md).

## Reutilização em um novo repositório

### 1. Copie o kit

Copie o `AGENTS.md` e a pasta `.codex` para a raiz Git da nova pipeline. Não copie
`.env`, dados, artefatos, histórico de treino ou credenciais deste projeto.

PowerShell:

```powershell
Copy-Item "<pipeline-origem>\AGENTS.md" "<nova-pipeline>\AGENTS.md"
Copy-Item "<pipeline-origem>\.codex" "<nova-pipeline>\.codex" -Recurse
Set-Location "<nova-pipeline>"
Copy-Item ".codex\PROJECT_CONTEXT.template.md" ".codex\PROJECT_CONTEXT.md"
```

Bash:

```bash
cp <pipeline-origem>/AGENTS.md <nova-pipeline>/AGENTS.md
cp -R <pipeline-origem>/.codex <nova-pipeline>/.codex
cd <nova-pipeline>
cp .codex/PROJECT_CONTEXT.template.md .codex/PROJECT_CONTEXT.md
```

O último comando substitui o contexto de fraude copiado por uma cópia do template.

### 2. Preencha o contrato antes de gerar código

Edite `.codex/PROJECT_CONTEXT.md` e elimine todos os `<PREENCHER>`. As decisões mais
importantes são target, momento da predição, disponibilidade point-in-time, split,
métrica principal, custo de erro, política de threshold e gates de promoção.

Se ainda não souber uma decisão de negócio, mantenha-a em `Decisões em aberto`. Peça ao
Codex para implementar apenas o que não depende dela. É melhor deixar um gate pendente
do que inventar um número que pareça preciso.

### 3. Inicie uma nova sessão do Codex

O Codex monta a cadeia de instruções ao iniciar a execução. Abra uma nova sessão na raiz
do novo repositório depois de copiar ou alterar o `AGENTS.md`.

Use este prompt de bootstrap:

```text
Leia AGENTS.md e .codex/PROJECT_CONTEXT.md integralmente. Audite o contexto contra os
arquivos existentes e liste apenas contradições ou decisões realmente bloqueantes.
Depois proponha uma arquitetura incremental para a nova pipeline, com contratos de
dados, estratégia anti-leakage, testes, artefatos de governança e critérios de pronto.
Não implemente premissas de negócio que não estejam confirmadas. Após a auditoria,
implemente a primeira fatia vertical executável e valide-a com testes.
```

Para confirmar que as instruções foram encontradas:

```bash
codex --ask-for-approval never "Resuma as instruções ativas deste repositório."
```

### 4. Gere em fatias verificáveis

Uma sequência segura para uma pipeline nova é:

1. configuração tipada, logging, estrutura de pacotes e testes básicos;
2. contrato de dados, ingestão e validação de schema;
3. split representativo e testes de leakage;
4. transformer de features e baseline simples end-to-end;
5. métricas, threshold e avaliação intocada;
6. persistência, manifesto, model card e histórico de runs;
7. MLflow com run pai, tags, métricas, artefatos e signature do modelo;
8. Optuna com objective isolado, orçamento, pruning e estudo persistente;
9. AutoML como challenger sob o mesmo split, métrica e orçamento;
10. inferência batch/API, migrations e observabilidade;
11. gates de promoção, Model Registry e rollback.

Cada etapa deve terminar com código executável, testes e documentação compatível. Evite
pedir toda a plataforma em um único prompt sem validar os contratos intermediários.

### 5. Valide a adaptação

Antes de aceitar a nova pipeline, peça explicitamente:

```text
Revise a implementação contra AGENTS.md e .codex/PROJECT_CONTEXT.md. Procure leakage,
fit fora do treino, amostragem em avaliação, uso de teste/OOT na seleção, divergência
entre treino e inferência, segredos, artefatos não versionados e promoção sem aprovação.
Crie testes de regressão para cada falha confirmada e execute a suíte definida no
contexto. Informe também o que não pôde ser validado.
```

Para configurar as ferramentas de experimentação, use também:

```text
Implemente MLflow, Optuna e os backends AutoML definidos no PROJECT_CONTEXT.md. Crie um
run MLflow pai por treinamento, registre lineage e use runs filhos apenas para trials ou
backends quando configurado. Garanta que Optuna e AutoML nunca acessem teste/OOT durante
a seleção, respeitem orçamento e persistam estudos/leaderboards. Compare o vencedor em
partições intocadas, congele o threshold no split definido e bloqueie promoção
automática. Teste também tracking indisponível, trial falho e backend opcional ausente.
```

## O que adaptar e o que preservar

Adapte domínio, fontes, schemas, target, split, métricas, modelos, custos, storage,
infraestrutura, SLOs e comandos. Preserve as invariantes: raw imutável, validação antes
do treino, transformações ajustadas só no treino, prevenção de leakage, avaliação
intocada, paridade treino/inferência, runs reproduzíveis, histórico imutável e promoção
governada.

Não copie automaticamente nomes `fraud_*`, limites de recall, custos, features
geográficas, proporções de amostragem ou thresholds deste projeto. Eles são decisões do
contexto financeiro, não boas práticas universais.

## Referências das ferramentas

- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/): runs, experimentos,
  datasets, backend e artifact store.
- [MLflow model signatures](https://mlflow.org/docs/latest/ml/model/signatures/):
  contrato de entrada e saída do modelo registrado.
- [Optuna Study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html):
  estudos, trials, storage e otimização.
- [AutoGluon Tabular](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html):
  presets e limites de tempo/recursos.
- [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html): leaderboard,
  limites e reprodutibilidade.
- [FLAML AutoML](https://microsoft.github.io/FLAML/docs/reference/automl/automl/):
  métrica, estimadores e orçamento.

## Especializações opcionais

Se uma subárvore precisar de regras próprias, adicione um `AGENTS.md` ou
`AGENTS.override.md` dentro dela, por exemplo em `src/api/` para contratos de API ou em
`migrations/` para regras de banco. Mantenha o arquivo da raiz abaixo do limite padrão
de instruções e coloque detalhes especializados próximos do código correspondente.
