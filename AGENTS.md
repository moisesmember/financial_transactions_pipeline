# Instruções para desenvolvimento de pipelines de Machine Learning

## Escopo e fonte de verdade

Estas instruções valem para todo o repositório. Antes de alterar código, leia
`.codex/PROJECT_CONTEXT.md`. Esse arquivo define o problema de negócio, o contrato dos
dados, a estratégia de validação, as métricas e os comandos concretos deste projeto.

Se o contexto estiver ausente ou contiver marcadores como `<PREENCHER>`:

1. inspecione o repositório, o README, os arquivos de configuração e os testes;
2. preencha apenas fatos verificáveis;
3. mantenha explícitas as decisões de negócio que ainda dependem do usuário;
4. não invente target, janela temporal, custos, SLOs ou critérios de promoção.

A ordem de precedência é: pedido atual do usuário, contratos e testes executáveis,
`.codex/PROJECT_CONTEXT.md`, este arquivo e, por último, documentação desatualizada.
Quando houver divergência, corrija a documentação ou registre claramente a pendência.

## Objetivo de engenharia

Construa uma pipeline reproduzível, testável e governada, na qual o mesmo caminho de
transformação seja usado no treino e na inferência. Prefira componentes pequenos,
contratos explícitos e decisões verificáveis a notebooks ou scripts monolíticos.

O fluxo de referência é:

`fonte -> ingestão -> validação/merge -> split -> amostragem de treino -> features -> seleção -> threshold -> avaliação intocada -> governança -> registro -> inferência`

Adapte nomes e módulos ao domínio, mas preserve a separação de responsabilidades.

## Forma de trabalhar

Antes de implementar uma mudança relevante:

- localize o contrato afetado e os testes correspondentes;
- identifique risco de leakage, alteração de schema, incompatibilidade de artefato e
  impacto na inferência;
- faça a menor mudança coerente e não reescreva módulos sem necessidade;
- preserve alterações existentes do usuário e não edite dados brutos ou artefatos
  históricos para fazer testes passarem;
- use configuração centralizada e validada; não espalhe constantes de negócio pelo
  código;
- adicione dependências somente quando o ganho justificar custo, segurança e
  reprodutibilidade.

Implemente em fatias verticais: contrato, código, testes, telemetria e documentação.
Execute primeiro testes focados e depois a suíte definida no contexto do projeto.

## Arquitetura esperada

Mantenha estas fronteiras, ainda que os nomes das pastas variem:

- `config`: configuração tipada, defaults seguros e validação entre campos;
- `ingestion`: fontes externas e staging isolado;
- `data`: carga, schema, deduplicação, merge e split;
- `features`: limpeza e transformações compatíveis com `fit`/`transform`;
- `models`: estimadores, busca, métricas, threshold, auditorias e governança;
- `pipelines`: orquestração de treino e inferência, sem concentrar toda a lógica;
- `storage`: interfaces e adaptadores para objetos e metadados;
- `api`: schemas e serviços; não replique regras de modelagem no controller;
- `tests`: contratos, casos de borda, regressões e integração;
- `scripts`: operações explícitas, idempotentes e documentadas;
- `notebooks`: exploração; nunca a única implementação de lógica de produção.

Use abstrações nas fronteiras externas. Código de domínio não deve depender diretamente
de Kaggle, S3/MinIO, banco, MLflow ou framework web. Adaptadores podem falhar de forma
controlada quando forem opcionais; falhas em contratos centrais devem interromper o run.

## Contratos de dados

- Trate dados raw como imutáveis. Downloads e conversões usam staging isolado e
  publicação explícita.
- Valide colunas obrigatórias, tipos, timezone, chaves, duplicidades, cardinalidade,
  valores do target e cobertura antes do treinamento.
- Preserve proveniência: origem, versão ou hash, intervalo observado, número de linhas e
  regras de exclusão.
- Não converta registros sem label em classe negativa. A política para labels ausentes
  ou ambíguos deve ser explícita e auditável.
- Faça o split antes de qualquer amostragem que possa alterar a distribuição de
  avaliação. A amostragem deve afetar somente treino, salvo decisão documentada.
- Preserve todos os eventos raros conhecidos quando essa for a política do projeto; se
  o orçamento tornar isso impossível, falhe com uma mensagem acionável.
- Transformações que aprendem estatísticas devem executar `fit` somente no treino.
- A inferência deve validar o schema e tolerar apenas as variações declaradas no
  contexto. Não preencha silenciosamente features críticas ausentes.

## Leakage e desenho de validação

Leakage é um defeito de correção, não apenas uma piora de qualidade.

- Escolha o split que representa o uso real: temporal para previsão futura, por grupo
  quando uma entidade não pode aparecer nos dois lados ou estratificado apenas quando
  a hipótese de intercambiabilidade for defensável.
- Em problemas temporais, ordene por uma coluna normalizada, não use shuffle e mantenha
  uma janela out-of-time intocada.
- Features históricas devem ser point-in-time correct. Use apenas eventos disponíveis
  antes da predição; agregações do próprio evento exigem `shift` ou lógica equivalente.
- Exclua IDs crus, PII sem necessidade, proxies de target, atributos pós-evento e
  snapshots cuja disponibilidade no instante da decisão não esteja comprovada.
- Busca de modelo e hiperparâmetros usa apenas treino e validação interna. Teste e OOT
  servem para confirmação ou rejeição, nunca para escolher candidato.
- Escolha o threshold somente na validação e aplique o mesmo valor, sem reajuste, no
  teste, OOT e produção.
- Uma métrica excepcionalmente alta exige auditoria de target, duplicidade, tempo,
  features e contaminação entre partições antes de ser celebrada.

Adicione testes que provem essas invariantes, inclusive limites temporais e ausência de
`fit` nas partições de avaliação.

## Treinamento e avaliação

- Comece com um baseline simples e interpretável antes de modelos complexos.
- Salve uma pipeline completa com limpeza, features, preprocessamento e estimador; não
  dependa de transformações manuais fora do artefato.
- Fixe seeds e registre versões de código, dados, feature set, bibliotecas, parâmetros e
  ambiente. Evite paralelismo não determinístico quando a comparação exigir rigor.
- Para classes desbalanceadas, não use accuracy como métrica principal. Defina no
  contexto métricas estatísticas e operacionais coerentes com o custo do domínio.
- Calcule métricas por partição e, quando aplicável, por período, grupo sensível,
  segmento e capacidade Top-K.
- Avalie calibração quando o score for interpretado como probabilidade ou alimentar
  decisões de custo.
- Compare desempenho com a taxa aleatória/base rate, não somente valores absolutos.
- Trate benchmarks e AutoML como challengers. Eles obedecem ao mesmo contrato de dados,
  split e threshold e nunca promovem a si próprios.
- Registre falhas ou dependências opcionais ausentes sem invalidar o treino principal;
  não esconda falhas do candidato vencedor ou dos artefatos obrigatórios.

## MLflow e rastreamento de experimentos

Use MLflow como trilha de auditoria e comparação, não como substituto dos contratos do
projeto. A política concreta de disponibilidade e falha pertence ao contexto.

- Crie um experimento por problema e ambiente, com nomes estáveis. Uma execução completa
  da pipeline corresponde a um run pai; folds, trials do Optuna e backends AutoML podem
  ser runs filhos para preservar a hierarquia sem poluir a visão principal.
- Defina `run_name` legível e tags pesquisáveis: `run_id`, ambiente, versão do dataset,
  hash/versão do código, versão do feature set, tipo de split, objetivo de seleção e
  nome do candidato. Não use parâmetros mutáveis como tags de identidade.
- Registre parâmetros de configuração antes do fit, métricas com nomes que incluam a
  partição (`validation_pr_auc`, `test_recall`) e métricas iterativas com `step`.
- Registre referência, versão, hash e schema do dataset. Não envie o dataset raw, PII,
  secrets ou exemplos sensíveis como artefato.
- Registre relatórios governados, curvas, leaderboard, estudo do Optuna, manifesto e
  ambiente de dependências. Artefatos grandes ou canônicos podem permanecer no object
  storage, desde que o run registre URI imutável e hash.
- Ao logar o modelo, inclua signature de entrada/saída e um input example pequeno e
  sanitizado. Valide que o modelo carregado pelo MLflow reproduz a saída do artefato
  local antes de permitir registro.
- Use autologging apenas quando habilitado explicitamente e teste quais campos ele
  produz. Evite duplicar runs, modelos ou métricas já registrados manualmente.
- Marque runs interrompidos como `FAILED` ou `KILLED` e preserve a causa de forma
  sanitizada. Nunca apresente um run parcial como concluído.
- O Model Registry não decide promoção. Registrar uma versão, associar alias ou mudar
  estágio é consequência dos gates e da aprovação definidos pela governança.
- Em desenvolvimento individual, tracking local é aceitável. Para equipe ou produção,
  use tracking server, backend persistente e artifact store compartilhado com controle
  de acesso. Credenciais ficam no ambiente ou secret manager.

Se MLflow for opcional, uma indisponibilidade deve gerar warning e resultado local
completo. Se for requisito regulatório ou operacional, falhe antes de treinar ou encerre
o run como inválido; essa escolha deve estar explícita em `.codex/PROJECT_CONTEXT.md`.

## Optuna e otimização de hiperparâmetros

- Defina uma função objetivo explícita, direção (`maximize`/`minimize`) e unidade. O
  objetivo deve refletir a métrica governada, incluindo penalidades de estabilidade ou
  custo quando documentadas, e retornar valores finitos para trials válidos.
- O objective acessa apenas treino e validação interna ou folds permitidos. Teste, OOT,
  baseline de produção e feedback futuro não podem orientar sugestões, pruning ou
  escolha do vencedor.
- Mantenha espaços de busca pequenos, tipados e condicionais. Use escalas logarítmicas
  quando a ordem de grandeza for relevante e não sugira combinações inválidas para o
  estimador.
- Configure sampler e pruner explicitamente. Fixe seed quando suportado. Só reporte
  valores intermediários comparáveis; pruning não pode usar uma partição reservada.
- Sempre limite a busca por `n_trials`, timeout ou ambos, além de CPU, memória e
  paralelismo. O contexto deve diferenciar orçamento rápido de desenvolvimento e
  orçamento oficial de comparação.
- Para retomar estudos, use storage persistente, `study_name` determinístico e política
  explícita de create/load. O nome deve separar versões incompatíveis de dataset,
  objective, espaço de busca e feature set.
- Registre por trial: estado, duração, parâmetros, seed, folds válidos, métricas,
  penalidades, motivo de pruning/falha e consumo relevante. Preserve trials falhos; não
  os converta silenciosamente em score ruim.
- Se MLflow estiver ativo, associe o estudo ao run pai e use run filho por trial apenas
  quando o volume for administrável. O resumo do estudo e o vencedor são artefatos
  obrigatórios do run pai.
- Depois da busca, reconstrua o vencedor pela mesma factory e valide seus parâmetros.
  A política de refit deve ser explícita e não pode incorporar teste/OOT. Congele o
  threshold na partição designada antes da avaliação final.
- Teste o caminho sem Optuna e o comportamento quando um modelo opcional estiver
  indisponível. Uma busca vazia ou sem trials válidos deve falhar de modo acionável.

## AutoML e benchmarks externos

AutoML é um challenger controlado, não uma exceção às regras de modelagem.

- Declare frameworks permitidos, versão, tarefa, métrica de ranking e seed. Não aceite
  a métrica default do framework sem confirmar que ela corresponde ao objetivo do
  projeto.
- Forneça as mesmas features governadas e as mesmas partições usadas pelos candidatos
  manuais. Desative splits aleatórios internos incompatíveis com tempo ou grupos; se o
  backend não aceitar o contrato de validação, não o use para selecionar o modelo.
- Defina limites de tempo, número de modelos/iterações, CPU, GPU, memória, disco e
  paralelismo. Cada backend escreve em diretório isolado pelo `run_id`.
- Leaderboard e ensembles usam somente dados permitidos para seleção. O líder é
  reavaliado uma única vez em teste/OOT, com o threshold congelado externamente quando
  essa for a política do projeto.
- Registre leaderboard completo, configuração, versões, duração, recursos, modelos
  ignorados, falhas e critério do vencedor. Compare também latência, tamanho,
  explicabilidade e portabilidade, não só a métrica principal.
- Timeouts podem reduzir reprodutibilidade. Para comparações oficiais, prefira também
  limites determinísticos de modelos/iterações quando o framework oferecer e registre
  a variabilidade restante.
- O modelo vencedor precisa ser exportável ou encapsulado por um adaptador que preserve
  schema, preprocessamento e inferência. Não promova um artefato que a aplicação não
  consegue carregar, versionar ou monitorar.
- Falhas de AutoML não apagam artefatos do treino principal. Um backend indisponível é
  registrado como `unavailable`; um backend obrigatório falha conforme a política do
  contexto.
- Nenhum líder AutoML é promovido automaticamente. Ele entra no mesmo relatório de
  challenger, gates, revisão humana e processo de registro dos demais modelos.

## Governança e ciclo de vida

Cada run concluído deve produzir, conforme aplicável:

- pipeline serializada e metadados de inferência;
- métricas de treino, validação, teste e OOT;
- auditorias de target, amostragem, leakage, drift e estabilidade;
- análise e justificativa do threshold;
- identificação de dataset, código, feature set e configuração;
- model card ou relatório consolidado para revisão humana;
- manifesto com tamanho e hash dos artefatos obrigatórios;
- histórico imutável identificado por `run_id`.

Gates de promoção devem ser determinísticos e retornar decisão, motivos e próximas
ações. Um modelo pode ser rejeitado automaticamente, mas promoção para produção exige
política explícita e, quando definido no contexto, aprovação humana confirmada. Nunca
sobrescreva baseline ou histórico silenciosamente. Operações de promoção, rollback e
migração devem ser separadas do treino, idempotentes quando possível e transacionais
entre storage e banco.

Não registre segredos, tokens, dados pessoais, amostras sensíveis ou URLs com
credenciais. `.env` é local; `.env.example` documenta apenas nomes e valores seguros.
Mudanças de schema persistente exigem migration versionada e teste de upgrade.

## Inferência e API

- Carregue pipeline e metadados compatíveis e valide versão/schema antes de servir.
- Use exatamente as transformações e o threshold registrados no run promovido.
- Mantenha score, decisão e versão do modelo rastreáveis na resposta ou telemetria.
- Valide entradas na borda e retorne erros acionáveis; não exponha stack traces,
  segredos ou detalhes internos.
- Garanta paridade entre predição em memória, batch e API com testes de contrato.
- Defina comportamento seguro para artefato ausente, incompatível ou corrompido.

## Python e qualidade de código

- Use type hints nas interfaces públicas e docstrings onde o contrato não for óbvio.
- Prefira `pathlib.Path`, dataclasses ou modelos tipados de configuração e logging em
  vez de `print` em código de produção.
- Faça funções puras para métricas e regras; injete I/O por interfaces.
- Use exceções específicas e mensagens que indiquem o campo, valor e correção esperada.
- Não capture `Exception` sem adicionar contexto e sem preservar a causa.
- Evite estado global mutável, imports com efeitos colaterais e caminhos absolutos.
- Mantenha compatibilidade com as versões declaradas; atualize requisitos, exemplos e
  testes juntos quando houver mudança.

## Estratégia de testes

Ao alterar comportamento, cubra no mínimo o nível correspondente:

- unitário: parsing de config, transformações, métricas, threshold e gates;
- contrato de dados: schema, labels, chaves, duplicidades e datas inválidas;
- temporal/leakage: ordenação, fronteiras, feature point-in-time e partições intocadas;
- integração: storage, tracking, migrations e serialização;
- paridade: pipeline salva versus carregada, batch versus API;
- regressão: todo bug corrigido ganha um teste que falhava antes.

Testes não devem depender da internet, de credenciais reais ou de grandes datasets.
Use fixtures pequenas, determinísticas e semanticamente representativas. Não reduza
assertivas de governança para acomodar uma implementação incorreta.

## Regras para mudanças comuns

Ao adicionar uma feature:

1. documente disponibilidade no instante da predição;
2. implemente em transformer compatível com a pipeline;
3. teste treino/inferência, missing values e dados futuros;
4. atualize versão do feature set e artefatos de estabilidade.

Ao adicionar um modelo:

1. registre-o na factory e valide parâmetros;
2. mantenha preprocessamento compatível;
3. inclua espaço de busca limitado e seed;
4. teste dependência ausente e ajuste mínimo;
5. compare pelo objetivo governado, sem tocar em teste/OOT.

Ao alterar target, tempo, amostragem ou split, trate a mudança como quebra de contrato:
atualize contexto, auditorias, testes, versão de dataset e política de baseline. Não
compare runs incompatíveis como se fossem equivalentes.

## Critério de pronto

Uma tarefa só está pronta quando:

- o comportamento pedido está implementado sem quebrar as invariantes acima;
- testes focados passam e a suíte proporcional ao risco foi executada;
- treino e inferência continuam compatíveis;
- configuração, `.env.example` e documentação refletem a mudança;
- novos artefatos são versionados, persistidos e incluídos no manifesto quando
  obrigatórios;
- riscos, testes não executados e decisões pendentes são informados na entrega.
