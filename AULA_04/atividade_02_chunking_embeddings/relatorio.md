# Relatório — Avaliação de Estratégias de Chunking com LangChain

## 1. Configuração dos 10 testes

| Teste | Estratégia | chunk_size | chunk_overlap | Total de chunks (12 docs) |
|---|---|---|---|---|
| 1 | fixed | 200 | 0 | 6855 |
| 2 | fixed | 500 | 0 | 2745 |
| 3 | fixed | 1000 | 0 | 1376 |
| 4 | fixed | 2000 | 0 | 692 |
| 5 | fixed_with_overlap | 500 | 50 (10%) | 3049 |
| 6 | fixed_with_overlap | 500 | 200 (40%) | 4565 |
| 7 | by_paragraph (separador \n\n) | - | 0 | 17 |
| 8 | by_sentence_group3 | - | 0 | 3761 |
| 9 | recursive | 1000 | 100 | 1854 |
| 10 | markdown_headers | - | 0 | 457 |

Modelo de embedding usado: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensões), mantido fixo em todos os 10 testes para garantir comparabilidade.

Extração PDF → Markdown feita com `pymupdf4llm`, de forma automatizada, para os 12 PDFs da pasta compartilhada pelo professor.

## 2. Análise da conversão PDF → Markdown

A extração com `pymupdf4llm` preservou bem os elementos textuais: títulos ficaram marcados com `#`/`##`, formatação em negrito/itálico foi mantida (`**texto**`, `_texto_`), e até elementos como superscript (`<sup>`) foram preservados — o que ajudou o Markdown Header Splitter (Teste 10) a capturar corretamente os headings nos metadados (ex.: heading_1: "Entre o algoritmo e o Juramento de Hipócrates...").

Por outro lado, dois pontos ficaram claramente prejudicados:

- **Quebras de parágrafo**: o texto extraído não preserva quebras de linha dupla entre parágrafos de forma consistente — o que fez o Teste 7 (chunking por parágrafo) praticamente não dividir o texto (ver seção 4, pergunta 9).
- **Imagens**: nenhuma imagem foi referenciada em nenhum dos 12 documentos (0 ocorrências de referências de imagem em todos). As figuras dos PDFs (ex.: diagramas de arquitetura em attention_is_all_you_need, gráficos em scaling_laws_llm) foram completamente descartadas na extração — não há sequer uma marcação de posição indicando que ali existia uma figura.

## 3. Análise de tabelas

As tabelas foram convertidas para o formato Markdown padrão, mas com qualidade variável dependendo da complexidade do layout original:

- **Boa preservação**: tabelas simples e bem estruturadas no PDF original, como em bert_pretraining (tabela de embeddings token/segmento/posição) e attention_is_all_you_need (tabela de complexidade por tipo de camada), saíram legíveis e com a estrutura semântica mantida.
- **Preservação problemática**: em documentos com layout mais complexo (colunas duplas, tabelas lado a lado, ou blocos de texto que se assemelham a tabelas), a extração confundiu estrutura visual com tabela real. Exemplos:
  - Em gpt3_language_models, a lista de autores do paper (organizada em colunas no PDF) foi convertida como se fosse uma tabela de dados.
  - Em retrieval_augmented_generation, duas tabelas de resultados distintas do PDF (lado a lado) foram mescladas em uma única tabela Markdown, misturando colunas de contextos diferentes.
  - Em scaling_laws_llm, o próprio sumário do paper (número da seção + título + página) foi interpretado como tabela.

Isso mostra que a extração automática por pymupdf4llm é confiável para tabelas de dados simples, mas não distingue bem layout multi-coluna de tabela semântica real — um ponto de atenção para qualquer pipeline de RAG que dependa de tabelas extraídas.

## 4. Respostas às 15 perguntas do enunciado

**1. Qual estratégia gerou mais chunks?**
O Teste 1 (fixo, 200 caracteres, sem overlap), com 6855 chunks no total — esperado, já que é o menor tamanho de chunk configurado.

**2. Qual gerou menos chunks?**
O Teste 7 (por parágrafo), com apenas 17 chunks no total para os 12 documentos — bem abaixo até do Teste 4 (2000 caracteres, 692 chunks), que era o esperado como "extremo baixo" em quantidade.

**3. Como o tamanho dos chunks variou?**
Do menor extremo (Teste 1, ~200 caracteres) ao maior extremo real (Teste 7, com chunks de até ~99.992 caracteres — nesse caso, o documento inteiro virando 1 a 3 chunks), a variação foi de mais de 400x entre estratégias. Os testes por tamanho fixo (1-6) tiveram tamanhos previsíveis e controlados; os testes por estrutura natural (7, 8, 9, 10) tiveram tamanhos irregulares, dependentes do conteúdo de cada documento.

**4. Qual estratégia preservou melhor a estrutura dos documentos?**
O Teste 10 (Markdown Header Splitter): os metadados capturam corretamente o heading da seção, permitindo saber exatamente de qual seção do documento cada chunk veio — algo que nenhum outro teste oferece nativamente.

**5. Como tabelas foram tratadas?**
Convertidas para formato Markdown, com boa fidelidade em tabelas simples e distorções em tabelas de layout complexo (colunas de autores, tabelas lado a lado mescladas, sumários interpretados como tabela) — ver seção 3.

**6. Como imagens foram tratadas?**
Completamente descartadas: 0 referências de imagem em qualquer um dos 12 documentos processados. Não há marcação de posição nem descrição textual — a informação visual é perdida integralmente na extração.

**7. Quais informações foram perdidas na conversão PDF → Markdown?**
Principalmente: (a) todo o conteúdo visual (figuras, diagramas, gráficos); (b) a granularidade de parágrafos (quebras duplas não preservadas de forma consistente); (c) a distinção entre tabela de dados real e blocos de texto organizados visualmente em colunas.

**8. O chunking por caracteres fragmentou conceitos ou estruturas importantes?**
Sim, especialmente nos testes de menor tamanho fixo (Testes 1 e 2, 200 e 500 caracteres). Como o corte acontece em uma posição de caractere fixa, sem nenhuma noção de sentença ou parágrafo, é comum o chunk terminar no meio de uma frase, de uma fórmula ou de um item de lista, e o próximo chunk começar retomando esse mesmo pensamento pela metade. Isso é coerente com o tamanho médio de apenas ~56 tokens por chunk no Teste 1 (ver Seção 7) — pequeno demais para conter uma ideia completa isoladamente. Os testes com overlap (5 e 6) reduzem o impacto, já que parte do contexto cortado reaparece no chunk seguinte, mas não eliminam o problema, pois o ponto de corte inicial continua sendo definido por contagem de caracteres, não por estrutura do texto.

**9. O chunking por parágrafo produziu chunks muito grandes?**
Sim, e de forma extrema. Como a extração via `pymupdf4llm` não preserva de forma confiável as quebras de linha dupla (`\n\n`) que definem parágrafos (ver Seção 2), o splitter praticamente não encontrou pontos de corte. Em 9 dos 12 documentos, o Teste 7 gerou **1 único chunk contendo o documento inteiro** — variando de ~42.274 caracteres (attention_is_all_you_need) a ~95.355 caracteres (scaling_laws_llm). Apenas 3 documentos (gpt3_language_models, gpt4_technical_report, instruct_gpt) foram divididos em mais de um pedaço (2 a 3 chunks), e mesmo assim cada pedaço ainda ficou com dezenas de milhares de caracteres — no caso mais extremo, um chunk de gpt4_technical_report chegou a 99.992 caracteres. Na prática, essa estratégia não funcionou como "chunking" nesse pipeline: sem parágrafos bem delimitados na extração, ela vira quase um "não-split".

**10. O chunking por sentença conseguiu preservar melhor o contexto?**
Parcialmente. Para a maior parte do texto corrido, agrupar 3 sentenças por chunk (Teste 8) produziu unidades coerentes e de tamanho relativamente controlado — a média por documento ficou majoritariamente entre ~200 e ~510 caracteres, próxima da faixa dos testes fixos de tamanho médio. Porém, o teste apresentou outliers grandes em vários documentos: gpt4_technical_report chegou a um chunk de 18.703 caracteres, gpt3_language_models a 6.044, e instruct_gpt a 3.684 — muito acima da média. Isso indica que, em trechos densos como tabelas, listas ou blocos de referências (onde há poucos "." que marcam fim de sentença), o detector de sentenças falhou em encontrar pontos de corte, e um bloco inteiro acabou sendo tratado como "uma sentença só" e agrupado em um chunk desproporcional. Ou seja: preserva bem o contexto em texto corrido, mas não é robusto diante de conteúdo estruturado.

**11. O Recursive Splitter apresentou vantagens?**
Sim, foi a estratégia com o comportamento mais consistente entre os testes estruturais. Com `chunk_size=1000` e `chunk_overlap=100`, o tamanho médio por documento ficou sempre numa faixa estreita (~730 a ~810 caracteres) e o tamanho máximo praticamente nunca ultrapassou o limite configurado (chegando no máximo a 995-999 caracteres em todos os documentos) — diferente do Teste 8, que teve outliers de até 18 mil caracteres. Isso mostra a vantagem central do Recursive Splitter: ele tenta cortar em separadores hierárquicos (parágrafo → linha → espaço → caractere), mas nunca deixa de respeitar o limite máximo de tamanho, combinando previsibilidade de tamanho fixo com uma tentativa de respeitar a estrutura natural do texto. O único ponto fraco observado foram tamanhos mínimos muito pequenos (2 caracteres em gpt3_language_models e gpt4_technical_report), provavelmente fragmentos residuais no fim de uma seção.

**12. O Markdown Splitter conseguiu preservar a estrutura semântica?**
Sim, no sentido de que os metadados (`heading_1`, `heading_2` etc.) sempre indicam corretamente de qual seção do documento o chunk veio — nenhum outro teste oferece essa rastreabilidade. Mas ele não controla o tamanho dos chunks: a média por documento variou de ~1.591 a ~4.982 caracteres, e o tamanho máximo chegou a extremos como 28.308 caracteres (gpt3_language_models), 23.641 (retrieval_augmented_generation) e 22.982 (gpt4_technical_report). Isso acontece quando um documento tem poucos headings cobrindo blocos grandes de conteúdo — o splitter respeita rigorosamente a fronteira do heading, mas não subdivide o que está dentro dela. Resultado: ótimo para saber "de onde" veio um chunk, mas ruim para garantir chunks de tamanho adequado para embeddings.

**13. Qual estratégia parece mais adequada para um sistema de RAG?**
Isoladamente, o Teste 9 (Recursive, 1000/100) é o mais equilibrado dos dez: respeita um limite máximo de tamanho (bom para embeddings e para o contexto do modelo), tenta cortar em fronteiras naturais do texto, e não apresenta os outliers extremos vistos nos Testes 7, 8 e 10. Para um sistema de RAG mais maduro, porém, o ideal seria uma abordagem em duas etapas: usar o Markdown Header Splitter (Teste 10) primeiro, para dividir o documento respeitando as seções, e depois aplicar o Recursive Splitter dentro de cada seção para garantir que nenhum chunk individual ultrapasse um tamanho razoável — unindo a rastreabilidade estrutural do Teste 10 com o controle de tamanho do Teste 9.

**14. Quais estratégias devem ser descartadas?**
O Teste 7 (por parágrafo) deve ser descartado neste pipeline: dado que a extração não preserva parágrafos de forma confiável, ele não cumpre sua função (na prática, quase não divide o documento). O Teste 1 (200 caracteres) também é problemático como estratégia isolada, pela fragmentação de ideias apontada na pergunta 8 — poderia servir apenas como sub-chunking dentro de uma estratégia estrutural, não como estratégia principal.

**15. Quais estratégias você acha que devem ser utilizadas nos próximos experimentos?**
Recomenda-se: (a) uma combinação de Markdown Header Splitter + Recursive Splitter, conforme descrito na pergunta 13; (b) investigar uma extração alternativa (ou pós-processamento) que reconstrua quebras de parágrafo de forma mais confiável, o que tornaria o Teste 7 viável e possivelmente competitivo; (c) explorar chunking semântico (baseado em similaridade de embeddings entre sentenças consecutivas) como evolução natural além das estratégias puramente estruturais testadas aqui.

## 5. Exemplos de chunks

Para ilustrar a diferença de comportamento entre uma estratégia de tamanho fixo e uma estrutural, seguem exemplos reais extraídos de `bioetica_e_ia.pdf`.

**Teste 3 (fixo, 1000 caracteres, sem overlap):**

- `chunk_id: bioetica_e_ia_test03_chunk001` — metadata vazio (`{}`). Texto inicial: cabeçalho da revista, título do artigo e primeiros autores, cortado no meio da lista de autores.
- `chunk_id: bioetica_e_ia_test03_chunk002` — continuação direta do chunk anterior, começando literalmente no meio da palavra "Hipócrates" ("ipócrates. A incorporação da inteligência artificial..."), evidenciando o corte cego por contagem de caracteres.

**Teste 10 (Markdown Header Splitter):**

- `chunk_id: bioetica_e_ia_test10_chunk001` — metadata vazio (cabeçalho da revista, antes de qualquer heading).
- `chunk_id: bioetica_e_ia_test10_chunk002` — metadata: `{'heading_1': 'Entre o algoritmo e o Juramento de Hipócrates: bioética na era da inteligência artificial'}`. Texto inicial: bloco de autores e afiliação, começando de forma limpa logo após o heading, sem cortar palavras ao meio.

A comparação confirma visualmente o que os números já indicavam: o Teste 3 corta sem respeitar nenhuma fronteira textual, enquanto o Teste 10 sempre inicia um novo chunk em um ponto estruturalmente coerente (logo após um heading), ao custo de não controlar o tamanho de cada chunk resultante.

## 6. Conclusão

Nenhuma das 10 estratégias isoladas é ideal para um sistema de RAG: as baseadas em tamanho fixo (1-6) são previsíveis mas cegas à estrutura do texto, e as baseadas em estrutura natural (7-10) respeitam melhor o conteúdo mas têm tamanho de chunk incontrolável — no limite, o Teste 7 mostrou que uma estratégia estrutural pode falhar completamente se a etapa de extração anterior não preservar os marcadores de que ela depende (quebras de parágrafo, no caso).

O Teste 9 (Recursive Splitter) foi o que melhor equilibrou os dois lados nesta avaliação: manteve o tamanho de chunk sempre dentro de um limite previsível, evitando os outliers extremos observados nos Testes 7, 8 e 10, enquanto ainda tentava respeitar separadores naturais do texto. O Teste 10 (Markdown Header Splitter) continua sendo insubstituível quando se precisa saber de qual seção do documento cada chunk veio, mas não deve ser usado sozinho quando o controle de tamanho é importante.

A recomendação para próximos experimentos é combinar as duas abordagens — dividir primeiro por estrutura semântica (headings) e depois por tamanho controlado (recursive) dentro de cada seção — além de revisitar a etapa de extração PDF → Markdown para melhorar a preservação de parágrafos e, assim, tornar viável uma comparação mais justa com o chunking por parágrafo.

## 7. Contagem de tokens (testes 1 a 6)

Utilizando o tokenizador cl100k_base (tiktoken), a média de tokens por chunk cresceu de forma proporcional ao chunk_size configurado, como esperado (~4 caracteres por token em média):

| Teste | chunk_size | chunk_overlap | Média de tokens/chunk |
|---|---|---|---|
| 1 | 200 | 0 | 56.0 |
| 2 | 500 | 0 | 138.6 |
| 3 | 1000 | 0 | 275.7 |
| 4 | 2000 | 0 | 545.4 |
| 5 | 500 | 50 | 138.6 |
| 6 | 500 | 200 | 138.9 |

Um ponto interessante: os Testes 2, 5 e 6 usam o mesmo chunk_size (500 caracteres), mas variam o overlap — e a média de tokens por chunk ficou praticamente idêntica entre eles (138.6, 138.6, 138.9). Isso é esperado, já que o overlap não altera o tamanho de cada chunk individual, apenas quanto de conteúdo se repete entre chunks consecutivos (o que se reflete no número total de chunks gerados, não no tamanho médio de cada um).

Isso confirma que os chunks do Teste 1 (200 caracteres, ~56 tokens em média) são pequenos demais para carregar contexto substancial isoladamente, enquanto os do Teste 4 (2000 caracteres, ~545 tokens) já se aproximam do limite de contexto recomendado para muitos embedding models, o que reforça a preferência por chunk sizes intermediários (como no Teste 3, ~276 tokens, ou nas estratégias estruturais como Recursive e Markdown Headers) para um sistema de RAG equilibrado.
