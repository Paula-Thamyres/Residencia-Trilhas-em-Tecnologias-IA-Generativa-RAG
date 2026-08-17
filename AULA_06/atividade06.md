# Aula 06 — Projeto e Arquitetura de uma Aplicação RAG

Escolhi dois cenários bem diferentes de propósito: um suporte de TI interno (procedimentos, baixo risco de erro) e uma consulta a contratos de locação de imobiliária (documento jurídico, erro caro). A ideia era forçar decisões diferentes de verdade, não só trocar o "tema" em cima do mesmo esqueleto.

---

# CENÁRIO 1 — Central de Suporte de TI

## O problema

Os atendentes N1 (galera nova, 0 a 2 anos de TI) do service desk perdem tempo demais procurando o procedimento certo numa wiki que foi exportada do Confluence pra Markdown/PDF há um tempo e nunca mais foi organizada direito. Na prática isso vira duas coisas ruins: ou o atendente segue um procedimento desatualizado, ou escala pro N2 um chamado que ele mesmo resolveria se achasse a informação rápido.

Quem usa: analistas de suporte, nível técnico baixo/médio, ainda aprendendo a stack interna (VPN, ERP, IAM, impressoras, políticas de senha). O acesso seria via chat dentro do próprio Freshdesk, não uma API — é apoio em tempo real pro atendente, enquanto o chamado está aberto.

Perguntas reais que apareceriam: "o notebook não conecta na VPN, já reiniciei o cliente Cisco, o que mais eu testo antes de escalar?", "posso liberar acesso à pasta do financeiro pra um estagiário ou precisa aprovação?", "a senha do Pedro expirou e ele não recebe o e-mail de redefinição, é bug conhecido?"

## Por que RAG faz sentido aqui

O conhecimento é 100% interno — nome de sistema, sequência de tela do ERP, isso não tá em lugar nenhum que um modelo genérico já saiba. Fine-tunar um modelo próprio pra isso seria caro e ficaria velho rápido (procedimento muda algumas vezes por mês). RAG resolve isso: o conhecimento fica nos documentos, atualizável, e o modelo só recupera e formula.

Um exemplo bom de erro que um LLM sem RAG cometeria: perguntado sobre liberar acesso a uma pasta de rede, ele provavelmente responderia algo como "peça pro admin adicionar o usuário ao grupo no Active Directory" — que soa certo, mas nessa empresa esse fluxo não passa mais direto pelo AD, existe uma aprovação obrigatória do gestor via um portal chamado IAM Corp. Um atendente que seguisse a resposta genérica pularia justamente o controle de segurança que existe ali.

## Quando RAG NÃO seria a resposta certa

Isso eu acho que é a parte mais importante de justificar, então fui ponto a ponto:

Busca por palavra-chave simples (tipo Elasticsearch) resolveria bem quando o atendente já sabe o termo técnico exato ("IAM Corp", "VPN concentrator") — é mais barato e mais previsível que RAG pra esse caso. RAG ganha quando a pergunta vem descrita pelo sintoma ("notebook não conecta") em vez do nome técnico.

Pra perguntas de contagem tipo "quantos chamados de VPN essa semana" a resposta certa é SQL direto no Freshdesk, nunca RAG — buscar isso em texto de manual é receita pra alucinação, porque RAG só recupera pedaços (top-k), nunca a base inteira, e o modelo tende a "preencher" o resto com um número que parece plausível mas não é.

Pros runbooks de incidente mais críticos e mais repetidos (queda de VPN, por exemplo), até vale transformar em checklist determinístico dentro do próprio sistema de chamados, em vez de depender de busca semântica — reduz variância. RAG fica melhor pros casos de cauda longa que não justificam virar automação fixa.

E tem parte que é resposta de API, não de texto: se a dúvida é "o e-mail de redefinição foi enviado ou não", isso o sistema de identidade sabe com certeza, um manual não.

Na prática, a arquitetura final combina as três coisas: RAG pro "como fazer", API/BD pro "qual o status agora", e o modelo decide qual usar.

## Organização dos documentos

Tipos: maioria Markdown (do Confluence), alguns PDFs antigos de Word, planilhas de referência (código de erro de impressora → causa), e prints de tela embutidos nos manuais. Volume: uns 350-450 documentos. Frequência: poucos documentos totalmente novos por mês, mas atualizações de existentes acontecem várias vezes por semana (toda vez que um sistema interno muda).

Estrutura de pastas:

```
documentos/
├── runbooks_incidente/ (rede, identidade_e_acesso, hardware)
├── manuais_procedimento/ (onboarding_offboarding, sistemas_internos, impressao_e_hardware)
├── politicas/ (seguranca, uso_de_recursos)
└── obsoletos/  (fora do índice, só pra auditoria)
```

A divisão segue como o atendente realmente pensa quando abre um chamado: "isso é um incidente agora" (runbook), "isso é dúvida de como fazer" (manual) ou "isso é regra/permissão" (política). Isso vira direto um metadado `category` usado como filtro — não é só estética, a pasta espelha o filtro que vai ser aplicado na busca.

O que não entra: manuais com credencial de exemplo real (já vi caso de atendente colar senha real de ambiente de teste num tutorial) e documentos de RH que vazaram pra wiki de TI por engano. Isso é barrado com uma triagem manual antes da ingestão inicial e depois por um scanner simples de regex (CPF, string parecida com senha).

Versão do documento: cada manual tem um `document_id` fixo — quando edita, substitui o conteúdo indexado, não duplica. Não guardo versão antiga pesquisável de propósito: se o procedimento de reset de senha mudou, não existe cenário em que o atendente precise da versão velha.

## Pipeline de ingestão

**Extração:** a maior parte já é Markdown, extração é trivial. Os PDFs antigos têm texto selecionável (não são digitalizados), então uso parser de texto normal, sem OCR. Tabela de erro de impressora eu preservo como Markdown table dentro do chunk — virar texto corrido destruiria a relação código→causa. Imagem (print de tela) hoje eu simplesmente descarto do texto extraído — não tem legenda na maioria dos manuais antigos, então extrair "sentido" da imagem custaria caro pra pouco retorno nessa primeira versão (fica marcado como risco conhecido). Um problema que já peguei numa atividade anterior: PDF gerado de um Word com duas colunas embaralhou a ordem do texto na extração simples, misturando uma linha da esquerda com uma da direita — resolvi configurando o parser pra respeitar posição (bounding box) em vez de fluxo bruto.

**Limpeza:** removo cabeçalho/rodapé repetido do template do Confluence, menu de navegação colado no export, e o sumário automático que duplica os próprios títulos. Padronizo encoding (tinha arquivo antigo em Latin-1 quebrando acentuação), quebra de linha e espaçamento. Risco real que já vi acontecer: um filtro de "linha muito curta = ruído" acabou apagando passo de checklist numerado curto tipo "3. Reinicie o serviço." — a lição foi nunca filtrar por tamanho isolado sem olhar o contexto estrutural.

**Frequência:** roda de duas formas — webhook dispara reprocessamento assim que o documento é editado, e uma varredura completa aos domingos à noite como rede de segurança. Quando um documento muda, reprocesso só ele (comparando hash de conteúdo), não a base inteira — reprocessar tudo a cada edição isolada seria desperdício de embedding sem necessidade real.

## Metadados

Documento: `document_id`, `title`, `author`, `source`, `document_type`, `category`, `created_at`, `updated_at`, `content_hash`, `criticidade`.
Chunk: os mesmos campos relevantes + `chunk_id` e `section` (derivado dos títulos Markdown no chunking).

`category` e `document_type` são os filtros mais usados — se o atendente já sinaliza "é sobre VPN", restrinjo a busca antes mesmo de rodar a semântica, o que evita recuperar, por exemplo, um trecho de política quando o assunto era procedimento técnico. `criticidade` dispara um aviso extra na resposta pros runbooks mais sensíveis ("confirme com N2 antes de executar em produção"). `updated_at` + `content_hash` são o que garante que o chunk mostrado é a versão vigente.

Pra citar fonte: `title` + `section` + `updated_at` (o atendente precisa saber se aquilo é recente antes de confiar).

`category` seria caro de adicionar depois porque hoje é atribuída manualmente por quem escreve o documento — reclassificar 400 documentos retroativamente exigiria revisão humana de cada um.

## Chunking

Recursivo por seção (títulos Markdown como corte preferencial), fallback por parágrafo. Tamanho: 500-700 tokens, overlap de 100.

Por quê esse tamanho: os runbooks são organizados em passos numerados curtos (3-6 frases cada). 500-700 tokens cobre um passo inteiro com uma margem, sem misturar dois passos no mesmo chunk — o que aconteceria fácil com 1500 tokens, diluindo a relevância dos dois. O overlap de 100 é principalmente pra não cortar frase de transição tipo "se o teste falhar, prossiga pro Passo 4".

Políticas (mais prosa corrida, menos passo a passo) uso chunk um pouco maior, ~900 tokens, porque o raciocínio de uma política se estende por 2-3 parágrafos ligados antes de chegar na regra final — cortar cedo perde a exceção que dá sentido à regra.

Chunk pequeno demais perde contexto (passo sem o resultado esperado do teste é inútil sozinho). Chunk grande demais dilui a relevância — o runbook inteiro num chunk faz a busca recuperar "o documento todo meio relevante" em vez do passo certo.

Pra validar se o chunking ficou bom: eu rodaria umas 30 perguntas reais coletadas com os próprios atendentes e checaria manualmente se o chunk do top-3 tem a informação completa, não só "relacionada".

## Embeddings

Escolhi `text-embedding-3-small` da OpenAI: 1536 dimensões, multilíngue (com português), até 8.191 tokens de entrada, US$ 0,02 por milhão de tokens (fonte: openai.com/index/new-embedding-models-and-api-updates e developers.openai.com/api/docs/guides/embeddings).

Faz sentido aqui porque os documentos não são sigilosos no sentido jurídico — o pouco de dado sensível (senha de exemplo) já é barrado antes da ingestão. Como a base é reprocessada com frequência, custo por token importa, e o `small` é a opção mais barata da família com suporte bom a português.

Considerei o `text-embedding-3-large` (3072 dim, ~US$ 0,13/milhão) e descartei — o ganho de precisão não compensa 6,5x mais custo, porque a maior parte dos erros de recuperação aqui vem de chunking ruim ou falta de filtro, não de limitação do embedding em si.

---

# CENÁRIO 2 — Contratos de Locação de uma Imobiliária

## O problema

Uma imobiliária que administra ~900 imóveis alugados, com histórico de 15 anos de contratos (ativos e encerrados). Corretores e o time administrativo precisam, várias vezes por dia, tirar dúvida pontual sobre uma cláusula específica — valor, índice de reajuste, responsabilidade de manutenção, multa — sem reabrir e ler manualmente um PDF de 15-25 páginas toda vez.

Quem usa: corretores e time administrativo/financeiro, perfil não técnico (confortável com WhatsApp e planilha, não com sistema complexo). Uso seria uma interface web simples: busca por endereço/nome do locatário pra achar o contrato certo, depois pergunta em linguagem natural sobre aquele contrato específico.

Perguntas reais: "o aluguel da Rua Voluntários da Pátria, 450 reajusta em que mês e por qual índice?", "esse contrato tem fiador ou é só caução? quanto ficou a caução?", "quem é responsável pelo portão da garagem, dono ou inquilino?"

## Por que RAG

O conhecimento aqui é 100% privado e varia contrato a contrato — não existe forma de um modelo genérico saber a multa de rescisão do contrato de um endereço específico. RAG recupera a cláusula certa daquele contrato e monta a resposta em linguagem simples.

Exemplo de erro perigoso sem RAG: perguntado sobre o índice de reajuste, um LLM tende a "assumir" o mais comum no mercado (IGP-M) e responder como fato — quando aquele contrato específico pode ter negociado IPCA ou índice fixo. É um erro silencioso e plausível, porque o corretor não tem motivo óbvio pra desconfiar de "IGP-M".

## Quando RAG não resolve

Busca por endereço ou nome exato é melhor resolvida por busca textual/metadado do que por semântica — por isso a arquitetura usa busca exata pra achar o contrato certo e só depois RAG dentro daquele documento específico, nunca uma busca solta nos 3.000 contratos misturados.

Dado estruturado (valor, data de vencimento, status) deveria estar espelhado num banco relacional/ERP e consultado direto ali, não via RAG sobre o PDF.

Cálculo de reajuste (aplicar índice numa data) tem que ser função determinística puxando o índice real de uma API oficial (FGV/IBGE) — nunca o LLM "calculando" a partir de texto, o risco de erro aritmético em valor financeiro é inaceitável.

Pergunta que RAG responde mal e SQL responde bem: "quais contratos vencem nos próximos 30 dias" — isso é varrer todos os contratos e comparar data, RAG (top-k de um documento) não faz isso. E se alguém perguntasse algo agregado tipo "quantos aluguéis reajustam em setembro", RAG recuperaria só alguns contratos "parecidos" com a pergunta e devolveria um número incompleto com aparência de certeza — exatamente o tipo de erro que pode gerar prejuízo real numa imobiliária.

## Organização dos documentos

PDFs — parte nascida digital (a partir de 2022, texto selecionável), parte escaneada (mais antigos). Tem também anexo de vistoria com fotos. Volume: ~3.000 contratos no total, ~900 ativos hoje. 15-25 novos por mês; um contrato individual quase nunca muda, só recebe aditivo formal (raro).

```
documentos/
├── contratos_ativos/ (residencial, comercial)
├── contratos_encerrados/ (residencial, comercial)
├── aditivos/
└── vistorias/
```

`ativo`/`encerrado` é o primeiro filtro que importa de longe — o mesmo endereço pode ter tido 3 locatários diferentes em 15 anos, cada um com seu próprio contrato, e ninguém quer que uma pergunta sobre o contrato vigente recupere, por acidente, cláusula de um já encerrado. `aditivos` fica separado porque complementa, não substitui — o sistema precisa juntar contrato + aditivos vinculados na resposta.

O que não entra na base: documento de identificação anexado (RG, comprovante de renda, CPF isolado) — não ajuda a responder pergunta de cláusula e só aumenta exposição de dado pessoal sensível à toa. Isso é resolvido simplesmente não incluindo essa subpasta no pipeline.

Versão: diferente do Cenário 1, aqui o problema não é "qual versão do texto" (contrato assinado não muda), é "qual contrato, entre vários do mesmo imóvel ao longo do tempo, é o vigente" — resolvido via `status` + `vigencia_inicio/fim`, com filtro `status: ativo` obrigatório por padrão em toda busca.

## Pipeline de ingestão

**Extração:** contratos recentes (texto nativo) usam parser de texto normal. Os escaneados (antes de 2022) exigem OCR, e aqui o cuidado é maior que no Cenário 1 porque o erro tem consequência financeira direta (um "8" lido como "3" no valor do aluguel é grave). Uso um OCR de qualidade melhor que Tesseract puro e marco todo chunk vindo de OCR com um score de confiança, pra avisar o usuário quando o valor merece conferência no original.

Tabela de cronograma de reajuste, quando existe, é tratada como bloco atômico, mesma lógica do Cenário 1.

Foto de vistoria eu não descarto (diferente do Cenário 1 com prints de tela), porque "o portão já tinha esse defeito na entrega?" depende diretamente da foto — a solução é gerar legenda textual via modelo de visão no momento da ingestão e indexar essa legenda vinculada ao laudo.

Problema real de OCR (já vivido em atividade anterior): assinatura manuscrita sobreposta ao final de um parágrafo fez o OCR interpretar parte do texto como ruído gráfico e descartar uma frase inteira da cláusula. Mitigo revisando amostralmente os primeiros lotes de cada categoria antes de confiar no pipeline.

**Limpeza:** removo carimbo/marca d'água de cartório, timbre repetido, numeração de página. Aqui o risco de limpar demais é mais sério que no Cenário 1 — cláusula tem valor jurídico literal, então a limpeza é estritamente cosmética, nunca reescreve ou resume o corpo da cláusula (diferente de manual de TI, onde reorganizar pra clareza seria aceitável — aqui mudar uma vírgula pode mudar sentido jurídico).

**Frequência:** não existe "documento que muda" aqui, existe "documento novo que chega". Roda sob demanda quando um contrato é arquivado formalmente. Aditivo entra como documento novo próprio, vinculado ao original via `contrato_original_id` — o original nunca é reprocessado.

## Metadados

Documento: `document_id`, `title`, `author`, `source`, `document_type`, `categoria_imovel`, `status`, `vigencia_inicio/fim`, `extraction_method`, `contrato_original_id`.
Chunk: os campos relevantes + `chunk_id`, `page`, `clausula`.

`status` é o metadado mais crítico do cenário inteiro — sem o filtro `status: ativo`, uma pergunta pode recuperar cláusula de um contrato de 2019 já encerrado sobre o mesmo endereço, com valor totalmente desatualizado. `vigencia_inicio/fim` vêm do ERP, não do PDF (não dá pra extrair do texto se um contrato ainda está vigente). `extraction_method` + confiança de OCR alimentam o aviso de "confira o original" — coisa que não existe no Cenário 1 porque lá não tem OCR. `clausula` é o que aparece na citação da fonte junto com a página.

`status` e `vigencia_fim` seriam caros de adicionar depois — exigiriam checar manualmente (ou cruzar com o ERP) cada um dos ~3.000 contratos históricos.

## Chunking

Por cláusula (usando a numeração/título de cláusula do próprio contrato via regex), fallback por parágrafo. Tamanho: 1.000-1.200 caracteres, overlap de 200.

Por quê: as cláusulas desse modelo de contrato têm 3-5 parágrafos. Testei com chunk menor (500-600 caracteres) e partiu cláusula no meio de uma condicional — separou "o locatário é responsável pela manutenção" da exceção logo depois ("salvo em caso de defeito estrutural preexistente"), o que faria o sistema responder de forma incompleta sobre responsabilidade. O overlap de 200 cobre a fronteira entre cláusulas vizinhas.

Aqui a prioridade número um é nunca cortar dentro de uma cláusula, mesmo que isso deixe os chunks mais irregulares que no Cenário 1 — porque uma cláusula do tipo "o locador NÃO é responsável, EXCETO quando..." cortada antes do "exceto" inverte o sentido jurídico de forma perigosa. Chunk grande demais também é problema: juntar "Do Reajuste" com "Da Rescisão" no mesmo chunk aumenta o risco do modelo confundir as duas condições.

Pra validar: montaria perguntas de teste cobrindo cada tipo de cláusula (valor, reajuste, responsabilidade, multa, garantia) numa amostra de 15-20 contratos, e checaria manualmente se o chunk recuperado tem a cláusula completa — com atenção redobrada pras condicionais/exceções, que é o ponto mais frágil.

## Embeddings

Escolhi `bge-m3` da BAAI, auto-hospedado: 1024 dimensões (modo denso), multilíngue (100+ idiomas, incluindo português), até 8.192 tokens de entrada, open source, roda localmente, sem custo por token — só custo de infraestrutura (fonte: huggingface.co/BAAI/bge-m3 e o paper original em arxiv.org/html/2402.03216v3).

A decisão aqui é o oposto do Cenário 1, e por um motivo específico: os contratos têm dado pessoal de terceiros protegido por LGPD (nome, CPF, endereço de locador/locatário). Mandar isso pra uma API de terceiro, por mais confiável que seja, cria uma dependência de compartilhamento de dado pessoal que a imobiliária precisaria justificar formalmente. Com volume controlado (~3.000 contratos, ingestão majoritariamente pontual), hospedar localmente é viável e elimina esse risco de vez.

Considerei o `text-embedding-3-large` da OpenAI pelo desempenho, mas descartei pelo mesmo motivo de privacidade — o critério decisivo aqui não foi qualidade do embedding, foi onde o dado sensível pode ou não trafegar.

---

# Arquitetura final

### Cenário 1

%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#F8F0F8",
    "primaryTextColor": "#332936",
    "lineColor": "#B889C5",
    "fontFamily": "Arial",
    "fontSize": "9px",
    "clusterBkg": "#F2E5F4",
    "clusterBorder": "#CDA8D5"
  },
  "flowchart": {
    "nodeSpacing": 8,
    "rankSpacing": 12,
    "curve": "basis",
    "padding": 3
  }
}}%%

flowchart TD

    A[Atendente pergunta no chat] --> B{Classificador de intenção}
    B -->|Procedimento / política| C[Busca vetorial filtrada<br/>category / document_type]
    B -->|Status / contagem| D[Consulta API<br/>Freshdesk / IAM Corp]
    C --> E[Top-k chunks]
    E --> F[LLM monta resposta<br/>com citação]
    D --> F
    F --> G[Resposta ao atendente<br/>+ aviso de criticidade]

    subgraph ING["Ingestão"]
        direction LR
        H[Documentos] --> I[Extração] --> J[Limpeza] --> K[Metadados] --> L[Chunking por seção] --> M[Embedding 3-small] --> N[(Banco vetorial)]
    end

    N --> C

    classDef fluxo fill:#FCE4EF,stroke:#CF78A4,color:#332936,stroke-width:1px;
    classDef ia fill:#EDE1F3,stroke:#9C70B3,color:#33263A,stroke-width:1px;
    classDef decisao fill:#F1E5F4,stroke:#B77BC4,color:#33263A,stroke-width:1px;
    classDef banco fill:#E3D5EC,stroke:#8F64A8,color:#302238,stroke-width:1px;
    classDef ingest fill:#F7EAF3,stroke:#C58CAA,color:#382A34,stroke-width:1px;

    class A,C,D,E,G fluxo;
    class F ia;
    class B decisao;
    class N banco;
    class H,I,J,K,L,M ingest;

    style ING fill:#F0E4F2,stroke:#CBA5D3,stroke-width:1px,color:#59435F;

### Cenário 2

%%{init: {
  "theme": "base",
  "themeVariables": {
    "background": "#F8F0F8",
    "primaryTextColor": "#332936",
    "lineColor": "#B889C5",
    "fontFamily": "Arial",
    "fontSize": "9px",
    "clusterBkg": "#F2E5F4",
    "clusterBorder": "#CDA8D5"
  },
  "flowchart": {
    "nodeSpacing": 8,
    "rankSpacing": 12,
    "curve": "basis",
    "padding": 3
  }
}}%%

flowchart TD

    A[Corretor busca imóvel / locatário] --> B[Busca exata por metadado<br/>+ status = ativo]
    B --> C[Contrato identificado]
    C --> D[Pergunta sobre esse contrato]
    D --> E[Busca vetorial restrita<br/>ao document_id]
    E --> F[Chunks da cláusula<br/>+ aditivos vinculados]
    F --> G[LLM monta resposta<br/>citando cláusula + página]
    G --> H{Envolve cálculo financeiro?}
    H -->|Sim| I[API de índice oficial<br/>IGP-M / IPCA]
    H -->|Não| J[Resposta final]
    I --> J

    subgraph ING["Ingestão"]
        direction LR
        K[PDFs] --> L{Tem texto nativo?}
        L -->|Sim| M[Extração de texto]
        L -->|Não| N[OCR + confiança]
        M --> O[Limpeza cosmética]
        N --> O
        O --> P[Metadados via ERP]
        P --> Q[Chunking por cláusula]
        Q --> R[bge-m3 local]
        R --> S[(Banco vetorial local)]
    end

    S --> E

    classDef fluxo fill:#FCE4EF,stroke:#CF78A4,color:#332936,stroke-width:1px;
    classDef ia fill:#EDE1F3,stroke:#9C70B3,color:#33263A,stroke-width:1px;
    classDef decisao fill:#F1E5F4,stroke:#B77BC4,color:#33263A,stroke-width:1px;
    classDef banco fill:#E3D5EC,stroke:#8F64A8,color:#302238,stroke-width:1px;
    classDef ingest fill:#F7EAF3,stroke:#C58CAA,color:#382A34,stroke-width:1px;

    class A,B,C,D,E,F,I,J fluxo;
    class G,R ia;
    class H,L decisao;
    class S banco;
    class K,M,N,O,P,Q ingest;

    style ING fill:#F0E4F2,stroke:#CBA5D3,stroke-width:1px,color:#59435F;

### Tabela de decisões

| Etapa           | Cenário 1                           | Cenário 2                                 |
| --------------- | ----------------------------------- | ----------------------------------------- |
| Extração        | Texto nativo, sem OCR               | Texto nativo ou OCR com score             |
| Limpeza         | Remove ruído estrutural             | Só cosmética, nunca toca na cláusula      |
| Chunking        | Por seção, 500-700 tok, overlap 100 | Por cláusula, 1000-1200 char, overlap 200 |
| Metadados-chave | category, criticidade               | status, vigencia_fim                      |
| Embedding       | text-embedding-3-small (API)        | bge-m3 (local, por LGPD)                  |

### Riscos que sei que não resolvo

**Cenário 1:** print de tela some do índice (perde sentido em manual que depende de imagem); classificador de intenção pode errar e mandar pergunta de status pra busca vetorial; nada garante automaticamente que o procedimento recuperado bate com a versão real do sistema em produção.

**Cenário 2:** OCR de baixa confiança pode errar valor numérico sem ser pego pelo score, principalmente com assinatura sobreposta; chunking depende do contrato seguir o template padrão — contrato atípico cai no fallback com qualidade pior; separação ativo/encerrado depende do ERP estar sempre sincronizado a tempo.

---

# Comparação entre os dois

O que mudou de verdade: embedding (API paga vs. local — decidido por LGPD, não desempenho), unidade de chunking (passo numerado vs. cláusula jurídica), tolerância a reescrita na limpeza (aceitável num manual, inaceitável num contrato) e o papel da busca por metadado (opcional no Cenário 1, obrigatória e anterior à busca semântica no Cenário 2).

O que ficou igual, e por que acho que é princípio de verdade e não repetição preguiçosa: os dois usam chunking recursivo respeitando a estrutura nativa do documento (nunca corte cego por caractere) e os dois combinam RAG com uma fonte estruturada pra qualquer coisa que envolva contar/somar/calcular — nunca deixando o LLM fazer isso a partir de texto recuperado. Isso não é coincidência de tema, é o mesmo problema aparecendo nos dois contextos.

Se tivesse que escolher só um pra construir: o Cenário 1. Não porque seja mais fácil — o Cenário 2 tem problema mais interessante (OCR, LGPD, estrutura jurídica) — mas porque erro no Cenário 1 é reversível (o atendente escala) e erro no Cenário 2 pode virar disputa jurídica ou prejuízo financeiro real. Prefiro validar o pipeline num ambiente de risco baixo antes de aplicar em um onde o erro custa caro.

---

# Referências

- OpenAI. _New embedding models and API updates._ https://openai.com/index/new-embedding-models-and-api-updates/
- OpenAI. _Vector embeddings — API Guide._ https://developers.openai.com/api/docs/guides/embeddings
- OpenAI. _text-embedding-3-large — Model card/pricing._ https://developers.openai.com/api/docs/models/text-embedding-3-large
- Hugging Face. _BAAI/bge-m3 — Model Card._ https://huggingface.co/BAAI/bge-m3
- Chen, J. et al. _BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation._ https://arxiv.org/html/2402.03216v3
- Mermaid Live Editor. https://mermaid.live/

---

# Como usei IA nesta atividade

Durante esta atividade, usei a IA principalmente como uma ferramenta de apoio para organizar minhas ideias, levantar possibilidades e revisar algumas decisões técnicas. Não usei a IA simplesmente para gerar uma resposta pronta e copiar, porque eu precisava entender por que cada escolha fazia sentido dentro de cada cenário.

Uma das partes em que a IA me ajudou foi na exploração das possibilidades de arquitetura. Como os dois cenários eram bem diferentes - um sistema de suporte de TI e uma aplicação para consulta de contratos de locação - usei a IA para discutir alternativas de uso de RAG, busca por palavras-chave, banco de dados, APIs e busca semântica. Isso me ajudou a perceber que RAG não deve ser usado para tudo. Por exemplo, para saber "quantos chamados aconteceram" ou "qual é o status atual" de alguma informação, faz mais sentido consultar diretamente uma API ou banco de dados do que tentar recuperar essa informação por meio de documentos.

Também usei a IA para questionar minhas próprias escolhas. Em vez de pensar apenas em "qual tecnologia é melhor?", fui analisando o contexto de cada problema: frequência de atualização dos documentos, quantidade de dados, risco de erro, necessidade de privacidade, tipo de usuário e natureza da informação. Isso foi importante principalmente na escolha dos embeddings. No cenário de suporte de TI, considerei o `text-embedding-3-small` por custo e praticidade. Já no cenário de contratos, a preocupação com dados pessoais e LGPD fez com que a opção por um modelo local, como o `bge-m3`, fizesse mais sentido. A IA ajudou a comparar essas possibilidades, mas a decisão final foi baseada nas características que defini para cada cenário.

Outra utilização foi na revisão do chunking. Eu já sabia que dividir os documentos em pedaços era necessário, mas a atividade me fez perceber que não existe um tamanho de chunk universalmente correto. A IA ajudou a discutir os impactos de chunks muito pequenos ou muito grandes e, a partir disso, consegui relacionar a estratégia à estrutura dos documentos. No suporte de TI, faz sentido preservar os passos dos procedimentos; nos contratos, preservar a cláusula inteira é muito mais importante, principalmente por causa de exceções e condicionais que podem mudar completamente o significado de uma obrigação.

Também utilizei IA como uma espécie de "segunda opinião" para identificar riscos que poderiam passar despercebidos. Isso apareceu, por exemplo, na preocupação com OCR em contratos antigos, com documentos desatualizados, com filtros de metadados e com o perigo de usar RAG para fazer contagens ou cálculos financeiros. Em vez de considerar apenas se a arquitetura funcionaria tecnicamente, passei a pensar também em como ela poderia falhar e qual seria a consequência desse erro.

As referências técnicas também serviram para confirmar informações específicas sobre os modelos de embedding e suas características. Nesse caso, procurei não depender apenas da resposta da IA e incluí as fontes utilizadas no trabalho, principalmente para informações de modelo, dimensões, custos e características técnicas.

De modo geral, considero que a IA funcionou mais como uma "parceira de estudo e revisão" do que como uma substituta do meu raciocínio. Eu usei as respostas para comparar alternativas, fazer perguntas, perceber pontos fracos e melhorar a organização das ideias. Algumas decisões surgiram justamente desse processo de questionar "e se isso der errado?" ou "será que RAG realmente é a melhor solução para esse caso?".

A parte mais importante para mim foi perceber que projetar uma aplicação RAG não significa simplesmente colocar documentos em um banco vetorial e fazer perguntas para um LLM. É preciso pensar em ingestão, limpeza, metadados, chunking, embeddings, recuperação, fontes estruturadas, atualização dos documentos, privacidade e, principalmente, nos tipos de erro que podem acontecer. A IA me ajudou a enxergar essas conexões e a aprofundar o raciocínio, mas as escolhas apresentadas no trabalho foram construídas a partir da análise dos dois cenários e das justificativas que desenvolvi para cada um.
