# Aula 05 — Documents, Metadados e Busca Vetorial com LangChain

Esta atividade é a continuação direta da Aula 04 (`atividade_02_chunking_embeddings`). Agora, em vez de montar a estrutura de chunk "na mão" (com `chunk_id`, `text`, `embedding`, `metadata` num dicionário Python qualquer), vamos usar o formato padrão do LangChain para isso: o `Document`.

## O que muda em relação à Aula 04

Na Aula 04 cada chunk era representado assim:

```json
{
  "chunk_id": "doc01_test05_chunk001",
  "text": "Conteúdo do chunk...",
  "embedding": [0.0123, -0.0345, "..."],
  "metadata": { "page": 10, "section": "Introdução" }
}
```

Agora usamos o `Document` do LangChain, que tem **só dois campos**:

```python
Document(
    page_content="texto do chunk",
    metadata={"fonte": "apostila.pdf", "pagina": 10}
)
```

Repare que **não existe campo de embedding dentro do `Document`**. Isso é proposital: o vetor (embedding) é responsabilidade da *vector store* (o banco de dados vetorial), não do documento em si. Se você procurar `doc.embedding` no código, não vai achar — e está certo que não exista.

## O que este pacote contém

- `Aula_05_Documents_Metadados_Busca_Vetorial.ipynb` → o notebook rodado no Google Colab, com o Exercício 1 e o Exercício 2 resolvidos, célula por célula, com explicações em texto entre os códigos.
- Este `README.md`.

## Como rodar (Google Colab)

1. Acesse [colab.research.google.com](https://colab.research.google.com), logado com sua conta Google.
2. `Arquivo` → `Fazer upload de notebook` → selecione o `.ipynb`.
3. Rode célula por célula, de cima para baixo (`Shift + Enter` em cada uma).
4. A primeira célula instala as bibliotecas necessárias (`langchain-core`, `sentence-transformers` etc.) — pode demorar um minutinho.
5. As células de texto entre os códigos explicam o que cada parte faz e respondem as perguntas da atividade.

## Resumo do que foi entregue

**Exercício 1** — lista com 6 `Document` cobrindo os temas *embeddings*, *chunking*, *RAG* e *tokenização*; print de `page_content` e `metadata` de cada um; `len(documentos)`; teste de metadata com lista e com dicionário aninhado (funciona, pois `metadata` aceita qualquer tipo Python — mas vector stores reais costumam só filtrar por valores simples); teste de `Document` sem metadata (não dá erro, o padrão é um dicionário vazio `{}`).

**Exercício 2** — schema de metadados com os 7 campos obrigatórios da atividade (`fonte`, `documento_id`, `chunk_index`, `estrategia`, `chunk_size`, `chunk_overlap`, `n_caracteres`) mais 3 campos próprios (`tema`, `data_processamento`, `hash_conteudo`), cada um com justificativa; exemplo em JSON de um chunk preenchido; respostas sobre qual campo usar para citar a fonte (`fonte` + `documento_id` + `chunk_index`) e por que `chunk_index` é útil (reconstruir o contexto quando o trecho recuperado está cortado no meio de uma explicação).