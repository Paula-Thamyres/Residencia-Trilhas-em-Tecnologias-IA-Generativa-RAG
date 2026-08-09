# AULA_03 - Busca Semântica com Embeddings

Notebook organizado seguindo a ordem do enunciado: (1) função de distância euclidiana, (2) função de distância de cosseno, (3) testes com vetores de exemplo, e depois a busca semântica sobre os documentos (linha → parágrafo → capítulo).

## Funções implementadas

- `distancia_euclidiana()`: norma L2 da diferença entre dois vetores
- `similaridade_cosseno()` e `distancia_cosseno()`: baseadas no produto escalar normalizado entre dois vetores

## Testes realizados

- Vetores de exemplo (`embedding_a=[1,0,0]`, `embedding_b=[0,1,0]`, `embedding_c=[1,0,0]`), comparando os 3 pares (A×B, A×C, B×C)
- Termos (gato, felino, cachorro, carro, caminhão, moto, banana, maçã, goiaba), com visualização de proximidade semântica via MDS em 2D e 3D
- Frases de comparação (similar, relacionado, de outro domínio, oposto/negação) contra uma frase âncora

## Busca semântica sobre documentos (.md)

- Divisão dos textos em linha, parágrafo e capítulo/seção
- Geração de embeddings com `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`)
- Cálculo de similaridade de cosseno da query contra cada trecho
- Retorno do TOP 3 resultados mais relevantes, para as perguntas:
  - "O que é Autonomia e opacidade algorítmica?"
  - "O que é o diário de bordo da IA?"
  - "Qual o impacto dos algoritmos nas redes sociais e no espaço público?"

## Como executar

```bash
pip install sentence-transformers scikit-learn matplotlib plotly pandas
```

Depois é só rodar as células do notebook `busca_semantica_embeddings.ipynb` em ordem (ou usar `jupyter nbconvert --to notebook --execute --inplace busca_semantica_embeddings.ipynb` pra rodar tudo de uma vez pelo terminal).