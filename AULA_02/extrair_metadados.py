from pathlib import Path
import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

load_dotenv("../.env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise Exception(
        "GROQ_API_KEY não encontrada. Verifique o arquivo .env"
    )

client = Groq(api_key=api_key)

PASTA_MD = Path("aula_2")
PASTA_SAIDA = Path("json_saida")
PASTA_SAIDA.mkdir(exist_ok=True)

HEAD_CHARS = 4000
TAIL_CHARS = 2500

SCHEMA_METADADOS = {
    "type": "object",
    "properties": {
        "titulo": {"type": "string"},
        "autores": {
            "type": "array",
            "items": {"type": "string"}
        },
        "ano": {"type": "integer"}
    },
    "required": ["titulo", "autores", "ano"],
    "additionalProperties": False
}


def montar_trecho(texto_completo: str) -> str:
    if len(texto_completo) <= HEAD_CHARS + TAIL_CHARS:
        return texto_completo

    inicio = texto_completo[:HEAD_CHARS]
    fim = texto_completo[-TAIL_CHARS:]
    return f"{inicio}\n\n[... trecho intermediário omitido ...]\n\n{fim}"


def extrair_metadados(caminho_md: Path) -> dict:
    texto_completo = caminho_md.read_text(encoding="utf-8")
    texto = montar_trecho(texto_completo)

    prompt = f"""
Você é um especialista em análise de artigos científicos.

Analise o conteúdo Markdown abaixo e extraia os metadados do artigo,
seguindo rigorosamente o schema fornecido. O trecho pode conter apenas
o início e o fim do documento, com o meio omitido.

Regras:

titulo:
- Deve ser somente o título principal do artigo.

autores:
- Deve conter somente os autores reais do artigo.
- Nunca incluir:
  - universidades;
  - instituições;
  - departamentos;
  - editores;
  - autores citados;
  - referências.

ano:
- É o ano de PUBLICAÇÃO deste artigo específico, não um ano qualquer mencionado no texto.
- Procure primeiro no cabeçalho/topo do documento, onde normalmente aparece junto ao
  nome da revista, volume e local (ex.: "Rev. X vol.34 Brasília 2026" ou "v. 33, e018, 2025").
- Se houver datas de recebido/aprovado/publicado no rodapé ou no fim do artigo
  (ex.: "Aprovado: 27.1.2026"), use a data de APROVAÇÃO ou PUBLICAÇÃO, nunca a de
  submissão/recebimento, como referência do ano.
- Se o cabeçalho da revista e a data de aprovação indicarem anos diferentes,
  priorize o ano que aparece no cabeçalho/masthead da revista (nome da revista + volume).
- NUNCA utilize anos que apareçam em: DOI, referências bibliográficas, citações de
  outros trabalhos, ou intervalos de datas de metodologia.
- Nunca invente ou estime um ano. Se realmente não encontrar nenhuma menção de ano
  de publicação, aprovação ou recebimento no texto, utilize o valor 0.

Conteúdo Markdown:

{texto}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Extraia metadados acadêmicos utilizando saída JSON estruturada."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "metadados_artigo",
                "strict": True,
                "schema": SCHEMA_METADADOS
            }
        }
    )

    resultado = json.loads(response.choices[0].message.content)
    return resultado


def main():
    arquivos_md = list(PASTA_MD.glob("*.md"))

    if not arquivos_md:
        print(f"Nenhum arquivo .md encontrado em '{PASTA_MD}/'.")
        return

    for arquivo in arquivos_md:
        print()
        print("==============================")
        print(f"Processando: {arquivo.name}")

        try:
            dados = extrair_metadados(arquivo)

            print(json.dumps(dados, indent=2, ensure_ascii=False))

            nome_saida = PASTA_SAIDA / f"output_{arquivo.stem}.json"

            with open(nome_saida, "w", encoding="utf-8") as f:
                json.dump(dados, f, indent=2, ensure_ascii=False)

            print(f"Arquivo salvo: {nome_saida}")

        except Exception as erro:
            print("Erro ao processar arquivo:")
            print(erro)

        time.sleep(5)


if __name__ == "__main__":
    main()