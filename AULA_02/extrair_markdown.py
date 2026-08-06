from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

PASTA_PDFS = Path("aula_2")
PASTA_MD = Path("aula_2")
PASTA_MD.mkdir(exist_ok=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False

conversor = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def converter_pdf(caminho_pdf: Path) -> None:
    resultado = conversor.convert(str(caminho_pdf))
    markdown = resultado.document.export_to_markdown()

    caminho_md = PASTA_MD / f"{caminho_pdf.stem}.md"
    caminho_md.write_text(markdown, encoding="utf-8")

    print(f"Convertido: {caminho_pdf.name} -> {caminho_md}")


def main():
    arquivos_pdf = list(PASTA_PDFS.glob("*.pdf"))

    if not arquivos_pdf:
        print(f"Nenhum arquivo .pdf encontrado em '{PASTA_PDFS}/'.")
        return

    for arquivo in arquivos_pdf:
        print()
        print("==============================")
        print(f"Processando: {arquivo.name}")

        try:
            converter_pdf(arquivo)
        except Exception as erro:
            print("Erro ao processar arquivo:")
            print(erro)


if __name__ == "__main__":
    main()