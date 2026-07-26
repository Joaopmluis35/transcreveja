# -*- coding: utf-8 -*-
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "frontend"

ES = {
    "Convert & Download": "Convertir y descargar",
    "Extrair Texto": "Extraer texto",
    "Conversor Imagem": "Convertir imagen",
    "Conversor para Word": "Convertir a Word",
    "Conversor e Descarregar": "Convertir y descargar",
    "Por favor, seleciona um ficheiro .docx.": "Selecciona un archivo .docx.",
    "Por favor, seleciona um ficheiro .pdf.": "Selecciona un archivo .pdf.",
    "📥 A processar ficheiro...": "📥 Procesando archivo...",
    "ficheiro_convertido.pdf": "archivo_convertido.pdf",
    "✅ Conversão concluída!": "✅ ¡Conversión completada!",
    "⚠️ Erro ao converter o ficheiro.": "⚠️ Error al convertir el archivo.",
    "texto_extraido.txt": "texto_extraido.txt",
    "✅ Texto extraído com sucesso!": "✅ ¡Texto extraído!",
    "⚠️ Erro ao processar o PDF.": "⚠️ Error al procesar el PDF.",
    "Por favor, seleciona uma imagem.": "Selecciona una imagen.",
    "imagem_convertida.pdf": "imagen_convertida.pdf",
    "⚠️ Erro ao converter a imagem.": "⚠️ Error al convertir la imagen.",
    "documento.docx": "documento.docx",
    "⚠️ Erro ao converter para Word.": "⚠️ Error al convertir a Word.",
}

FR = {
    "Convert & Download": "Convertir et télécharger",
    "Extrair Texto": "Extraire le texte",
    "Conversor Imagem": "Convertir l'image",
    "Conversor para Word": "Convertir en Word",
    "Conversor e Descarregar": "Convertir et télécharger",
    "Por favor, seleciona um ficheiro .docx.": "Choisis un fichier .docx.",
    "Por favor, seleciona um ficheiro .pdf.": "Choisis un fichier .pdf.",
    "📥 A processar ficheiro...": "📥 Traitement du fichier...",
    "ficheiro_convertido.pdf": "fichier_converti.pdf",
    "✅ Conversão concluída!": "✅ Conversion terminée !",
    "⚠️ Erro ao converter o ficheiro.": "⚠️ Erreur lors de la conversion.",
    "texto_extraido.txt": "texte_extrait.txt",
    "✅ Texto extraído com sucesso!": "✅ Texte extrait !",
    "⚠️ Erro ao processar o PDF.": "⚠️ Erreur lors du traitement du PDF.",
    "Por favor, seleciona uma imagem.": "Choisis une image.",
    "imagem_convertida.pdf": "image_convertie.pdf",
    "⚠️ Erro ao converter a imagem.": "⚠️ Erreur lors de la conversion de l'image.",
    "documento.docx": "document.docx",
    "⚠️ Erro ao converter para Word.": "⚠️ Erreur lors de la conversion en Word.",
}


def apply(path: Path, mapping: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    n = 0
    for old, new in mapping.items():
        if old in text:
            text = text.replace(old, new)
            n += 1
    path.write_text(text, encoding="utf-8")
    print(path, n)


if __name__ == "__main__":
    apply(ROOT / "es" / "conversor.html", ES)
    apply(ROOT / "fr" / "conversor.html", FR)
