# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

FRONT = Path(__file__).resolve().parents[1] / "frontend"

ES = {
    "Ouviescrevi — Transcrição de Áudio y Vídeo com IA Grátis": "Ouviescrevi — Transcripción de audio y vídeo con IA gratis",
    "Pronto! Escolhe o que queres descarregar:": "¡Listo! Elige qué quieres descargar:",
    "🎨 Estilo das legendas & pré-visualização": "🎨 Estilo de subtítulos y vista previa",
    "Queremos evoluir todos os dias. Deixa aqui a tua ideia 👇": "Queremos mejorar cada día. Deja aquí tu idea 👇",
    '"🧩 A montar o quebra-cabeças da tua gravação..."': '"🧩 Montando el puzzle de tu grabación..."',
    '"🧠 A preparar os neurónios para ouvir o ficheiro..."': '"🧠 Preparando la IA para escuchar el archivo..."',
    '"❌ Não foi possível obter a transcrição. Tenta novamente mais tarde."': '"❌ No se pudo obtener la transcripción. Inténtalo más tarde."',
    '"✅ Transcrição concluída com sucesso!"': '"✅ ¡Transcripción completada!"',
    '"⚠️ Não conseguimos transcribir desta vez."': '"⚠️ No pudimos transcribir esta vez."',
    '"⚠️ Esta opção é apenas para ficheiros de vídeo."': '"⚠️ Esta opción es solo para archivos de vídeo."',
    '"❌ Não foi possível gerar as legendas."': '"❌ No se pudieron generar los subtítulos."',
    '"Não foi possível aceder ao microfone. Verifica permissões."': '"No se pudo acceder al micrófono. Revisa los permisos."',
    '"Não foi possível iniciar a gravação neste navegador."': '"No se pudo iniciar la grabación en este navegador."',
    '"🕐 A processar o áudio..."': '"🕐 Procesando el audio..."',
    '"✅ Transcrição concluída!"': '"✅ ¡Transcripción completada!"',
    '"⚠️ Transcrição vazia."': '"⚠️ Transcripción vacía."',
    '"⚠️ Transcrição/Resumo vazio."': '"⚠️ Transcripción/resumen vacío."',
    "`Limite: ${getMaxFileMb()} MB — para ficheiros maiores, extrai só o áudio.`": "`Límite: ${getMaxFileMb()} MB — para archivos más grandes, extrae solo el audio.`",
}

FR = {
    "Ouviescrevi — Transcrição de Áudio et Vídeo com IA Grátis": "Ouviescrevi — Transcription audio et vidéo avec IA gratuite",
    "Pronto! Escolhe o que queres descarregar:": "C’est prêt ! Choisis ce que tu veux télécharger :",
    "🎨 Estilo das legendas & pré-visualização": "🎨 Style des sous-titres et aperçu",
    "Queremos evoluir todos os dias. Deixa aqui a tua ideia 👇": "Nous voulons progresser chaque jour. Laisse ici ton idée 👇",
    '"🧩 A montar o quebra-cabeças da tua gravação..."': '"🧩 Assemblage du puzzle de ton enregistrement..."',
    '"🧠 A preparar os neurónios para ouvir o ficheiro..."': '"🧠 Préparation de l’IA pour écouter le fichier..."',
    '"❌ Não foi possível obter a transcrição. Tenta novamente mais tarde."': '"❌ Impossible d’obtenir la transcription. Réessaie plus tard."',
    '"✅ Transcrição concluída com sucesso!"': '"✅ Transcription terminée !"',
    '"⚠️ Não conseguimos transcrire desta vez."': '"⚠️ Impossible de transcrire cette fois."',
    '"⚠️ Esta opção é apenas para ficheiros de vídeo."': '"⚠️ Cette option est réservée aux fichiers vidéo."',
    '"❌ Não foi possível gerar as legendas."': '"❌ Impossible de générer les sous-titres."',
    '"Não foi possível aceder ao microfone. Verifica permissões."': '"Impossible d’accéder au micro. Vérifie les autorisations."',
    '"Não foi possível iniciar a gravação neste navegador."': '"Impossible de démarrer l’enregistrement dans ce navigateur."',
    '"🕐 A processar o áudio..."': '"🕐 Traitement de l’audio..."',
    '"✅ Transcrição concluída!"': '"✅ Transcription terminée !"',
    '"⚠️ Transcrição vazia."': '"⚠️ Transcription vide."',
    '"⚠️ Transcrição/Resumo vazio."': '"⚠️ Transcription/résumé vide."',
    "`Limite: ${getMaxFileMb()} MB — para ficheiros maiores, extrai só o áudio.`": "`Limite : ${getMaxFileMb()} Mo — pour les gros fichiers, extrais seulement l’audio.`",
}


def apply(path: Path, mapping: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    hit = miss = 0
    for old, new in mapping.items():
        if old in text:
            text = text.replace(old, new)
            hit += 1
        else:
            miss += 1
            print("MISS", path.name, old[:50])
    path.write_text(text, encoding="utf-8")
    print(f"{path}: {hit} replaced, {miss} missed")


if __name__ == "__main__":
    apply(FRONT / "es" / "index.html", ES)
    apply(FRONT / "fr" / "index.html", FR)
