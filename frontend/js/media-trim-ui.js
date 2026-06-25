(function (global) {
  "use strict";

  var DURATION_THRESHOLD_SEC = 300;
  var SIZE_THRESHOLD_MB = 50;
  /** Cortar localmente se o ficheiro for grande e o trecho for claramente mais curto. */
  var CLIENT_TRIM_MIN_MB = 50;
  var CLIENT_TRIM_MAX_SEGMENT_RATIO = 0.85;
  var CLIENT_TRIM_LARGE_MB = 120;
  /** Acima disto o FFmpeg.wasm tende a falhar — usar gravação via <video>/<audio>. */
  var WASM_TRIM_MAX_BYTES = 150 * 1024 * 1024;

  var state = {
    file: null,
    objectUrl: null,
    isVideo: false,
    duration: 0,
    startSec: 0,
    endSec: 0,
    mode: "full",
    visible: false,
    maxFileMb: 500,
  };

  function $(id) {
    return document.getElementById(id);
  }

  function formatTime(sec) {
    sec = Math.max(0, Math.round(sec));
    var h = Math.floor(sec / 3600);
    var m = Math.floor((sec % 3600) / 60);
    var s = sec % 60;
    if (h > 0) {
      return h + ":" + String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
    }
    return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
  }

  function formatDuration(sec) {
    sec = Math.max(0, Math.round(sec));
    if (sec < 60) return sec + " s";
    var m = Math.round(sec / 60);
    if (m < 60) return m + " min";
    var h = Math.floor(m / 60);
    var rm = m % 60;
    return h + " h" + (rm ? " " + rm + " min" : "");
  }

  function fileSizeMb(file) {
    return file.size / (1024 * 1024);
  }

  function shouldOfferTrim(durationSec, sizeMb) {
    return durationSec > DURATION_THRESHOLD_SEC || sizeMb > SIZE_THRESHOLD_MB;
  }

  function isOverUploadLimit(file) {
    return file && fileSizeMb(file) > state.maxFileMb;
  }

  function ensurePanel() {
    var panel = $("oeTrimPanel");
    if (panel) return panel;
    panel = document.createElement("section");
    panel.id = "oeTrimPanel";
    panel.className = "oe-trim-panel hidden";
    panel.setAttribute("aria-label", "Escolher trecho a transcrever");
    panel.innerHTML =
      '<div class="oe-trim-panel__head">' +
      '<h2 class="oe-trim-panel__title">Ficheiro longo — o que queres transcrever?</h2>' +
      '<p class="oe-trim-panel__meta" id="oeTrimMeta"></p>' +
      "</div>" +
      '<div class="oe-trim-mode" role="radiogroup" aria-label="Modo de transcrição">' +
      '<label class="oe-trim-mode__opt"><input type="radio" name="oeTrimMode" value="full" checked> Ficheiro completo</label>' +
      '<label class="oe-trim-mode__opt"><input type="radio" name="oeTrimMode" value="segment"> Só um trecho</label>' +
      "</div>" +
      '<p class="oe-trim-panel__note hidden" id="oeTrimForceNote"></p>' +
      '<div class="oe-trim-segment hidden" id="oeTrimSegment">' +
      '<div class="oe-trim-preview-wrap">' +
      '<video id="oeTrimVideo" class="oe-trim-preview hidden" controls playsinline preload="metadata"></video>' +
      '<audio id="oeTrimAudio" class="oe-trim-preview hidden" controls preload="metadata"></audio>' +
      "</div>" +
      '<div class="oe-trim-timeline">' +
      '<div class="oe-trim-timeline__labels"><span id="oeTrimStartLabel">00:00</span><span id="oeTrimEndLabel">00:00</span></div>' +
      '<div class="oe-trim-timeline__track">' +
      '<div class="oe-trim-timeline__fill" id="oeTrimFill"></div>' +
      '<input type="range" id="oeTrimStart" class="oe-trim-timeline__range oe-trim-timeline__range--start" min="0" max="1000" value="0" step="1" aria-label="Início do trecho">' +
      '<input type="range" id="oeTrimEnd" class="oe-trim-timeline__range oe-trim-timeline__range--end" min="0" max="1000" value="1000" step="1" aria-label="Fim do trecho">' +
      "</div>" +
      "</div>" +
      '<div class="oe-trim-presets">' +
      '<button type="button" class="oe-trim-preset" data-sec="900">Primeiros 15 min</button>' +
      '<button type="button" class="oe-trim-preset" data-sec="1800">Primeiros 30 min</button>' +
      '<button type="button" class="oe-trim-preset" data-sec="3600">Primeira hora</button>' +
      "</div>" +
      '<p class="oe-trim-summary" id="oeTrimSummary"></p>' +
      '<p class="oe-trim-panel__note">Com um trecho escolhido, extraímos o áudio no teu dispositivo antes do envio — em MP4 costuma levar segundos a poucos minutos, não o tempo inteiro do trecho.</p>' +
      '<button type="button" class="oe-trim-play" id="oeTrimPlay">▶ Ouvir / ver trecho</button>' +
      "</div>";
    var anchor = $("videoPreviewWrap") || $("dropZone");
    if (anchor && anchor.parentNode) {
      anchor.parentNode.insertBefore(panel, anchor.nextSibling);
    } else {
      var form = $("uploadForm");
      if (form) form.insertBefore(panel, form.firstChild);
    }
    bindPanelEvents(panel);
    return panel;
  }

  function getMediaEl() {
    return state.isVideo ? $("oeTrimVideo") : $("oeTrimAudio");
  }

  function updateFill() {
    var fill = $("oeTrimFill");
    if (!fill || !state.duration) return;
    var left = (state.startSec / state.duration) * 100;
    var width = ((state.endSec - state.startSec) / state.duration) * 100;
    fill.style.left = left + "%";
    fill.style.width = Math.max(0, width) + "%";
  }

  function updateSummary() {
    var summary = $("oeTrimSummary");
    var startLabel = $("oeTrimStartLabel");
    var endLabel = $("oeTrimEndLabel");
    if (startLabel) startLabel.textContent = formatTime(state.startSec);
    if (endLabel) endLabel.textContent = formatTime(state.endSec);
    if (summary) {
      var len = Math.max(0, state.endSec - state.startSec);
      summary.textContent =
        "Trecho: " +
        formatTime(state.startSec) +
        " → " +
        formatTime(state.endSec) +
        " (" +
        formatDuration(len) +
        ")";
    }
    updateFill();
    syncRangeInputs();
    global.dispatchEvent(new CustomEvent("oe-trim-change", { detail: getSelection() }));
  }

  function syncRangeInputs() {
    var max = Math.max(1, Math.round(state.duration * 10));
    var start = $("oeTrimStart");
    var end = $("oeTrimEnd");
    if (!start || !end) return;
    start.max = String(max);
    end.max = String(max);
    start.value = String(Math.round(state.startSec * 10));
    end.value = String(Math.round(state.endSec * 10));
  }

  function readRanges() {
    var start = $("oeTrimStart");
    var end = $("oeTrimEnd");
    if (!start || !end || !state.duration) return;
    stopSegmentPreview();
    state.startSec = Math.min(parseInt(start.value, 10) / 10, state.duration);
    state.endSec = Math.min(parseInt(end.value, 10) / 10, state.duration);
    if (state.endSec <= state.startSec + 1) {
      state.endSec = Math.min(state.duration, state.startSec + 60);
      end.value = String(Math.round(state.endSec * 10));
    }
    updateSummary();
  }

  function setMode(mode) {
    stopSegmentPreview();
    state.mode = mode;
    var segment = $("oeTrimSegment");
    var radios = document.querySelectorAll('input[name="oeTrimMode"]');
    radios.forEach(function (r) {
      r.checked = r.value === mode;
    });
    if (segment) segment.classList.toggle("hidden", mode !== "segment");
    updateForceNote();
    global.dispatchEvent(new CustomEvent("oe-trim-change", { detail: getSelection() }));
  }

  function updateForceNote() {
    var note = $("oeTrimForceNote");
    if (!note || !state.file) return;
    if (isOverUploadLimit(state.file)) {
      note.textContent =
        "Este ficheiro passa o limite de " +
        state.maxFileMb +
        " MB — escolhe «Só um trecho». Cortamos no teu browser antes de enviar.";
      note.classList.remove("hidden");
      if (state.mode === "full") setMode("segment");
    } else if (isOverUploadLimit(state.file) && state.mode === "segment") {
      note.textContent =
        "Vamos cortar o trecho no teu browser antes de enviar (ficheiro acima do limite).";
      note.classList.remove("hidden");
    } else {
      note.classList.add("hidden");
    }
  }

  function applyPreset(seconds) {
    if (!state.duration) return;
    stopSegmentPreview();
    state.startSec = 0;
    state.endSec = Math.min(state.duration, seconds);
    updateSummary();
  }

  function bindPanelEvents(panel) {
    panel.querySelectorAll('input[name="oeTrimMode"]').forEach(function (radio) {
      radio.addEventListener("change", function () {
        if (radio.checked) setMode(radio.value);
      });
    });
    var start = $("oeTrimStart");
    var end = $("oeTrimEnd");
    if (start) start.addEventListener("input", readRanges);
    if (end) end.addEventListener("input", readRanges);
    panel.querySelectorAll(".oe-trim-preset").forEach(function (btn) {
      btn.addEventListener("click", function () {
        applyPreset(parseInt(btn.getAttribute("data-sec"), 10) || 900);
      });
    });
    var playBtn = $("oeTrimPlay");
    if (playBtn) {
      playBtn.addEventListener("click", function () {
        if (isSegmentPreviewPlaying()) {
          stopSegmentPreview();
        } else {
          playSegmentPreview();
        }
      });
    }
  }

  function showPanel(file, durationSec, objectUrl, isVideo) {
    var panel = ensurePanel();
    state.file = file;
    state.objectUrl = objectUrl;
    state.isVideo = isVideo;
    state.duration = durationSec || 0;
    state.startSec = 0;
    state.endSec = state.duration || 0;
    state.visible = true;

    var meta = $("oeTrimMeta");
    if (meta) {
      meta.textContent =
        file.name +
        " · " +
        formatDuration(state.duration) +
        " · " +
        Math.round(fileSizeMb(file)) +
        " MB";
    }

    var video = $("oeTrimVideo");
    var audio = $("oeTrimAudio");
    if (video && audio) {
      video.classList.toggle("hidden", !isVideo);
      audio.classList.toggle("hidden", isVideo);
      var media = isVideo ? video : audio;
      media.src = objectUrl || "";
      media.load();
      bindPreviewMediaEvents();
    }

    panel.classList.remove("hidden");
    if (isOverUploadLimit(file)) {
      setMode("segment");
    } else {
      setMode("full");
    }
    updateSummary();
    updateForceNote();
  }

  function hidePanel() {
    stopSegmentPreview();
    var panel = $("oeTrimPanel");
    if (panel) panel.classList.add("hidden");
    state.file = null;
    state.visible = false;
    var video = $("oeTrimVideo");
    var audio = $("oeTrimAudio");
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
  }

  function getSelection() {
    return {
      mode: state.mode,
      startSec: state.startSec,
      endSec: state.endSec,
      duration: state.duration,
      requiresSegment: state.file ? isOverUploadLimit(state.file) : false,
      isValid:
        state.mode === "full" ||
        (state.endSec > state.startSec + 1 && state.duration > 0),
    };
  }

  function appendToFormData(formData) {
    var sel = getSelection();
    if (sel.mode === "segment" && sel.isValid) {
      formData.append("trim_start_sec", String(sel.startSec));
      formData.append("trim_end_sec", String(sel.endSec));
    }
  }

  function canUploadFile(file) {
    if (!file) return false;
    if (!state.visible || !isSameFile(state.file, file)) return !isOverUploadLimit(file);
    var sel = getSelection();
    if (isOverUploadLimit(file) && sel.mode !== "segment") return false;
    if (sel.mode === "segment" && !sel.isValid) return false;
    return true;
  }

  function uploadBlockReason(file) {
    if (!file) return "Nenhum ficheiro selecionado.";
    if (!canUploadFile(file)) {
      if (isOverUploadLimit(file)) {
        return "Ficheiro acima de " + state.maxFileMb + " MB — escolhe um trecho mais curto.";
      }
      return "Ajusta o início e fim do trecho.";
    }
    return "";
  }

  var ffmpegCache = null;
  var ffmpegLoadPromise = null;
  var previewTimeListener = null;

  var PREVIEW_LABEL_PLAY = "▶ Ouvir / ver trecho";
  var PREVIEW_LABEL_STOP = "⏹ Parar";

  function stopSegmentPreview() {
    var media = getMediaEl();
    var playBtn = $("oeTrimPlay");
    if (media) {
      media.pause();
      if (previewTimeListener) {
        media.removeEventListener("timeupdate", previewTimeListener);
        previewTimeListener = null;
      }
    }
    if (playBtn) {
      playBtn.textContent = PREVIEW_LABEL_PLAY;
      playBtn.setAttribute("aria-pressed", "false");
    }
  }

  function isSegmentPreviewPlaying() {
    var media = getMediaEl();
    return !!(media && !media.paused && !media.ended);
  }

  function playSegmentPreview() {
    var media = getMediaEl();
    var playBtn = $("oeTrimPlay");
    if (!media) return;
    stopSegmentPreview();
    media.currentTime = state.startSec;
    var playPromise = media.play();
    if (playBtn) {
      playBtn.textContent = PREVIEW_LABEL_STOP;
      playBtn.setAttribute("aria-pressed", "true");
    }
    previewTimeListener = function () {
      if (media.currentTime >= state.endSec - 0.05) {
        stopSegmentPreview();
      }
    };
    media.addEventListener("timeupdate", previewTimeListener);
    if (playPromise && playPromise.catch) {
      playPromise.catch(function () {
        stopSegmentPreview();
      });
    }
  }

  function bindPreviewMediaEvents() {
    ["oeTrimVideo", "oeTrimAudio"].forEach(function (id) {
      var media = $(id);
      if (!media || media.dataset.oeTrimBound) return;
      media.dataset.oeTrimBound = "1";
      media.addEventListener("pause", function () {
        if ($("oeTrimPlay") && $("oeTrimPlay").getAttribute("aria-pressed") === "true") {
          stopSegmentPreview();
        }
      });
      media.addEventListener("ended", stopSegmentPreview);
    });
  }

  async function loadFfmpeg(onProgress) {
    if (ffmpegCache) return ffmpegCache;
    if (ffmpegLoadPromise) return ffmpegLoadPromise;

    ffmpegLoadPromise = (async function () {
      onProgress = onProgress || function () {};
      onProgress("A carregar ferramenta de corte…");
      var ffmpegMod = await import(
        "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js"
      );
      var utilMod = await import(
        "https://cdn.jsdelivr.net/npm/@ffmpeg/util@0.12.1/dist/esm/index.js"
      );
      var FFmpeg = ffmpegMod.FFmpeg;
      var fetchFile = utilMod.fetchFile;
      var toBlobURL = utilMod.toBlobURL;
      var ffmpeg = new FFmpeg();
      ffmpeg.on("progress", function (ev) {
        var pct = ev && ev.progress != null ? Math.round(ev.progress * 100) : 0;
        onProgress("A cortar no browser… " + pct + "%");
      });
      var coreBase = "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.6/dist/esm";
      var pkgBase = "https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.10/dist/esm";
      await ffmpeg.load({
        coreURL: await toBlobURL(coreBase + "/ffmpeg-core.js", "text/javascript"),
        wasmURL: await toBlobURL(coreBase + "/ffmpeg-core.wasm", "application/wasm"),
        workerURL: await toBlobURL(pkgBase + "/worker.js", "text/javascript"),
      });
      ffmpegCache = ffmpeg;
      ffmpeg._fetchFile = fetchFile;
      return ffmpeg;
    })();

    try {
      return await ffmpegLoadPromise;
    } catch (err) {
      ffmpegLoadPromise = null;
      throw err;
    }
  }

  function segmentDurationSec(sel) {
    if (!sel || sel.mode !== "segment") return 0;
    return Math.max(0, sel.endSec - sel.startSec);
  }

  function shouldTrimClientSide(file, sel) {
    if (!file || !sel || sel.mode !== "segment" || !sel.isValid) return false;
    if (isOverUploadLimit(file)) return true;
    var sizeMb = fileSizeMb(file);
    var segmentSec = segmentDurationSec(sel);
    var totalSec = state.duration || segmentSec || 1;
    var ratio = segmentSec > 0 ? segmentSec / totalSec : 1;
    if (sizeMb >= CLIENT_TRIM_MIN_MB && ratio < CLIENT_TRIM_MAX_SEGMENT_RATIO) return true;
    if (sizeMb >= CLIENT_TRIM_LARGE_MB && ratio < 0.95) return true;
    return false;
  }

  function isSameFile(a, b) {
    return !!a && !!b && a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
  }

  function pickRecorderMime() {
    var types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    for (var i = 0; i < types.length; i++) {
      if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(types[i])) {
        return types[i];
      }
    }
    return "";
  }

  function mimeToAudioExt(mime) {
    if (!mime) return "webm";
    if (mime.indexOf("ogg") >= 0) return "ogg";
    if (mime.indexOf("mp4") >= 0 || mime.indexOf("m4a") >= 0) return "m4a";
    return "webm";
  }

  function isMp4LikeFile(file) {
    if (!file) return false;
    var ext = (file.name.split(".").pop() || "").toLowerCase();
    if (ext === "mp4" || ext === "m4v" || ext === "mov") return true;
    var type = (file.type || "").toLowerCase();
    return type.indexOf("mp4") >= 0 || type.indexOf("quicktime") >= 0;
  }

  var avCliperLoadPromise = null;

  function loadAvCliper() {
    if (avCliperLoadPromise) return avCliperLoadPromise;
    avCliperLoadPromise = import("https://esm.sh/@webav/av-cliper@1.2.8").catch(function (err) {
      avCliperLoadPromise = null;
      throw err;
    });
    return avCliperLoadPromise;
  }

  async function readReadableStreamToBlob(stream, mime) {
    var reader = stream.getReader();
    var parts = [];
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      if (chunk.value) parts.push(chunk.value);
    }
    return new Blob(parts, { type: mime || "application/octet-stream" });
  }

  function destroyClipSafe(clip) {
    if (clip && typeof clip.destroy === "function") {
      try {
        clip.destroy();
      } catch (_) {}
    }
  }

  function pickAudioExportClip(tracks, fallback) {
    if (!tracks || !tracks.length) return fallback;
    for (var i = 0; i < tracks.length; i++) {
      var meta = tracks[i].meta || {};
      if (meta.audioChanCount > 0 && (!meta.width || meta.width <= 2)) return tracks[i];
    }
    for (var j = 0; j < tracks.length; j++) {
      if ((tracks[j].meta || {}).audioChanCount > 0) return tracks[j];
    }
    return fallback;
  }

  /**
   * Corte rápido de MP4/MOV via WebCodecs (sem gravar em tempo real).
   */
  async function extractAudioSegmentViaWebAV(file, startSec, endSec, onProgress) {
    onProgress = onProgress || function () {};
    onProgress("A carregar motor de corte rápido…");
    var mod = await loadAvCliper();
    var MP4Clip = mod.MP4Clip;
    var Combinator = mod.Combinator;
    var OffscreenSprite = mod.OffscreenSprite;
    if (Combinator && Combinator.isSupported) {
      var supported = await Combinator.isSupported();
      if (!supported) throw new Error("WebCodecs indisponível neste browser.");
    }

    onProgress("A analisar vídeo…");
    var sourceClip = new MP4Clip(file.stream());
    await sourceClip.ready;
    var startUs = Math.round(startSec * 1e6);
    var durationUs = Math.round(Math.max(0.5, endSec - startSec) * 1e6);
    var splitHead = await sourceClip.split(startUs);
    var afterStart = splitHead[1];
    destroyClipSafe(sourceClip);
    var splitSegment = await afterStart.split(durationUs);
    var segment = splitSegment[0];
    destroyClipSafe(afterStart);
    destroyClipSafe(splitSegment[1]);

    var exportClip = segment;
    var splitTracks = null;
    try {
      splitTracks = await segment.splitTrack();
      exportClip = pickAudioExportClip(splitTracks, segment);
      if (splitTracks) {
        splitTracks.forEach(function (track) {
          if (track !== exportClip) destroyClipSafe(track);
        });
      }
    } catch (_) {}

    onProgress("A extrair áudio do trecho (rápido)… 0%");
    var sprite = new OffscreenSprite(exportClip);
    var meta = exportClip.meta || {};
    var com = new Combinator({
      width: Math.max(2, meta.width || 2),
      height: Math.max(2, meta.height || 2),
      bitrate: 500000,
    });
    var stopProgress = null;
    if (com.on) {
      stopProgress = com.on("OutputProgress", function (progress) {
        onProgress("A extrair áudio do trecho (rápido)… " + Math.round(progress * 100) + "%");
      });
    }
    try {
      await com.addSprite(sprite, { main: true });
      var outBlob = await readReadableStreamToBlob(com.output(), "audio/mp4");
      if (!outBlob.size) throw new Error("Trecho de áudio vazio.");
      onProgress(
        "Áudio do trecho pronto (" + Math.max(1, Math.round(outBlob.size / (1024 * 1024))) + " MB)."
      );
      var base = (file.name || "media").replace(/\.[^.]+$/, "");
      return new File([outBlob], base + "_trecho.m4a", { type: outBlob.type || "audio/mp4" });
    } finally {
      if (stopProgress) stopProgress();
      com.destroy();
      destroyClipSafe(sprite);
      destroyClipSafe(exportClip);
      if (exportClip !== segment) destroyClipSafe(segment);
    }
  }

  /**
   * Fallback: grava em tempo real via <video>/<audio> (mais lento).
   */
  function extractAudioSegmentViaMedia(startSec, endSec, onProgress) {
    onProgress = onProgress || function () {};
    var media = getMediaEl();
    if (!media) return Promise.reject(new Error("Pré-visualização do vídeo indisponível."));
    var mime = pickRecorderMime();
    if (!mime) {
      return Promise.reject(new Error("O browser não consegue gravar áudio deste vídeo."));
    }
    var segmentSec = Math.max(0.5, endSec - startSec);
    onProgress(
      "A gravar áudio do trecho (~" +
        Math.max(1, Math.ceil(segmentSec / 60)) +
        " min em tempo real — método lento)…"
    );

    return new Promise(function (resolve, reject) {
      stopSegmentPreview();
      var chunks = [];
      var capture =
        typeof media.captureStream === "function"
          ? media.captureStream()
          : typeof media.mozCaptureStream === "function"
            ? media.mozCaptureStream()
            : null;
      if (!capture || !capture.getAudioTracks().length) {
        reject(new Error("Não foi possível capturar áudio do vídeo."));
        return;
      }
      var audioStream = new MediaStream(capture.getAudioTracks());
      var recorder;
      try {
        recorder = new MediaRecorder(audioStream, { mimeType: mime, audioBitsPerSecond: 64000 });
      } catch (err) {
        reject(err);
        return;
      }
      var tickTimer = null;
      var startedAt = 0;

      function cleanup() {
        if (tickTimer) clearInterval(tickTimer);
        tickTimer = null;
        media.pause();
      }

      recorder.ondataavailable = function (ev) {
        if (ev.data && ev.data.size) chunks.push(ev.data);
      };
      recorder.onerror = function () {
        cleanup();
        reject(new Error("Falha ao gravar áudio do trecho."));
      };
      recorder.onstop = function () {
        cleanup();
        var blob = new Blob(chunks, { type: mime });
        if (!blob.size) {
          reject(new Error("Não foi possível extrair áudio do trecho."));
          return;
        }
        onProgress("Áudio do trecho pronto (" + Math.max(1, Math.round(blob.size / (1024 * 1024))) + " MB).");
        var base = (state.file.name || "media").replace(/\.[^.]+$/, "");
        resolve(new File([blob], base + "_trecho." + mimeToAudioExt(mime), { type: mime }));
      };

      function startRecording() {
        chunks = [];
        try {
          recorder.start(400);
        } catch (err) {
          reject(err);
          return;
        }
        startedAt = Date.now();
        tickTimer = setInterval(function () {
          var elapsed = (Date.now() - startedAt) / 1000;
          var pct = Math.min(99, Math.round((elapsed / segmentSec) * 100));
          onProgress("A gravar áudio do trecho… " + pct + "%");
        }, 400);

        function onTimeUpdate() {
          if (media.currentTime >= endSec - 0.08 || media.ended) {
            media.removeEventListener("timeupdate", onTimeUpdate);
            media.pause();
            try {
              if (recorder.state !== "inactive") recorder.stop();
            } catch (_) {}
          }
        }
        media.addEventListener("timeupdate", onTimeUpdate);
        var playPromise = media.play();
        if (playPromise && playPromise.catch) {
          playPromise.catch(function (err) {
            cleanup();
            reject(err);
          });
        }
      }

      media.pause();
      var seekTo = Math.max(0, startSec);
      function afterSeek() {
        startRecording();
      }
      if (Math.abs(media.currentTime - seekTo) < 0.12 && media.readyState >= 2) {
        afterSeek();
        return;
      }
      var onSeeked = function () {
        media.removeEventListener("seeked", onSeeked);
        afterSeek();
      };
      media.addEventListener("seeked", onSeeked);
      try {
        media.currentTime = seekTo;
      } catch (err) {
        media.removeEventListener("seeked", onSeeked);
        reject(err);
      }
    });
  }

  async function trimClientSide(file, startSec, endSec, onProgress, opts) {
    onProgress = onProgress || function () {};
    opts = opts || {};
    if (opts.audioOnly && file.size > WASM_TRIM_MAX_BYTES) {
      if (isMp4LikeFile(file)) {
        try {
          return await extractAudioSegmentViaWebAV(file, startSec, endSec, onProgress);
        } catch (err) {
          console.warn("OuviescreviMediaTrim: WebAV falhou, fallback tempo real", err);
        }
      }
      return extractAudioSegmentViaMedia(startSec, endSec, onProgress);
    }
    var ffmpeg = await loadFfmpeg(onProgress);
    var fetchFile = ffmpeg._fetchFile;
    var ext = (file.name.split(".").pop() || "mp4").toLowerCase();
    var inName = "input." + ext;
    var outName = opts.audioOnly ? "trecho.wav" : "trecho." + ext;
    onProgress(
      opts.audioOnly
        ? "A extrair só o áudio do trecho no browser…"
        : "A cortar o trecho no browser…"
    );
    await ffmpeg.writeFile(inName, await fetchFile(file));
    var args;
    if (opts.audioOnly) {
      args = [
        "-ss",
        String(startSec),
        "-to",
        String(endSec),
        "-i",
        inName,
        "-vn",
        "-sn",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        outName,
      ];
      await ffmpeg.exec(args);
    } else {
      args = ["-ss", String(startSec), "-to", String(endSec), "-i", inName, "-c", "copy", outName];
      try {
        await ffmpeg.exec(args);
      } catch (err) {
        args = [
          "-ss",
          String(startSec),
          "-to",
          String(endSec),
          "-i",
          inName,
          "-c:v",
          "libx264",
          "-preset",
          "ultrafast",
          "-crf",
          "28",
          "-c:a",
          "aac",
          outName,
        ];
        await ffmpeg.exec(args);
      }
    }
    var data = await ffmpeg.readFile(outName);
    var mime = opts.audioOnly ? "audio/wav" : file.type || "application/octet-stream";
    var blob = new Blob([data.buffer], { type: mime });
    var base = (file.name || "media").replace(/\.[^.]+$/, "");
    var outExt = opts.audioOnly ? "wav" : ext;
    return new File([blob], base + "_trecho." + outExt, { type: mime });
  }

  async function prepareForUpload(file, onProgress, opts) {
    opts = opts || {};
    var sel = getSelection();
    if (!state.visible || !isSameFile(state.file, file) || sel.mode === "full") {
      return { file: file, trimmed: false };
    }
    if (!sel.isValid) {
      throw new Error("Trecho inválido — ajusta início e fim.");
    }
    if (shouldTrimClientSide(file, sel)) {
      try {
        var trimmed = await trimClientSide(file, sel.startSec, sel.endSec, onProgress, {
          audioOnly: !!opts.audioOnly,
        });
        return { file: trimmed, trimmed: true };
      } catch (err) {
        console.warn("OuviescreviMediaTrim: corte no browser falhou", err);
        if (isOverUploadLimit(file)) {
          throw new Error(
            "Não foi possível cortar no browser. Tenta um trecho mais curto ou comprime o vídeo antes de enviar."
          );
        }
        onProgress(
          "Corte local falhou — a enviar ficheiro completo (" +
            Math.round(fileSizeMb(file)) +
            " MB). O servidor usa só o trecho (mais lento)."
        );
        return { file: file, trimmed: false, trimStart: sel.startSec, trimEnd: sel.endSec, fallbackServerTrim: true };
      }
    }
    return { file: file, trimmed: false, trimStart: sel.startSec, trimEnd: sel.endSec };
  }

  function onFileSelected(file, opts) {
    opts = opts || {};
    state.maxFileMb =
      global.OuviescreviAPI && global.OuviescreviAPI.getMaxFileSizeMb
        ? global.OuviescreviAPI.getMaxFileSizeMb()
        : 500;
    if (!file) {
      hidePanel();
      return Promise.resolve();
    }
    var sizeMb = fileSizeMb(file);
    var objectUrl = opts.objectUrl || null;
    var isVideo = !!opts.isVideo;

    function tryDurationFromMedia() {
      return new Promise(function (resolve) {
        var el = document.createElement(isVideo ? "video" : "audio");
        el.preload = "metadata";
        el.muted = true;
        if (isVideo) el.playsInline = true;
        var url = objectUrl || URL.createObjectURL(file);
        var revoke = !objectUrl;
        el.src = url;
        el.onloadedmetadata = function () {
          var d = el.duration;
          if (revoke) URL.revokeObjectURL(url);
          resolve(isFinite(d) ? d : 0);
        };
        el.onerror = function () {
          if (revoke) URL.revokeObjectURL(url);
          resolve(0);
        };
      });
    }

    return tryDurationFromMedia().then(function (durationSec) {
      if (!shouldOfferTrim(durationSec, sizeMb) && !isOverUploadLimit(file)) {
        hidePanel();
        return;
      }
      if (!objectUrl) objectUrl = URL.createObjectURL(file);
      showPanel(file, durationSec, objectUrl, isVideo);
    });
  }

  function setMaxFileMb(mb) {
    state.maxFileMb = mb || 500;
    updateForceNote();
  }

  global.OuviescreviMediaTrim = {
    onFileSelected: onFileSelected,
    hide: hidePanel,
    getSelection: getSelection,
    appendToFormData: appendToFormData,
    prepareForUpload: prepareForUpload,
    canUploadFile: canUploadFile,
    uploadBlockReason: uploadBlockReason,
    setMaxFileMb: setMaxFileMb,
    shouldTrimClientSide: shouldTrimClientSide,
  };
})(window);
