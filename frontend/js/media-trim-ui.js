(function (global) {
  "use strict";

  var DURATION_THRESHOLD_SEC = 300;
  var SIZE_THRESHOLD_MB = 50;
  var CLIENT_TRIM_MB = 100;

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

  function needsClientTrim(file) {
    return file && fileSizeMb(file) > CLIENT_TRIM_MB;
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
      '<button type="button" class="oe-trim-play" id="oeTrimPlay" type="button">▶ Ouvir / ver trecho</button>' +
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
    state.startSec = Math.min(parseInt(start.value, 10) / 10, state.duration);
    state.endSec = Math.min(parseInt(end.value, 10) / 10, state.duration);
    if (state.endSec <= state.startSec + 1) {
      state.endSec = Math.min(state.duration, state.startSec + 60);
      end.value = String(Math.round(state.endSec * 10));
    }
    updateSummary();
  }

  function setMode(mode) {
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
    } else if (needsClientTrim(state.file) && state.mode === "segment") {
      note.textContent = "Vamos cortar o trecho no teu browser para acelerar o envio.";
      note.classList.remove("hidden");
    } else {
      note.classList.add("hidden");
    }
  }

  function applyPreset(seconds) {
    if (!state.duration) return;
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
        var media = getMediaEl();
        if (!media) return;
        media.currentTime = state.startSec;
        media.play();
        function onTime() {
          if (media.currentTime >= state.endSec - 0.05) {
            media.pause();
            media.removeEventListener("timeupdate", onTime);
          }
        }
        media.addEventListener("timeupdate", onTime);
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
    if (!state.visible || state.file !== file) return !isOverUploadLimit(file);
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

  async function trimClientSide(file, startSec, endSec, onProgress) {
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
    await ffmpeg.load({
      coreURL: await toBlobURL(coreBase + "/ffmpeg-core.js", "text/javascript"),
      wasmURL: await toBlobURL(coreBase + "/ffmpeg-core.wasm", "application/wasm"),
    });
    var ext = (file.name.split(".").pop() || "mp4").toLowerCase();
    var inName = "input." + ext;
    var outName = "trecho." + ext;
    await ffmpeg.writeFile(inName, await fetchFile(file));
    var args = ["-ss", String(startSec), "-to", String(endSec), "-i", inName, "-c", "copy", outName];
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
    var data = await ffmpeg.readFile(outName);
    var blob = new Blob([data.buffer], { type: file.type || "application/octet-stream" });
    var base = (file.name || "media").replace(/\.[^.]+$/, "");
    return new File([blob], base + "_trecho." + ext, { type: blob.type || file.type });
  }

  async function prepareForUpload(file, onProgress) {
    var sel = getSelection();
    if (!state.visible || state.file !== file || sel.mode === "full") {
      return { file: file, trimmed: false };
    }
    if (!sel.isValid) {
      throw new Error("Trecho inválido — ajusta início e fim.");
    }
    if (needsClientTrim(file) || isOverUploadLimit(file)) {
      var trimmed = await trimClientSide(file, sel.startSec, sel.endSec, onProgress);
      return { file: trimmed, trimmed: true };
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
  };
})(window);
