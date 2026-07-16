/**
 * Polling partilhado para jobs /transcribe e /video-subs.
 * Expõe window.OuviescreviJobs.
 */
(function (global) {
  "use strict";

  function apiBase() {
    if (global.OuviescreviAPI && typeof global.OuviescreviAPI.getBase === "function") {
      return global.OuviescreviAPI.getBase();
    }
    return "";
  }

  function authHeaders(extra) {
    if (global.OuviescreviAPI && typeof global.OuviescreviAPI.authHeaders === "function") {
      return global.OuviescreviAPI.authHeaders(extra || {});
    }
    return extra || {};
  }

  function sleep(ms) {
    return new Promise(function (r) {
      setTimeout(r, ms);
    });
  }

  /**
   * @param {string} jobId
   * @param {{
   *   endpoint?: string,
   *   statusEl?: HTMLElement|null,
   *   progressBar?: HTMLElement|null,
   *   onTick?: function(object): void,
   *   maxWaitMs?: number,
   *   intervalMs?: number,
   *   progressBase?: number,
   *   progressSpan?: number,
   * }} [opts]
   */
  async function pollJob(jobId, opts) {
    opts = opts || {};
    var endpoint = opts.endpoint || "/transcribe/jobs/";
    var maxWait = opts.maxWaitMs != null ? opts.maxWaitMs : 900000;
    var intervalMs = opts.intervalMs != null ? opts.intervalMs : 2500;
    var progressBase = opts.progressBase != null ? opts.progressBase : 45;
    var progressSpan = opts.progressSpan != null ? opts.progressSpan : 55;
    var started = Date.now();
    var lastProgress = -1;
    var lastProgressAt = Date.now();
    var base = apiBase();

    while (Date.now() - started < maxWait) {
      await sleep(intervalMs);
      var stRes;
      try {
        stRes = await fetch(base + endpoint + encodeURIComponent(jobId), {
          headers: authHeaders(),
        });
      } catch (netErr) {
        if (opts.statusEl && netErr && netErr.message === "Failed to fetch") {
          opts.statusEl.textContent =
            "🔄 Ligação interrompida — a verificar se o servidor ainda processa…";
        }
        continue;
      }
      var st = {};
      try {
        st = await stRes.json();
      } catch (e) {
        st = {};
      }
      if (!stRes.ok) {
        throw new Error(st.detail || st.error || "Erro " + stRes.status);
      }
      if (typeof opts.onTick === "function") {
        try {
          opts.onTick(st);
        } catch (e2) {
          /* ignore */
        }
      }
      var localElapsed = Math.floor((Date.now() - started) / 1000);
      var localFmt =
        Math.floor(localElapsed / 60) +
        ":" +
        String(localElapsed % 60).padStart(2, "0");
      var msg = st.message || "A processar no servidor…";
      if (st.stage_elapsed_sec != null && st.stage_elapsed_sec > 20) {
        msg +=
          " (esta fase: " +
          Math.floor(st.stage_elapsed_sec / 60) +
          ":" +
          String(st.stage_elapsed_sec % 60).padStart(2, "0") +
          ")";
      }
      if (opts.statusEl) {
        opts.statusEl.textContent = "🔄 " + msg + " · tempo total " + localFmt;
      }
      if (st.progress != null && opts.progressBar) {
        var pct = Math.min(
          99,
          progressBase + Math.round((Number(st.progress) || 0) * (progressSpan / 100))
        );
        opts.progressBar.style.width = pct + "%";
        if (pct !== lastProgress) {
          lastProgress = pct;
          lastProgressAt = Date.now();
        } else if (Date.now() - lastProgressAt > 20000 && opts.statusEl) {
          opts.statusEl.textContent =
            "🔄 " + msg + " · tempo total " + localFmt + " — o servidor continua a processar";
        }
      }
      if (st.status === "completed") return st;
      if (st.status === "failed") {
        throw new Error(st.error || st.warning || "Falha no processamento.");
      }
    }
    throw new Error("Tempo limite excedido (15 min). Tenta um trecho mais curto.");
  }

  async function pollTranscribeJob(jobId, opts) {
    opts = opts || {};
    opts.endpoint = "/transcribe/jobs/";
    return pollJob(jobId, opts);
  }

  async function pollVideoSubsJob(jobId, opts) {
    opts = opts || {};
    opts.endpoint = "/video-subs/jobs/";
    return pollJob(jobId, opts);
  }

  /** Locale da UI a partir do path (pt|en|es|fr|de). */
  function detectUiLocale() {
    try {
      var p = (location.pathname || "/").toLowerCase();
      if (p === "/en" || p.indexOf("/en/") === 0) return "en";
      if (p === "/es" || p.indexOf("/es/") === 0) return "es";
      if (p === "/fr" || p.indexOf("/fr/") === 0) return "fr";
      if (p === "/de" || p.indexOf("/de/") === 0) return "de";
    } catch (e) {
      /* ignore */
    }
    return "pt";
  }

  /** Acrescenta ui_locale + page_path a um FormData de upload. */
  function appendAnalyticsFields(formData) {
    if (!formData || typeof formData.append !== "function") return formData;
    try {
      formData.append("ui_locale", detectUiLocale());
      formData.append("page_path", location.pathname || "/");
    } catch (e) {
      /* ignore */
    }
    return formData;
  }

  /**
   * Se a resposta tiver job_id, faz poll até completed.
   * Caso contrário devolve data tal como veio.
   */
  async function awaitTranscribeResult(data, opts) {
    if (data && data.job_id) {
      return pollTranscribeJob(data.job_id, opts);
    }
    return data;
  }

  async function awaitVideoSubsResult(data, opts) {
    if (data && data.job_id) {
      return pollVideoSubsJob(data.job_id, opts);
    }
    return data;
  }

  /**
   * Diarização via API (GPT). Fallback local se falhar.
   * @param {string} text
   * @param {{ names?: string[], lang?: string }} [opts]
   */
  async function diarizeSpeakers(text, opts) {
    opts = opts || {};
    var names = opts.names || ["Speaker 1", "Speaker 2"];
    var fallback = function () {
      return applyAlternatingSpeakers(text, names);
    };
    if (!text || !String(text).trim()) return text;
    try {
      if (global.OuviescreviAPI && typeof global.OuviescreviAPI.init === "function") {
        await global.OuviescreviAPI.init();
      }
      var res = await fetch(apiBase() + "/api/diarize", {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify(
          global.OuviescreviAPI && typeof global.OuviescreviAPI.authJson === "function"
            ? global.OuviescreviAPI.authJson({
                text: text,
                names: names,
                language: opts.lang || detectUiLocale(),
              })
            : { text: text, names: names, language: opts.lang || detectUiLocale() }
        ),
      });
      var data = await res.json().catch(function () {
        return {};
      });
      if (!res.ok) throw new Error(data.detail || data.error || "diarize failed");
      if (data.text && String(data.text).trim()) return data.text;
    } catch (e) {
      /* fallback */
    }
    return fallback();
  }

  function applyAlternatingSpeakers(transcricao, names) {
    names = names && names.length ? names : ["Speaker 1", "Speaker 2"];
    var linhas = String(transcricao || "").split("\n");
    var resultado = "";
    var idx = 0;
    for (var i = 0; i < linhas.length; i++) {
      var linha = linhas[i].trim();
      if (!linha) continue;
      var nova = linha.replace(/^\[(\d{2}:\d{2})\]/, function (m, ts) {
        var nome = names[idx % names.length];
        idx += 1;
        return "[" + ts + "] " + nome + ":";
      });
      resultado += nova + "\n";
    }
    return resultado.trim();
  }

  function defaultSpeakerNames(lang) {
    var map = {
      pt: ["João", "Maria"],
      en: ["John", "Mary"],
      es: ["Juan", "María"],
      fr: ["Jean", "Marie"],
      de: ["Hans", "Maria"],
    };
    return map[lang] || map.pt;
  }

  global.OuviescreviJobs = {
    pollJob: pollJob,
    pollTranscribeJob: pollTranscribeJob,
    pollVideoSubsJob: pollVideoSubsJob,
    awaitTranscribeResult: awaitTranscribeResult,
    awaitVideoSubsResult: awaitVideoSubsResult,
    appendAnalyticsFields: appendAnalyticsFields,
    detectUiLocale: detectUiLocale,
    diarizeSpeakers: diarizeSpeakers,
    applyAlternatingSpeakers: applyAlternatingSpeakers,
    defaultSpeakerNames: defaultSpeakerNames,
  };
})(typeof window !== "undefined" ? window : globalThis);
