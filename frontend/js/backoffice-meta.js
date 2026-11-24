/**
 * Backoffice — topo do Painel: pessoas reais, semáforo da meta de fevereiro, percurso
 * (entrou → carregou → terminou → voltou) e reconhecimento deste aparelho como da equipa.
 */
(function (global) {
  var META_VERDE = 10; // visitantes reais por dia até ao fim de fevereiro de 2027
  var META_AMARELO = 3;
  var TEAM_KEY = "oe_team_device"; // token do aparelho; "nao" = o João desmarcou este aparelho

  function $(id) {
    return document.getElementById(id);
  }

  function api() {
    return global.OuviescreviAPI;
  }

  function base() {
    return api().getBase() || api().detectApiBase();
  }

  function setText(id, text) {
    var el = $(id);
    if (el) el.textContent = text;
  }

  function soma(dias, campo) {
    return dias.reduce(function (acc, d) { return acc + (Number(d[campo]) || 0); }, 0);
  }

  function pct(a, b) {
    return b ? " (" + Math.round((a / b) * 100) + "%)" : "";
  }

  function readTeam() {
    try { return localStorage.getItem(TEAM_KEY) || ""; } catch (e) { return ""; }
  }

  function writeTeam(value) {
    try {
      if (value) localStorage.setItem(TEAM_KEY, value);
      else localStorage.removeItem(TEAM_KEY);
    } catch (e) {}
  }

  async function registerDevice(force) {
    var cur = readTeam();
    if (!force && (cur === "nao" || cur)) return;
    if (!(global.OuviescreviAdmin && global.OuviescreviAdmin.roleAtLeast("admin"))) return;
    try {
      var res = await fetch(base() + "/api/admin/team-device", { method: "POST", headers: api().adminAuthHeaders() });
      var data = await res.json().catch(function () { return {}; });
      if (res.ok && data.token) writeTeam(data.token);
    } catch (e) {}
  }

  function renderDevice() {
    var cur = readTeam();
    setText(
      "metaAparelho",
      cur && cur !== "nao"
        ? "✅ Este aparelho está marcado como teu: as tuas visitas não contam, em qualquer rede."
        : "Este aparelho não está marcado como teu: as visitas dele contam como de outras pessoas."
    );
  }

  function render(r) {
    var dias = r.por_dia || [];
    var ult = dias.slice(-7);
    var ant = dias.slice(-14, -7);
    var hoje = dias.length ? dias[dias.length - 1] : {};
    var v7 = soma(ult, "visitantes");
    var media = v7 / 7;
    var google = ult.reduce(function (acc, d) { return acc + (Number((d.origens || {}).Google) || 0); }, 0);
    var sinal = media >= META_VERDE ? "🟢" : media >= META_AMARELO ? "🟡" : "🔴";
    var frase =
      media >= META_VERDE
        ? "Acima da meta de fevereiro."
        : media >= META_AMARELO
          ? "A meio caminho: vê se está a subir de semana para semana."
          : "Ainda abaixo de " + META_AMARELO + " pessoas por dia.";
    setText("metaSinal", sinal);
    setText("metaFrase", frase + " Meta: " + META_VERDE + " pessoas reais por dia até ao fim de fevereiro.");
    setText("metaHoje", String(hoje.visitantes || 0));
    setText("metaHojeSub", (hoje.transcricoes || 0) + " transcrições hoje");
    setText("metaSemana", String(v7));
    setText("metaSemanaSub", "Semana anterior: " + soma(ant, "visitantes"));
    setText("metaMedia", media.toFixed(1).replace(".", ","));
    setText("metaGoogle", String(google));
    var p = r.percurso || {};
    setText(
      "metaPercurso",
      (p.entraram || 0) + " entraram → " +
        (p.carregaram || 0) + " carregaram um ficheiro" + pct(p.carregaram || 0, p.entraram || 0) + " → " +
        (p.terminaram || 0) + " terminaram" + pct(p.terminaram || 0, p.entraram || 0) + " → " +
        (p.voltaram || 0) + " voltaram noutro dia" + pct(p.voltaram || 0, p.entraram || 0)
    );
    renderDevice();
  }

  async function load() {
    if (!$("metaPanel")) return;
    await registerDevice(false);
    try {
      var res = await fetch(base() + "/api/admin/marketing?days=14", { headers: api().adminAuthHeaders() });
      if (!res.ok) throw new Error();
      render(await res.json());
    } catch (e) {
      setText("metaFrase", "Não foi possível carregar os números (a API pode estar a acordar). Carrega em Atualizar.");
    }
  }

  async function sendWeeklyTest() {
    var btn = $("btnWeeklyTest");
    if (btn) btn.disabled = true;
    try {
      var res = await fetch(base() + "/api/admin/weekly-report/test", { method: "POST", headers: api().adminAuthHeaders() });
      var data = await res.json().catch(function () { return {}; });
      if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Falha ao enviar.");
      global.OuviescreviUI.toast("Resumo enviado para o teu email.", "success");
    } catch (e) {
      global.OuviescreviUI.toast(e.message || "Falha ao enviar.", "error");
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    var btn = $("btnWeeklyTest");
    if (btn) btn.addEventListener("click", sendWeeklyTest);
  });

  global.OuviescreviAdminMeta = {
    load: load,
    // Chamado pelos botões «Marcar IP» / «Desmarcar» do Painel.
    markDevice: function (mine) {
      if (mine) {
        writeTeam("");
        return registerDevice(true).then(renderDevice);
      }
      writeTeam("nao");
      renderDevice();
      return Promise.resolve();
    },
  };
})(window);
