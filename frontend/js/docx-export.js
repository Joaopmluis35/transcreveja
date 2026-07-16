/**
 * Exportação DOCX real (OOXML mínimo no browser + endpoint Pro).
 * Expõe window.OuviescreviDocx.
 */
(function (global) {
  "use strict";

  function crc32(buf) {
    var table = crc32._t;
    if (!table) {
      table = new Uint32Array(256);
      for (var n = 0; n < 256; n++) {
        var c = n;
        for (var k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c;
      }
      crc32._t = table;
    }
    var crc = 0 ^ -1;
    for (var i = 0; i < buf.length; i++) crc = (crc >>> 8) ^ table[(crc ^ buf[i]) & 0xff];
    return (crc ^ -1) >>> 0;
  }

  function strToU8(s) {
    if (typeof TextEncoder !== "undefined") return new TextEncoder().encode(s);
    var arr = new Uint8Array(s.length);
    for (var i = 0; i < s.length; i++) arr[i] = s.charCodeAt(i) & 0xff;
    return arr;
  }

  function u32(n) {
    return new Uint8Array([n & 255, (n >>> 8) & 255, (n >>> 16) & 255, (n >>> 24) & 255]);
  }

  function u16(n) {
    return new Uint8Array([n & 255, (n >>> 8) & 255]);
  }

  function concat(chunks) {
    var len = 0;
    for (var i = 0; i < chunks.length; i++) len += chunks[i].length;
    var out = new Uint8Array(len);
    var off = 0;
    for (var j = 0; j < chunks.length; j++) {
      out.set(chunks[j], off);
      off += chunks[j].length;
    }
    return out;
  }

  function zipStore(files) {
    var localParts = [];
    var centralParts = [];
    var offset = 0;
    for (var i = 0; i < files.length; i++) {
      var name = strToU8(files[i].name);
      var data = files[i].data instanceof Uint8Array ? files[i].data : strToU8(files[i].data);
      var crc = crc32(data);
      var local = concat([
        u32(0x04034b50),
        u16(20),
        u16(0),
        u16(0),
        u16(0),
        u16(0),
        u32(crc),
        u32(data.length),
        u32(data.length),
        u16(name.length),
        u16(0),
        name,
        data,
      ]);
      localParts.push(local);
      centralParts.push(
        concat([
          u32(0x02014b50),
          u16(20),
          u16(20),
          u16(0),
          u16(0),
          u16(0),
          u16(0),
          u32(crc),
          u32(data.length),
          u32(data.length),
          u16(name.length),
          u16(0),
          u16(0),
          u16(0),
          u16(0),
          u32(0),
          u32(offset),
          name,
        ])
      );
      offset += local.length;
    }
    var central = concat(centralParts);
    var end = concat([
      u32(0x06054b50),
      u16(0),
      u16(0),
      u16(files.length),
      u16(files.length),
      u32(central.length),
      u32(offset),
      u16(0),
    ]);
    return concat(localParts.concat([central, end]));
  }

  function xmlEscape(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function buildDocumentXml(text, title) {
    var paras = String(text || "").split(/\r?\n/);
    var body = [];
    if (title) {
      body.push(
        "<w:p><w:pPr><w:pStyle w:val=\"Title\"/></w:pPr><w:r><w:t>" +
          xmlEscape(title) +
          "</w:t></w:r></w:p>"
      );
    }
    for (var i = 0; i < paras.length; i++) {
      var t = paras[i];
      if (!t) {
        body.push("<w:p/>");
      } else {
        body.push("<w:p><w:r><w:t xml:space=\"preserve\">" + xmlEscape(t) + "</w:t></w:r></w:p>");
      }
    }
    return (
      '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
      '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">' +
      "<w:body>" +
      body.join("") +
      '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/></w:sectPr>' +
      "</w:body></w:document>"
    );
  }

  function buildDocxBytes(text, title) {
    var contentTypes =
      '<?xml version="1.0" encoding="UTF-8"?>' +
      '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
      '<Default Extension="xml" ContentType="application/xml"/>' +
      '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>' +
      "</Types>";
    var rels =
      '<?xml version="1.0" encoding="UTF-8"?>' +
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
      '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>' +
      "</Relationships>";
    var docRels =
      '<?xml version="1.0" encoding="UTF-8"?>' +
      '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"></Relationships>';
    return zipStore([
      { name: "[Content_Types].xml", data: contentTypes },
      { name: "_rels/.rels", data: rels },
      { name: "word/document.xml", data: buildDocumentXml(text, title || "") },
      { name: "word/_rels/document.xml.rels", data: docRels },
    ]);
  }

  function downloadBytes(bytes, filename, mime) {
    var blob = new Blob([bytes], {
      type: mime || "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = filename || "documento.docx";
    a.click();
    URL.revokeObjectURL(url);
  }

  /** DOCX real no browser (sem Pro). */
  function exportLocalDocx(text, filename, title) {
    var bytes = buildDocxBytes(text, title || "Ouviescrevi");
    downloadBytes(bytes, filename || "documento.docx");
    return true;
  }

  /** DOCX Pro via API; fallback local se 403/503. */
  async function exportDocxPro(text, opts) {
    opts = opts || {};
    var title = opts.title || "Transcrição Ouviescrevi";
    var filename = opts.filename || "transcricao.docx";
    try {
      if (global.OuviescreviAPI && typeof global.OuviescreviAPI.init === "function") {
        await global.OuviescreviAPI.init();
      }
      var base =
        global.OuviescreviAPI && typeof global.OuviescreviAPI.getBase === "function"
          ? global.OuviescreviAPI.getBase()
          : "";
      var headers =
        global.OuviescreviAPI && typeof global.OuviescreviAPI.authHeaders === "function"
          ? global.OuviescreviAPI.authHeaders({ "Content-Type": "application/json" })
          : { "Content-Type": "application/json" };
      var res = await fetch(base + "/api/export/docx", {
        method: "POST",
        headers: headers,
        body: JSON.stringify({ text: text, title: title }),
      });
      if (res.status === 403 || res.status === 503) {
        if (opts.allowLocalFallback !== false) {
          exportLocalDocx(text, filename, title);
          return { ok: true, via: "local", proRequired: true };
        }
        var err = await res.json().catch(function () {
          return {};
        });
        return { ok: false, proRequired: true, detail: err.detail || "Disponível no plano Pro." };
      }
      if (!res.ok) throw new Error("Falha na exportação");
      var blob = await res.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
      return { ok: true, via: "api" };
    } catch (e) {
      if (opts.allowLocalFallback !== false) {
        exportLocalDocx(text, filename, title);
        return { ok: true, via: "local", error: e && e.message };
      }
      throw e;
    }
  }

  global.OuviescreviDocx = {
    buildDocxBytes: buildDocxBytes,
    exportLocalDocx: exportLocalDocx,
    exportDocxPro: exportDocxPro,
  };
})(typeof window !== "undefined" ? window : globalThis);
