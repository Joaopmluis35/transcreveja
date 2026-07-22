"""Envio de notificações por email — Resend (HTTPS) ou SMTP."""
from __future__ import annotations

import logging
import os
import re
import smtplib
from email.message import EmailMessage

import requests

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _normalize_email(addr: str) -> str:
    return re.sub(r"\s+", "", (addr or "").strip())


def _db_config() -> dict[str, str]:
    try:
        import admin_store

        return admin_store.get_config()
    except Exception:
        return {}


def notify_email_to() -> str:
    cfg = _db_config()
    db_to = _normalize_email(cfg.get("alert_email_to") or "")
    if db_to:
        return db_to
    return _normalize_email(
        os.getenv("SMTP_TO") or os.getenv("NOTIFY_EMAIL_TO") or "ouviescrevi@gmail.com"
    )


def activity_notifications_enabled() -> bool:
    return _db_config().get("notify_activity_enabled", "1") != "0"


def notify_from_address() -> str:
    return (
        os.getenv("RESEND_FROM")
        or os.getenv("SMTP_FROM")
        or "Ouviescrevi <onboarding@resend.dev>"
    ).strip()


def _resend_configured() -> bool:
    return bool((os.getenv("RESEND_API_KEY") or "").strip())


def _smtp_configured() -> bool:
    return bool((os.getenv("SMTP_USER") or "").strip() and (os.getenv("SMTP_PASSWORD") or "").strip())


def _smtp_fallback_enabled() -> bool:
    explicit = (os.getenv("EMAIL_SMTP_FALLBACK") or "").strip().lower()
    if explicit in ("1", "true", "yes", "on"):
        return True
    if explicit in ("0", "false", "no", "off"):
        return False
    # Com Resend configurado, não tentar SMTP no Render (bloqueado e demora ~40s).
    return not _resend_configured()


def get_email_status() -> dict:
    resend = _resend_configured()
    smtp = _smtp_configured()
    cfg = _db_config()
    last_failure = None
    try:
        import admin_store

        for row in admin_store.list_email_notifications(limit=10):
            if row.get("status") == "failed":
                last_failure = row
                break
    except Exception:
        pass

    from_addr = notify_from_address()
    hints: list[str] = []
    if not resend and not smtp:
        hints.append("Configura RESEND_API_KEY no Render (recomendado).")
    elif resend and "onboarding@resend.dev" not in from_addr and "@" in from_addr:
        hints.append(
            "O domínio em RESEND_FROM tem de estar verificado em resend.com/domains. "
            "Para teste rápido usa RESEND_FROM=onboarding@resend.dev (só envia para o email da conta Resend)."
        )
    if resend and smtp and not _smtp_fallback_enabled():
        hints.append("SMTP ignorado — só Resend (EMAIL_SMTP_FALLBACK não está ativo).")

    return {
        "resend_configured": resend,
        "smtp_configured": smtp,
        "smtp_fallback": _smtp_fallback_enabled(),
        "provider_ready": resend or smtp,
        "from_address": from_addr,
        "default_to": notify_email_to(),
        "notify_activity_enabled": activity_notifications_enabled(),
        "alert_email_enabled": cfg.get("alert_email_enabled") == "1",
        "alert_transcriptions_daily": int(cfg.get("alert_transcriptions_daily") or 0),
        "alert_visits_daily": int(cfg.get("alert_visits_daily") or 0),
        "last_failure": last_failure,
        "render_hint": " ".join(hints),
    }


def _record_email_log(
    kind: str,
    recipient: str,
    subject: str,
    status: str,
    *,
    detail: str | None = None,
    actor: str | None = None,
) -> None:
    try:
        import admin_store

        admin_store.log_email_notification(
            kind, recipient, subject, status, detail=detail, actor=actor
        )
    except Exception as exc:
        logger.warning("Falha ao registar log de email: %s", exc)


def _parse_resend_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict):
            return str(data.get("message") or data.get("error") or data)[:300]
    except Exception:
        pass
    return (resp.text or "")[:300]


def _send_via_resend(to: str, subject: str, body: str) -> tuple[bool, str | None]:
    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    if not api_key:
        return False, None
    from_addr = notify_from_address()
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": from_addr,
                "to": [to],
                "subject": subject,
                "text": body,
            },
            timeout=25,
        )
        if resp.status_code in (200, 201):
            logger.info("Notificação enviada via Resend → %s (%s)", to, subject)
            return True, "resend"
        detail = _parse_resend_error(resp)
        err = f"Resend {resp.status_code}: {detail}"
        if resp.status_code in (403, 422):
            err += (
                " — verifica RESEND_FROM e o domínio em resend.com/domains "
                "(ou usa onboarding@resend.dev para teste)."
            )
        logger.warning(err)
        return False, err
    except Exception as exc:
        err = f"Resend: {exc}"
        logger.warning(err)
        return False, err[:300]


def _send_via_smtp(
    msg: EmailMessage,
    *,
    use_ssl: bool,
    port: int,
    timeout: int = 8,
) -> tuple[bool, str | None]:
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not (smtp_user and smtp_password):
        return False, None
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_host, port, timeout=timeout) as smtp:
                smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, port, timeout=timeout) as smtp:
                smtp.ehlo()
                smtp.starttls()
                smtp.ehlo()
                smtp.login(smtp_user, smtp_password)
                smtp.send_message(msg)
        logger.info("Notificação enviada via SMTP → %s (%s)", msg["To"], msg["Subject"])
        mode = "smtp_ssl" if use_ssl else "smtp_starttls"
        return True, mode
    except Exception as exc:
        mode = "SSL" if use_ssl else "STARTTLS"
        err = f"SMTP {smtp_host}:{port} ({mode}): {exc}"
        logger.warning(err)
        return False, str(exc)[:200]


def send_notification_email(
    mensagem: str,
    assunto: str = "Nova atividade no Ouviescrevi",
    *,
    to: str | None = None,
    kind: str = "activity",
    actor: str | None = None,
) -> tuple[bool, str | None]:
    """Envia email. Retorna (ok, erro). Preferir RESEND_API_KEY no Render."""
    recipient = _normalize_email(to or notify_email_to())
    if not recipient:
        detail = "Destinatário em falta — define o email no backoffice → Emails."
        _record_email_log(kind, "-", assunto, "failed", detail=detail, actor=actor)
        return False, detail
    if not _EMAIL_RE.match(recipient):
        detail = f"Email de destino inválido: {recipient!r}"
        _record_email_log(kind, recipient, assunto, "failed", detail=detail, actor=actor)
        return False, detail

    ok, via = _send_via_resend(recipient, assunto, mensagem)
    if ok:
        _record_email_log(kind, recipient, assunto, "sent", detail=via, actor=actor)
        return True, None

    last_err = via

    if _resend_configured() and not _smtp_fallback_enabled():
        detail = last_err or "Resend falhou e SMTP fallback está desativado."
        _record_email_log(kind, recipient, assunto, "failed", detail=detail, actor=actor)
        return False, detail

    if not _smtp_configured():
        detail = last_err or "Configure RESEND_API_KEY (recomendado) ou SMTP_USER/SMTP_PASSWORD."
        _record_email_log(kind, recipient, assunto, "failed", detail=detail, actor=actor)
        return False, detail

    msg = EmailMessage()
    msg.set_content(mensagem)
    msg["Subject"] = assunto
    msg["From"] = os.getenv("SMTP_FROM", "notificacoes@ouviescrevi.pt")
    msg["To"] = recipient

    preferred_port = int(os.getenv("SMTP_PORT", "465"))
    attempts: list[tuple[bool, int]] = (
        [(False, 587), (True, 465)] if preferred_port == 587 else [(True, preferred_port), (False, 587)]
    )

    for use_ssl, port in attempts:
        ok, err = _send_via_smtp(msg, use_ssl=use_ssl, port=port)
        if ok:
            _record_email_log(kind, recipient, assunto, "sent", detail=err, actor=actor)
            return True, None
        last_err = err

    detail = last_err or "Falha SMTP e Resend."
    _record_email_log(kind, recipient, assunto, "failed", detail=detail, actor=actor)
    return False, detail


def send_welcome_email(to: str, name: str | None = None) -> tuple[bool, str | None]:
    """Email de boas-vindas após registo no site."""
    recipient = _normalize_email(to)
    if not recipient or not _EMAIL_RE.match(recipient):
        return False, "Email de destino inválido."
    display = (name or "").strip() or recipient.split("@")[0]
    subject = "Bem-vindo ao Ouviescrevi"
    body = (
        f"Olá {display},\n\n"
        "Obrigado por te registares no Ouviescrevi!\n\n"
        "Com a tua conta tens 20 transcrições por dia (em vez de 3 sem registo), "
        "histórico das transcrições e acesso às novidades em primeira mão.\n\n"
        "Começa já em https://www.ouviescrevi.pt\n\n"
        "— Equipa Ouviescrevi"
    )
    return send_notification_email(
        body, subject, to=recipient, kind="welcome", actor=recipient
    )


def send_password_reset_email(to: str, reset_url: str) -> tuple[bool, str | None]:
    recipient = _normalize_email(to)
    if not recipient or not _EMAIL_RE.match(recipient):
        return False, "Email de destino inválido."
    subject = "Repor palavra-passe — Ouviescrevi"
    body = (
        "Olá,\n\n"
        "Recebemos um pedido para repor a palavra-passe da tua conta Ouviescrevi.\n\n"
        f"Abre este link (válido por 2 horas):\n{reset_url}\n\n"
        "Se não foste tu, ignora este email.\n\n"
        "— Equipa Ouviescrevi"
    )
    return send_notification_email(
        body, subject, to=recipient, kind="password_reset", actor=recipient
    )


def send_quota_nudge_email(to: str, name: str | None = None) -> tuple[bool, str | None]:
    recipient = _normalize_email(to)
    if not recipient or not _EMAIL_RE.match(recipient):
        return False, "Email de destino inválido."
    display = (name or "").strip() or recipient.split("@")[0]
    subject = "Ainda tens transcrições disponíveis hoje"
    body = (
        f"Olá {display},\n\n"
        "A tua conta Ouviescrevi tem quota diária disponível. "
        "Podes voltar a transcrever áudio ou vídeo em https://www.ouviescrevi.pt\n\n"
        "Dica: depois de transcrever, experimenta o Resumo ou as Perguntas de estudo.\n\n"
        "— Equipa Ouviescrevi\n"
        "(Recebes isto porque ativaste avisos por email. Para parar, responde a este email.)"
    )
    return send_notification_email(
        body, subject, to=recipient, kind="quota_nudge", actor=recipient
    )


def send_weekly_tip_email(to: str, name: str | None = None) -> tuple[bool, str | None]:
    recipient = _normalize_email(to)
    if not recipient or not _EMAIL_RE.match(recipient):
        return False, "Email de destino inválido."
    display = (name or "").strip() or recipient.split("@")[0]
    subject = "Dica Ouviescrevi: legendas e capítulos"
    body = (
        f"Olá {display},\n\n"
        "Dica da semana: depois de transcreveres um vídeo, gera legendas SRT "
        "ou capítulos automaticamente — ideal para YouTube e aulas.\n\n"
        "Experimenta em https://www.ouviescrevi.pt\n\n"
        "— Equipa Ouviescrevi"
    )
    return send_notification_email(
        body, subject, to=recipient, kind="weekly_tip", actor=recipient
    )


def send_suggestion_notification(
    suggestion_id: int,
    nome: str | None,
    mensagem: str,
    lang: str = "pt",
    referer: str | None = None,
) -> tuple[bool, str | None]:
    """Notifica o administrador de uma nova sugestão."""
    who = (nome or "").strip() or "Anónimo"
    subject = f"Nova sugestão #{suggestion_id} — Ouviescrevi"
    lines = [
        f"Nova sugestão no site (ID {suggestion_id}):",
        "",
        f"De: {who}",
        f"Idioma: {lang or 'pt'}",
    ]
    if referer:
        lines.append(f"Origem: {referer}")
    lines.extend(
        [
            "",
            "Mensagem:",
            "—" * 40,
            mensagem,
            "",
            "—" * 40,
            "Ver no backoffice → separador Sugestões.",
        ]
    )
    return send_notification_email(
        "\n".join(lines), subject, kind="suggestion", actor=who
    )
