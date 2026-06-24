"""Envio de notificações por email — Resend (HTTPS) ou SMTP."""
from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

import requests

logger = logging.getLogger(__name__)


def _db_config() -> dict[str, str]:
    try:
        import admin_store

        return admin_store.get_config()
    except Exception:
        return {}


def notify_email_to() -> str:
    cfg = _db_config()
    db_to = (cfg.get("alert_email_to") or "").strip()
    if db_to:
        return db_to
    return (os.getenv("SMTP_TO") or os.getenv("NOTIFY_EMAIL_TO") or "ouviescrevi@gmail.com").strip()


def activity_notifications_enabled() -> bool:
    return _db_config().get("notify_activity_enabled", "1") != "0"


def notify_from_address() -> str:
    return (
        os.getenv("RESEND_FROM")
        or os.getenv("SMTP_FROM")
        or "Ouviescrevi <notificacoes@ouviescrevi.pt>"
    ).strip()


def get_email_status() -> dict:
    resend = bool((os.getenv("RESEND_API_KEY") or "").strip())
    smtp_user = bool((os.getenv("SMTP_USER") or "").strip())
    smtp_pass = bool((os.getenv("SMTP_PASSWORD") or "").strip())
    smtp = smtp_user and smtp_pass
    cfg = _db_config()
    return {
        "resend_configured": resend,
        "smtp_configured": smtp,
        "provider_ready": resend or smtp,
        "from_address": notify_from_address(),
        "default_to": notify_email_to(),
        "notify_activity_enabled": activity_notifications_enabled(),
        "alert_email_enabled": cfg.get("alert_email_enabled") == "1",
        "alert_transcriptions_daily": int(cfg.get("alert_transcriptions_daily") or 0),
        "alert_visits_daily": int(cfg.get("alert_visits_daily") or 0),
        "render_hint": (
            "No Render, SMTP (portas 465/587) costuma estar bloqueado. "
            "Configura RESEND_API_KEY nas variáveis de ambiente."
            if not resend
            else ""
        ),
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


def _send_via_resend(to: str, subject: str, body: str) -> tuple[bool, str | None]:
    api_key = (os.getenv("RESEND_API_KEY") or "").strip()
    if not api_key:
        return False, None
    try:
        resp = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": notify_from_address(),
                "to": [to],
                "subject": subject,
                "text": body,
            },
            timeout=20,
        )
        if resp.status_code in (200, 201):
            logger.info("Notificação enviada via Resend → %s (%s)", to, subject)
            return True, "resend"
        err = f"Resend HTTP {resp.status_code}: {(resp.text or '')[:200]}"
        logger.error(err)
        return False, err
    except Exception as exc:
        err = str(exc)[:200]
        logger.error("Resend falhou: %s", exc)
        return False, err


def _send_via_smtp(msg: EmailMessage, *, use_ssl: bool, port: int, timeout: int = 20) -> tuple[bool, str | None]:
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
        logger.error(err)
        return False, str(exc)[:200]


def send_notification_email(
    mensagem: str,
    assunto: str = "Nova atividade no Ouviescrevi",
    *,
    to: str | None = None,
    kind: str = "activity",
    actor: str | None = None,
) -> bool:
    """Envia email de atividade. Preferir RESEND_API_KEY em produção (Render bloqueia SMTP)."""
    recipient = (to or notify_email_to()).strip()
    if not recipient:
        _record_email_log(kind, "-", assunto, "failed", detail="Destinatário em falta", actor=actor)
        logger.warning("Notificação não enviada: destinatário em falta.")
        return False

    ok, via = _send_via_resend(recipient, assunto, mensagem)
    if ok:
        _record_email_log(kind, recipient, assunto, "sent", detail=via, actor=actor)
        return True

    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if not (smtp_user and smtp_password):
        detail = via or "Configure RESEND_API_KEY ou SMTP_USER/SMTP_PASSWORD"
        _record_email_log(kind, recipient, assunto, "failed", detail=detail, actor=actor)
        logger.warning("Notificação não enviada: %s", detail)
        return False

    msg = EmailMessage()
    msg.set_content(mensagem)
    msg["Subject"] = assunto
    msg["From"] = os.getenv("SMTP_FROM", "notificacoes@ouviescrevi.pt")
    msg["To"] = recipient

    preferred_port = int(os.getenv("SMTP_PORT", "465"))
    attempts: list[tuple[bool, int]] = []
    if preferred_port == 587:
        attempts = [(False, 587), (True, 465)]
    else:
        attempts = [(True, preferred_port), (False, 587)]

    last_err = via
    for use_ssl, port in attempts:
        ok, err = _send_via_smtp(msg, use_ssl=use_ssl, port=port)
        if ok:
            _record_email_log(kind, recipient, assunto, "sent", detail=err, actor=actor)
            return True
        last_err = err

    _record_email_log(kind, recipient, assunto, "failed", detail=last_err, actor=actor)
    return False
