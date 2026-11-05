"""Outgoing email. With SMTP_URL unset (dev), messages are written as JSON files to
DEV_MAIL_DIR and logged, so codes and links can be read locally and by e2e tests."""

import json
import logging
import smtplib
import time
from email.message import EmailMessage
from urllib.parse import unquote, urlparse

from .config import get_settings

log = logging.getLogger(__name__)


def send(to: str, subject: str, text: str, meta: dict | None = None) -> None:
    s = get_settings()
    if not s.smtp_url:
        s.dev_mail_dir.mkdir(parents=True, exist_ok=True)
        path = s.dev_mail_dir / f"{time.time_ns()}-{to.replace('@', '_at_')}.json"
        path.write_text(json.dumps({"to": to, "subject": subject, "text": text, **(meta or {})}, indent=1))
        log.warning("dev mail to %s: %s (%s)", to, subject, path)
        return
    url = urlparse(s.smtp_url)
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = s.mail_from, to, subject
    msg.set_content(text)
    smtp_cls = smtplib.SMTP_SSL if url.scheme == "smtps" else smtplib.SMTP
    with smtp_cls(url.hostname, url.port or (465 if url.scheme == "smtps" else 587), timeout=20) as smtp:
        if url.scheme == "smtp":
            smtp.starttls()
        if url.username:
            smtp.login(unquote(url.username), unquote(url.password or ""))
        smtp.send_message(msg)
