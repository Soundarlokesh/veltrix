"""Personal-data protection (TASK.pdf section 7.4): personal data means full_name, email, phone.

- Emails are masked to first character + *** + domain (r***@veltrix.example).
- Phone numbers are removed from answers and logs.
- Log records are sanitised before they are written (LogSanitizer), so a stray email or phone
  in a log message cannot leak.
"""

from __future__ import annotations

import logging
import re

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
PHONE_CANDIDATE_RE = re.compile(r"(?<![\w-])\+?\d[\d ().-]{7,}\d(?![\w-])")
UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
PHONE_MARK = "[phone removed]"


def mask_email(address: str) -> str:
    match = EMAIL_RE.fullmatch(address.strip())
    if not match:
        return "[email removed]"
    return f"{match.group(1)}***@{match.group(2)}"


def redact_emails(text: str) -> str:
    return EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def redact_phones(text: str) -> str:
    """Remove phone-like digit runs. UUIDs and ISO dates are left alone."""
    protected = [m.span() for m in UUID_RE.finditer(text)]

    def replace(match: re.Match) -> str:
        start, end = match.span()
        if any(s <= start < e or s < end <= e for s, e in protected):
            return match.group(0)
        digits = sum(ch.isdigit() for ch in match.group(0))
        if digits < 10:
            return match.group(0)
        return PHONE_MARK

    return PHONE_CANDIDATE_RE.sub(replace, text)


def redact_text(text: str) -> str:
    return redact_phones(redact_emails(text))


class LogSanitizer(logging.Filter):
    """Masks emails and phone numbers in every log record before it is emitted."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact_text(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = ()
        return True


def install_log_sanitizer() -> None:
    sanitizer = LogSanitizer()
    for handler in logging.getLogger().handlers:
        handler.addFilter(sanitizer)
    logging.getLogger().addFilter(sanitizer)
