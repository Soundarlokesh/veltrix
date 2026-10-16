"""Deterministic intent detection. No LLM is involved, so routing cannot be talked into a tool call.

Order matters: manipulation and bulk requests are checked first, then confirmations, then the
leave actions, then policy questions. Anything unrecognised is off-topic.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    MANIPULATION = "manipulation"
    BULK = "bulk"
    CONFIRM = "confirm"
    GREETING = "greeting"
    CANCEL = "cancel"
    APPLY = "apply"
    BALANCE = "balance"
    REQUESTS = "requests"
    TEAM = "team"
    POLICY = "policy"
    HYBRID = "hybrid"  # balance and a future-looking policy question together
    OFF_TOPIC = "off_topic"


BULK_RE = re.compile(
    r"\b(everyone|everybody|every employee|all employees|all staff|all of (my|the) (team|employees|staff)"
    r"|whole team|all balances|everyone'?s|every balance)\b",
    re.IGNORECASE,
)
CONFIRM_RE = re.compile(
    r"^\s*(yes|y|yes please|yep|yeah|confirm|confirmed|go ahead|please do|proceed|do it|ok|okay|sure)\s*[.!]*\s*$",
    re.IGNORECASE,
)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good morning|good afternoon)\b[\s\w!.,?']{0,30}$", re.IGNORECASE)
CANCEL_RE = re.compile(r"\b(cancel|withdraw)\b", re.IGNORECASE)
APPLY_RE = re.compile(
    r"\b(apply for|book|submit)\b.*\b(leave|time off|days? off|holiday)\b"
    r"|\b(i want|i'?d like|i need|please)\b.*\b(to take|leave|time off|days? off)\b"
    r"|\btake\b.*\b(annual|sick)\b.*\bleave\b|\bapply\b.*\bleave\b",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(r"^\s*(how|what|when|why|which|can i|do i|does|is|are|should)\b", re.IGNORECASE)
BALANCE_RE = re.compile(
    r"\bbalance\b|\bdays? (left|remaining)\b|\bremaining (leave|days)\b|\bhow many\b.*\b(annual|sick|leave|days)\b.*\b(do i have|have i|will i have|left|remain)\b",
    re.IGNORECASE,
)
FUTURE_POLICY_RE = re.compile(
    r"\b(will i have|by (december|the end|end of)|end of (the )?(year|month)|accrue|accrual|carry(-| )?forward|carry over)\b",
    re.IGNORECASE,
)
REQUESTS_RE = re.compile(
    r"\b(my|list|show)\b.*\b(requests?|applications?)\b|\bstatus of my leave\b|\bmy leave (requests|history)\b",
    re.IGNORECASE,
)
TEAM_RE = re.compile(
    r"\b(my team|direct reports?|my reports|team'?s? leave|team leave|who is off|who'?s off)\b",
    re.IGNORECASE,
)
POLICY_RE = re.compile(
    r"\b(policy|policies|leave|holiday|holidays|sick|annual|parental|maternity|paternity|bereavement|jury|voting"
    r"|carry|accrual|allowance|benefit|benefits|remote|hybrid|work from home|expense|travel|probation|notice"
    r"|certificate|public holiday|restricted|wellbeing|working time|overtime|compensation|learning|conduct"
    r"|security|hr|people team|it policy|benefit|pension|insurance)\b",
    re.IGNORECASE,
)
MANIPULATION_WORDS_RE = re.compile(r"\b(ignore|bypass|override|jailbreak)\b", re.IGNORECASE)

EMPLOYEE_ID_RE = re.compile(r"\bE\d{4}\b", re.IGNORECASE)
LEAVE_TYPE_RE = {
    "annual": re.compile(r"\bannual\b", re.IGNORECASE),
    "sick": re.compile(r"\bsick\b", re.IGNORECASE),
}
STATUS_RE = {
    "pending": re.compile(r"\bpending\b", re.IGNORECASE),
    "approved": re.compile(r"\bapproved\b", re.IGNORECASE),
    "cancelled": re.compile(r"\bcancelled\b|\bcanceled\b", re.IGNORECASE),
}


def classify(message: str, *, manipulation: bool) -> Intent:
    text = message.strip()
    if manipulation:
        return Intent.MANIPULATION
    if BULK_RE.search(text):
        return Intent.BULK
    if CONFIRM_RE.match(text):
        return Intent.CONFIRM
    if GREETING_RE.match(text) and not POLICY_RE.search(text):
        return Intent.GREETING
    if CANCEL_RE.search(text):
        return Intent.CANCEL
    if APPLY_RE.search(text) and not QUESTION_START_RE.match(text):
        return Intent.APPLY
    if BALANCE_RE.search(text):
        if FUTURE_POLICY_RE.search(text):
            return Intent.HYBRID
        return Intent.BALANCE
    if REQUESTS_RE.search(text):
        return Intent.REQUESTS
    if TEAM_RE.search(text):
        return Intent.TEAM
    if POLICY_RE.search(text):
        return Intent.POLICY
    return Intent.OFF_TOPIC


def employee_ids(message: str) -> set[str]:
    return {m.upper() for m in EMPLOYEE_ID_RE.findall(message)}


def leave_type_mentions(message: str) -> set[str]:
    return {name for name, pattern in LEAVE_TYPE_RE.items() if pattern.search(message)}


def status_mention(message: str) -> str | None:
    found = [name for name, pattern in STATUS_RE.items() if pattern.search(message)]
    return found[0] if len(found) == 1 else None


def mentions_manipulation_words(message: str) -> bool:
    return bool(MANIPULATION_WORDS_RE.search(message))
