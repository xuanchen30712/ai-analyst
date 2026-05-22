#!/usr/bin/env python3
import csv
import os
import re
import sys
from datetime import datetime
from html import unescape
from pathlib import Path

from webexteamssdk import WebexTeamsAPI

SPACE_TITLE = "Cisco IQ Feedback"
BOT_EMAIL = "cisco-iq-feedback@webex.bot"
EXCLUDE_THREAD_REPLIES = False


def html_to_text(s: str) -> str:
    s = unescape(s or "")
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()


def to_plain_obj(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "to_dict"):
        try:
            return x.to_dict()
        except Exception:
            return {}
    return {}


def flatten_attachment_content(obj):
    parts = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text", "title", "value") and isinstance(v, str):
                parts.append(v)
            else:
                parts.extend(flatten_attachment_content(v))
    elif isinstance(obj, list):
        for item in obj:
            parts.extend(flatten_attachment_content(item))
    elif isinstance(obj, str):
        parts.append(obj)
    return parts


def get_best_body(msg) -> str:
    md = getattr(msg, "markdown", None) or ""
    tx = getattr(msg, "text", None) or ""
    html = getattr(msg, "html", None) or ""
    base = (md.strip() or tx.strip() or html_to_text(html)).strip()

    attachment_chunks = []
    attachments = getattr(msg, "attachments", None) or []
    for a in attachments:
        a_dict = to_plain_obj(a)
        content = a_dict.get("content")
        if content:
            attachment_chunks.extend(flatten_attachment_content(content))

    attachment_text = "\n".join([c.strip() for c in attachment_chunks if c and c.strip()])
    if attachment_text:
        return (base + "\n" + attachment_text).strip() if base else attachment_text
    return base


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def extract_submission_type_and_id(body: str):
    t = body or ""
    t = t.replace("\r", "\n")
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)

    m = re.search(
        r"(?is)(?:^|\n)\s*(AI Application Feedback Submitted|General Feedback Submitted)\s*:\s*([0-9a-fA-F\-]{8,})",
        t,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", ""


def looks_like_feedback(body: str) -> bool:
    b = (body or "").lower()
    return (
        "feedback submitted" in b
        or "account id:" in b
        or "user email:" in b
        or "feedback:" in b
        or "comment:" in b
    )


def parse_fields(body: str) -> dict:
    t = body or ""
    t = t.replace("\r", "\n").replace("\t", " ").replace("\xa0", " ")

    # Convert markdown links/code/bold/headings to plain text
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", t)
    t = re.sub(r"`([^`]*)`", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)

    # Convert inline bullets into line-separated items
    t = re.sub(r"\s+[•\u2022]\s+", "\n- ", t)
    t = re.sub(r"\s+-\s+(?=[A-Za-z][A-Za-z ]+\s*:)", "\n- ", t)
    t = re.sub(r"\n+", "\n", t).strip()

    labels = [
        "Application",
        "Jira",
        "Timestamp",
        "Account ID",
        "Account Name",
        "User ID",
        "User Name",
        "User Email",
        "Cisco User",
        "Feedback",
        "Reasons",
        "Comment",
        "Page",
        "Environment",
        "Region",
    ]
    label_union = "|".join(re.escape(x) for x in labels)

    pattern = re.compile(
        rf"(?is)(?:^|\n)\s*-\s*({label_union})\s*:\s*(.*?)\s*(?=(?:\n\s*-\s*(?:{label_union})\s*:)|$)"
    )

    out = {}
    for m in pattern.finditer(t):
        key = m.group(1).strip().lower().replace(" ", "_")
        val = normalize_whitespace(m.group(2))
        if key and val and key not in out:
            out[key] = val

    return out


def parse_feedback_row(msg, body: str) -> dict:
    submission_type, submission_id = extract_submission_type_and_id(body)
    f = parse_fields(body)

    return {
        "created": str(getattr(msg, "created", "")),
        "message_id": getattr(msg, "id", ""),
        "room_id": getattr(msg, "roomId", ""),
        "person_email": getattr(msg, "personEmail", ""),
        "person_display_name": getattr(msg, "personDisplayName", ""),
        "is_thread_reply": bool(getattr(msg, "parentId", None)),
        "submission_type": submission_type,
        "submission_id": submission_id,
        "jira": f.get("jira", ""),
        "event_timestamp": f.get("timestamp", ""),
        "application": f.get("application", ""),
        "account_id": f.get("account_id", ""),
        "account_name": f.get("account_name", ""),
        "user_id": f.get("user_id", ""),
        "user_name": f.get("user_name", ""),
        "user_email": f.get("user_email", ""),
        "cisco_user": f.get("cisco_user", ""),
        "feedback": f.get("feedback", ""),
        "reasons": f.get("reasons", ""),
        "comment": f.get("comment", ""),
        "page": f.get("page", ""),
        "environment": f.get("environment", ""),
        "region": f.get("region", ""),
        "raw_text": body,
    }


def main():
    token = os.getenv("WEBEX_TOKEN", "").strip()
    if not token:
        print("ERROR: WEBEX_TOKEN is not set")
        print('Set it with: export WEBEX_TOKEN="your_token"')
        sys.exit(1)

    api = WebexTeamsAPI(access_token=token)

    room = None
    for r in api.rooms.list():
        if (r.title or "").strip() == SPACE_TITLE:
            room = r
            break

    if not room:
        print(f'ERROR: Space "{SPACE_TITLE}" not found')
        sys.exit(1)

    print(f'Found space: "{room.title}"')
    print("Retrieving all messages...")

    total = 0
    bot_seen = 0
    thread_excluded = 0
    empty_body = 0
    non_feedback = 0
    exported = []

    for m in api.messages.list(roomId=room.id):
        total += 1

        sender_email = (getattr(m, "personEmail", "") or "").strip().lower()
        if sender_email != BOT_EMAIL:
            continue
        bot_seen += 1

        try:
            msg = api.messages.get(m.id)
        except Exception:
            msg = m

        if EXCLUDE_THREAD_REPLIES and getattr(msg, "parentId", None):
            thread_excluded += 1
            continue

        body = get_best_body(msg)
        if not body:
            empty_body += 1
            exported.append(parse_feedback_row(msg, ""))
            continue

        if not looks_like_feedback(body):
            non_feedback += 1
            exported.append(parse_feedback_row(msg, body))
            continue

        exported.append(parse_feedback_row(msg, body))

    print("\nValidation preview (first 5 parsed rows):")
    for i, row in enumerate(exported[:5], 1):
        print(
            f"{i}. created={row['created']} | feedback={row['feedback']} | "
            f"cisco_user={row['cisco_user']} | user_email={row['user_email']} | "
            f"account_id={row['account_id']} | comment={row['comment'][:80]}"
        )

    print("\nDebug raw_text preview (first 2 rows):")
    for i, row in enumerate(exported[:2], 1):
        preview = (row.get("raw_text", "") or "")[:350].replace("\n", " ")
        print(f"{i}. {preview}")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"webex_iq_feedback_bot_only_{ts}.csv"

    fieldnames = [
        "created",
        "message_id",
        "room_id",
        "person_email",
        "person_display_name",
        "is_thread_reply",
        "submission_type",
        "submission_id",
        "jira",
        "event_timestamp",
        "application",
        "account_id",
        "account_name",
        "user_id",
        "user_name",
        "user_email",
        "cisco_user",
        "feedback",
        "reasons",
        "comment",
        "page",
        "environment",
        "region",
        "raw_text",
    ]

    with out_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(exported)

    print(f"\nTotal scanned: {total}")
    print(f"Bot seen: {bot_seen}")
    print(f"Thread excluded: {thread_excluded}")
    print(f"Empty body: {empty_body}")
    print(f"Non-feedback body: {non_feedback}")
    print(f"Rows exported: {len(exported)}")
    print(f"CSV written: {out_file}")


if __name__ == "__main__":
    main()