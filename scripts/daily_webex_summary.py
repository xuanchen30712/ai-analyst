#!/usr/bin/env python3
"""
Daily Cisco IQ Space Summary
Fetches yesterday's messages from the Ask-Cisco IQ space,
builds a markdown summary, and sends it as a Webex DM.

Usage (local test):
    export WEBEX_TOKEN="your_personal_or_bot_token"
    export RECIPIENT_EMAIL="your.email@cisco.com"
    python scripts/daily_webex_summary.py

Environment variables:
    WEBEX_TOKEN       — Webex personal or bot access token (required)
    RECIPIENT_EMAIL   — Email to send the DM to (required)
    SPACE_TITLE       — Override the space name (optional)
    LOOKBACK_HOURS    — Hours to look back, default 24 (optional)
"""

import os
import re
import sys
import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
from html import unescape

from webexteamssdk import WebexTeamsAPI


# ── Config (override via env vars) ───────────────────────────────────────────
SPACE_TITLE        = os.getenv("SPACE_TITLE", "Ask-Cisco IQ")
ROOM_ID            = os.getenv("ROOM_ID", "15d842c0-48ce-11ec-afc8-f34a2c0fc5c9")

TARGET_ROOM_ID     = os.getenv("TARGET_ROOM_ID", "").strip()
TARGET_ROOM_TITLE  = os.getenv("TARGET_ROOM_TITLE", "Xuan Chen").strip()
RECIPIENT_EMAIL    = os.getenv("RECIPIENT_EMAIL", "").strip()  # fallback only

LOOKBACK_HOURS     = int(os.getenv("LOOKBACK_HOURS", "24"))

# ── Helpers (reused from retrieve_webex_feedback.py) ─────────────────────────
def html_to_text(s: str) -> str:
    s = unescape(s or "")
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"</p\s*>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return s.strip()

def room_deeplink_from_room_id(room_id: str) -> str:
    """Best-effort room deeplink: webexteams://im?space=<uuid>."""
    if not room_id:
        return ""
    try:
        decoded = base64.b64decode(room_id + "==").decode("utf-8", errors="ignore")
        m = re.search(r"/ROOM/([0-9a-fA-F-]{36})", decoded)
        if m:
            return f"webexteams://im?space={m.group(1)}"
    except Exception:
        pass
    return ""

def get_message_text(msg) -> str:
    md   = getattr(msg, "markdown", None) or ""
    text = getattr(msg, "text", None) or ""
    html = getattr(msg, "html", None) or ""
    return (md.strip() or text.strip() or html_to_text(html)).strip()

def find_room_by_title_contains(api, title: str):
    t = (title or "").strip().lower()
    if not t:
        return None
    for r in api.rooms.list():
        room_title = (r.title or "").strip().lower()
        if t in room_title:
            return r
    return None

def extract_themes_from_messages(messages: list, max_themes: int = 5) -> list:
    """Extract top discussion themes/topics from messages."""
    from collections import Counter
    import re
    
    all_text = " ".join([get_message_text(m).lower() for m in messages])
    
    # Common question/issue patterns
    patterns = {
        "Questions": r"(how|what|when|where|why|can i|do you|is it|should i)\b",
        "Problems/Bugs": r"(error|bug|crash|fail|broken|not work|doesn't work|issue|problem)",
        "Feature Requests": r"(add|need|want|request|feature|could|would be|wish)",
        "Performance": r"(slow|fast|speed|lag|timeout|delay|hang)",
        "Integration": r"(integration|api|connect|sync|export|import|webhook)",
        "UI/UX": r"(button|interface|difficult|confusing|unclear|hard to find)",
        "Data": r"(data|export|report|csv|excel|metric|accuracy)",
    }
    
    theme_counts = {}
    for theme_name, pattern in patterns.items():
        count = len(re.findall(pattern, all_text))
        if count > 0:
            theme_counts[theme_name] = count
    
    themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:max_themes]
    return themes

def build_review_queue(scored_messages: list, max_items: int = 8) -> list:
    """Return top actionable messages for triage queue."""
    actionable = []
    for m in scored_messages:
        msg_type = m.get("type", "General")
        text = (m.get("text", "") or "").lower()

        is_actionable_type = msg_type in {"Bug/Issue", "Performance", "Feature Request"}
        has_urgent_signal = any(
            k in text for k in ["urgent", "critical", "blocker", "sev", "production", "outage"]
        )

        if not is_actionable_type and not has_urgent_signal:
            continue

        # Priority label
        score = float(m.get("importance", 0))
        if has_urgent_signal or score >= 75:
            priority = "P0"
        elif score >= 60:
            priority = "P1"
        else:
            priority = "P2"

        actionable.append({
            "priority": priority,
            "importance": score,
            "type": msg_type,
            "sender": m.get("sender", m.get("email", "unknown")),
            "snippet": m.get("snippet", ""),
            "message_id": m.get("message_id", ""),
            "created_short": m.get("created_short", "unknown"),
            "room_link": m.get("room_link", ""),
            "created_raw": m.get("created_raw", ""),
        })

    # Sort by priority then importance desc then recency desc
    prio_rank = {"P0": 0, "P1": 1, "P2": 2}
    actionable.sort(
        key=lambda x: (
            prio_rank.get(x["priority"], 9),
            -x["importance"],
            x["created_raw"],
        ),
        reverse=False,
    )

    return actionable[:max_items]

def categorize_message_type(text: str) -> str:
    """Classify message as question, bug report, feature request, etc (stricter)."""
    t = (text or "").lower()

    bug_patterns = [
        r"\berror\b", r"\bexception\b", r"\bcrash(ed|ing)?\b",
        r"\bfail(ed|ure|ing)?\b", r"\bbroken\b",
        r"\bnot working\b", r"\bdoesn'?t work\b",
    ]
    perf_patterns = [
        r"\bslow\b", r"\blag\b", r"\btimeout\b", r"\bdelay\b",
        r"\bhang(ing)?\b", r"\bperformance\b",
    ]
    feature_patterns = [
        r"\bfeature request\b", r"\brequest\b", r"\bplease add\b",
        r"\bneed\b", r"\bwould like\b", r"\bwish\b",
    ]
    question_patterns = [
        r"\?$", r"\bhow\b", r"\bwhat\b", r"\bwhen\b", r"\bwhere\b", r"\bwhy\b",
        r"\bcan i\b", r"\bdo we\b", r"\bis there\b",
    ]

    if any(re.search(p, t) for p in bug_patterns):
        return "Bug/Issue"
    if any(re.search(p, t) for p in perf_patterns):
        return "Performance"
    if any(re.search(p, t) for p in feature_patterns):
        return "Feature Request"
    if any(re.search(p, t) for p in question_patterns):
        return "Question"
    return "General"


def score_message_importance(msg, sender_counts: dict) -> float:
    """Score importance 0-100 based on content and sender prominence."""
    text = get_message_text(msg)
    sender_email = (getattr(msg, "personEmail", "") or "").lower()
    
    # Base score on message length (longer = more substantive)
    length_score = min(len(text) / 500 * 30, 30)
    
    # Boost for critical keywords
    critical_words = ["urgent", "critical", "blocker", "broken", "error", "fail"]
    keyword_score = 10 if any(w in text.lower() for w in critical_words) else 0
    
    # Boost based on sender frequency (active contributors matter more)
    sender_freq = sender_counts.get(sender_email, 1)
    freq_score = min(sender_freq / 10 * 20, 20)
    
    # Message type score
    msg_type = categorize_message_type(text)
    type_scores = {
        "Bug/Issue": 30,
        "Feature Request": 15,
        "Question": 10,
        "Performance": 25,
        "General": 0
    }
    type_score = type_scores.get(msg_type, 0)
    
    total = length_score + keyword_score + freq_score + type_score
    return min(total, 100)


# ── Core logic ───────────────────────────────────────────────────────────────
def fetch_yesterday_messages(api, room_id: str) -> list:
    """Pull messages from the last LOOKBACK_HOURS, stop early once past window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    messages = []

    for m in api.messages.list(roomId=room_id):
        created = getattr(m, "created", None)
        if not created:
            continue

        # Parse ISO timestamp — webexteamssdk returns strings like "2026-05-18T14:32:00.000Z"
        if isinstance(created, str):
            created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
        else:
            created_dt = created

        if created_dt < cutoff:
            break  # Messages are newest-first; stop when we go past the window

        messages.append(m)

    return messages


def build_summary_markdown(messages: list, date_label: str) -> str:
    """Build a rich markdown summary with themes, priorities, and recommendations."""
    total = len(messages)

    if total == 0:
        return (
            f"## 📋 Cisco IQ Daily Summary — {date_label}\n\n"
            f"No messages were posted in **{SPACE_TITLE}** in the last 24 hours."
        )

    # Sender stats (exclude bots)
    sender_counts = Counter()
    sender_names   = {}
    for m in messages:
        email = (getattr(m, "personEmail", "") or "").strip().lower()
        name  = (getattr(m, "personDisplayName", "") or email).strip()
        if email and not email.endswith(".bot"):
            sender_counts[email] += 1
            sender_names[email] = name

    # Extract themes
    themes = extract_themes_from_messages(messages)

    # Score and rank messages by importance
    scored_messages = []
    for m in messages:
        email = (getattr(m, "personEmail", "") or "").lower()
        if email.endswith(".bot"):
            continue
        text = get_message_text(m).strip()
        if not text or len(text) < 10:
            continue
        
        importance = score_message_importance(m, sender_counts)
        msg_type = categorize_message_type(text)
        
        created_raw = str(getattr(m, "created", "") or "")
        created_short = created_raw.replace("T", " ").replace("Z", " UTC")[:19] if created_raw else "unknown"

        scored_messages.append({
            "text": text,
            "email": email,
            "sender": sender_names.get(email, email),
            "importance": importance,
            "type": msg_type,
            "snippet": text[:120].replace("\n", " ") + ("…" if len(text) > 120 else ""),
            "message_id": str(getattr(m, "id", "") or ""),
            "room_id": str(getattr(m, "roomId", "") or ""),
            "created_raw": created_raw,
            "created_short": created_short,
            "room_link": room_deeplink_from_room_id(str(getattr(m, "roomId", "") or "")),
        })
    # Sort by importance descending
    scored_messages.sort(key=lambda x: x["importance"], reverse=True)

    # Build summary md
    lines = [
        f"## 📋 Cisco IQ Daily Summary — {date_label}",
        f"",
        f"**{total} messages** from **{len(sender_counts)} contributors**",
        f"",
        f"---",
        f"",
        f"### 📊 Key Metrics",
        f"- Total messages: {total}",
        f"- Unique contributors: {len(sender_counts)}",
        f"- Avg messages per contributor: {total // len(sender_counts) if sender_counts else 0}",
    ]

    # Add themes
    if themes:
        lines += [
            f"",
            f"### 🎯 Discussion Themes",
        ]
        for theme_name, count in themes:
            pct = (count / total * 100)
            lines.append(f"- {theme_name}: **{count}** mentions ({pct:.0f}%)")

    # Add top contributors
    lines += [
        f"",
        f"### 👥 Top Contributors",
    ]
    for email, count in sender_counts.most_common(5):
        name = sender_names.get(email, email)
        lines.append(f"- {name}: {count} message{'s' if count > 1 else ''}")

    # Add message type distribution
    type_counts = Counter(m["type"] for m in scored_messages)
    if type_counts:
        lines += [
            f"",
            f"### 📝 Message Types",
        ]
        for msg_type, count in type_counts.most_common():
            lines.append(f"- {msg_type}: {count}")

    # Add HIGH PRIORITY messages (importance > 60)
    critical_msgs = [m for m in scored_messages if m["importance"] >= 60]
    if critical_msgs:
        critical_show_n = min(5, len(critical_msgs))
        lines += [
            "",
            f"### 🚨 HIGH PRIORITY ({len(critical_msgs)} total, showing top {critical_show_n})",
        ]
        for msg in critical_msgs[:critical_show_n]:
            meta = f"id={msg['message_id'][:12]}..., time={msg['created_short']}"
            if msg["room_link"]:
                meta += f", [Open Space]({msg['room_link']})"
            lines.append(f"- **[{msg['type']}]** {msg['sender']}: {msg['snippet']} ({meta})")

    # Add medium priority messages (importance 30-60)
    medium_msgs = [m for m in scored_messages if 30 <= m["importance"] < 60]
    if medium_msgs:
        medium_show_n = min(3, len(medium_msgs))
        lines += [
            "",
            f"### ⚠️ MEDIUM PRIORITY ({len(medium_msgs)} total, showing top {medium_show_n})",
        ]
        for msg in medium_msgs[:medium_show_n]:
            meta = f"id={msg['message_id'][:12]}..., time={msg['created_short']}"
            if msg["room_link"]:
                meta += f", [Open Space]({msg['room_link']})"
            lines.append(f"- **[{msg['type']}]** {msg['sender']}: {msg['snippet']} ({meta})")

    # Add review queue (actionable triage list)
    review_queue = build_review_queue(scored_messages, max_items=8)
    if review_queue:
        p0 = sum(1 for x in review_queue if x["priority"] == "P0")
        p1 = sum(1 for x in review_queue if x["priority"] == "P1")
        p2 = sum(1 for x in review_queue if x["priority"] == "P2")

        lines += [
            "",
            f"### 📌 REVIEW QUEUE ({len(review_queue)} items: P0={p0}, P1={p1}, P2={p2})",
            "_Triage focus: bugs, performance, urgent asks_",
        ]

        for item in review_queue:
            meta = f"id={item['message_id'][:12]}..., time={item['created_short']}"
            if item["room_link"]:
                meta += f", [Open Space]({item['room_link']})"
            lines.append(
                f"- **{item['priority']} · [{item['type']}]** {item['sender']}: "
                f"{item['snippet']} ({meta})"
            )

    # Generate recommendations
    recommendations = generate_recommendations(scored_messages, type_counts, sender_counts)
    if recommendations:
        lines += [
            f"",
            f"### 💡 Recommended Actions",
        ]
        for i, rec in enumerate(recommendations[:3], 1):
            lines.append(f"{i}. {rec}")

    lines += [
        f"",
        f"---",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_  ",
        f"_Space: {SPACE_TITLE} · Lookback: {LOOKBACK_HOURS}h_",
    ]

    return "\n".join(lines)


def generate_recommendations(scored_messages: list, type_counts: dict, sender_counts: dict) -> list:
    """Generate 3-5 actionable recommendations from message patterns."""
    recommendations = []

    # Recommendation 1: High-priority issues need triage
    bug_msgs = [m for m in scored_messages if m["type"] == "Bug/Issue"]
    if len(bug_msgs) >= 2:
        recommendations.append(
            f"Triage **{len(bug_msgs)} bug/issue reports** — review high-priority items in next standup"
        )

    # Recommendation 2: Common feature requests
    feature_msgs = [m for m in scored_messages if m["type"] == "Feature Request"]
    if len(feature_msgs) >= 3:
        recommendations.append(
            f"Consolidate **{len(feature_msgs)} feature requests** into backlog for roadmap review"
        )

    # Recommendation 3: Performance complaints
    perf_msgs = [m for m in scored_messages if m["type"] == "Performance"]
    if len(perf_msgs) >= 2:
        recommendations.append(
            f"Investigate **performance concerns** raised by {len(set(m['email'] for m in perf_msgs))} users"
        )

    # Recommendation 4: High-volume contributor reaching out
    if sender_counts:
        top_sender = sender_counts.most_common(1)[0]
        if top_sender[1] >= 15:
            recommendations.append(
                f"Check in with top contributor **{top_sender[0]}** ({top_sender[1]} messages) — may need support"
            )

    # Recommendation 5: Many unanswered questions
    question_msgs = [m for m in scored_messages if m["type"] == "Question"]
    if len(question_msgs) >= 5:
        recommendations.append(
            f"**{len(question_msgs)} unanswered questions** in space — schedule FAQ/docs update session"
        )

    return recommendations


def find_room(api, title: str):
    """Find a Webex room by ID (preferred) or partial title match."""
    # Try direct room ID lookup first — most reliable
    if ROOM_ID:
        for r in api.rooms.list():
            # Webex API room IDs are base64 but the UUID appears at the end after decode
            import base64
            try:
                decoded = base64.b64decode(r.id + "==").decode("utf-8", errors="ignore")
                if ROOM_ID.lower() in decoded.lower():
                    return r
            except Exception:
                pass
            # Also try direct match in case SDK returns short IDs
            if ROOM_ID.lower() in (r.id or "").lower():
                return r

    # Fall back to partial title match (handles long space names)
    title_lower = title.lower()
    for r in api.rooms.list():
        room_title = (r.title or "").lower()
        if title_lower in room_title or room_title.startswith(title_lower):
            return r

    return None


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    token = os.getenv("WEBEX_TOKEN", "").strip()
    if not token:
        print("ERROR: WEBEX_TOKEN is not set.")
        print('  export WEBEX_TOKEN="your_token"')
        sys.exit(1)



    print(f"Connecting to Webex...")
    api = WebexTeamsAPI(access_token=token)

    # Verify token works
    try:
        me = api.people.me()
        print(f"Authenticated as: {me.displayName} ({me.emails[0] if me.emails else 'unknown'})")
    except Exception as e:
        print(f"ERROR: Token invalid or expired — {e}")
        sys.exit(1)

    # Find space
    print(f'Looking for space: "{SPACE_TITLE}"...')
    room = find_room(api, SPACE_TITLE)
    if not room:
        print(f'ERROR: Space "{SPACE_TITLE}" not found. Check SPACE_TITLE env var.')
        print("Available spaces:")
        for r in api.rooms.list():
            print(f"  - {r.title}")
        sys.exit(1)
    print(f'  Found: "{room.title}" (id: {room.id[:12]}...)')

    # Fetch messages
    date_label = (datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)).strftime("%Y-%m-%d")
    print(f"Fetching messages from last {LOOKBACK_HOURS}h (since ~{date_label})...")
    messages = fetch_yesterday_messages(api, room.id)
    print(f"  Found {len(messages)} messages")

    # Build summary
    summary_md = build_summary_markdown(messages, date_label)
    print("\n--- SUMMARY PREVIEW ---")
    print(summary_md[:600] + ("..." if len(summary_md) > 600 else ""))
    print("--- END PREVIEW ---\n")

       # Send summary with 3-level fallback:
    # 1) TARGET_ROOM_ID
    # 2) TARGET_ROOM_TITLE
    # 3) RECIPIENT_EMAIL (legacy behavior)
    sent = False

    if TARGET_ROOM_ID:
        try:
            print(f"Sending to TARGET_ROOM_ID: {TARGET_ROOM_ID[:12]}...")
            api.messages.create(roomId=TARGET_ROOM_ID, markdown=summary_md)
            print("✅ Summary sent via TARGET_ROOM_ID")
            sent = True
        except Exception as e:
            print(f"WARN: send via TARGET_ROOM_ID failed: {e}")

    if not sent and TARGET_ROOM_TITLE:
        room_out = find_room_by_title_contains(api, TARGET_ROOM_TITLE)
        if room_out:
            try:
                print(f'Sending to room title match: "{room_out.title}"')
                api.messages.create(roomId=room_out.id, markdown=summary_md)
                print(f'✅ Summary sent to room "{room_out.title}"')
                sent = True
            except Exception as e:
                print(f"WARN: send via TARGET_ROOM_TITLE failed: {e}")
        else:
            print(f'WARN: no room found matching TARGET_ROOM_TITLE="{TARGET_ROOM_TITLE}"')

    if not sent and RECIPIENT_EMAIL:
        try:
            print(f"Sending fallback DM to RECIPIENT_EMAIL: {RECIPIENT_EMAIL}")
            api.messages.create(toPersonEmail=RECIPIENT_EMAIL, markdown=summary_md)
            print(f"✅ Summary sent via RECIPIENT_EMAIL: {RECIPIENT_EMAIL}")
            sent = True
        except Exception as e:
            print(f"WARN: send via RECIPIENT_EMAIL failed: {e}")

    if not sent:
        print("ERROR: Could not send summary via TARGET_ROOM_ID, TARGET_ROOM_TITLE, or RECIPIENT_EMAIL.")
        print("Tip: set TARGET_ROOM_ID for the most reliable delivery.")
        sys.exit(1)

if __name__ == "__main__":
    main()