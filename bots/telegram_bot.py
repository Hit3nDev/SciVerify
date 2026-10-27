"""SciVerify on Telegram: send a scientific claim, get a plain-language verdict with the papers behind it.

    1. Start the API:         uvicorn api.main:app --port 8000
    2. Put your bot token in .env as TELEGRAM_BOT_TOKEN=...   (get one from @BotFather)
    3. Start the bot:         python -m bots.telegram_bot

The bot holds no model; it calls the API (SCIVERIFY_API, default http://127.0.0.1:8000).
Everything that builds message text is a pure function, so it is tested without Telegram.
"""

from __future__ import annotations

import html
import json
import logging
import os
import random
import re
import sys
import time
from typing import Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import httpx

log = logging.getLogger("sciverify.bot")

MAX_MESSAGE = 4000                     # Telegram allows 4096
MAX_CLAIM_CHARS = 1000
MIN_CLAIM_WORDS = 3
KEEP_RESULTS = 30                      # per chat, for the buttons under old answers

VERDICTS = {
    "SUPPORTED": {"icon": "✔", "title": "Supported by research", "short": "Supported",
                  "meaning": "The papers I found <b>back up</b> this claim."},
    "CONTRADICTED": {"icon": "✘", "title": "Contradicted by research", "short": "Contradicted",
                     "meaning": "The papers I found <b>say the opposite</b> of this claim."},
    "CONFLICTING": {"icon": "⇄", "title": "Mixed evidence", "short": "Mixed",
                    "meaning": "Some papers agree with this claim and others disagree. The research I can see "
                               "<b>hasn't settled it</b>."},
    "UNKNOWN": {"icon": "?", "title": "Not enough evidence", "short": "Not enough evidence",
                "meaning": "I couldn't find papers that clearly agree or disagree. This does <b>not</b> mean the "
                           "claim is false. It means my library doesn't answer it."},
}
JUDGEMENTS = {"SUPPORT": ("✔", "Agrees"), "CONTRADICT": ("✘", "Disagrees"), "NEUTRAL": ("–", "Doesn't say either way")}
KEEP_IN_MIND = {
    "SUPPORTED": "Supported means the papers I read agree, not that it's proven for everyone.",
    "CONTRADICTED": "Contradicted means the papers I read disagree, but newer or other research might differ.",
    "CONFLICTING": "When research is mixed, details like dose, age group or study type usually explain the "
                   "difference. The papers below are a good place to start reading.",
    "UNKNOWN": "Try rewording the claim with more specific scientific terms, or check a few papers yourself.",
}
STATUS = {
    "consistent": ("Passed", "Good sign. The papers back one version and reject the other, which is what you'd "
                             "expect if I'm really reading them."),
    "inconsistent": ("Failed: don't trust this verdict",
                     "I gave the <b>same answer</b> to your claim and to its opposite. That means I'm probably "
                     "reacting to matching words rather than meaning."),
    "inconclusive": ("Can't tell", "At least one side came back as \"not enough evidence\" or \"mixed\", so this "
                                   "check can't confirm or rule out a problem."),
}
QUIZ_CHOICES = {"SUPPORTED": "Supported", "CONTRADICTED": "Contradicted", "UNKNOWN": "Not enough evidence"}
GREETINGS = {"hi", "hello", "hey", "hii", "hiii", "yo", "hola", "namaste", "good morning", "good evening",
             "good afternoon", "start", "hey there", "hello there"}
THANKS = {"thanks", "thank you", "thx", "ty", "thank u", "thanks a lot", "great", "nice", "cool", "ok", "okay"}


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def fit(text: str) -> str:
    """Keep a message under Telegram's limit without cutting through an HTML tag."""
    if len(text) <= MAX_MESSAGE:
        return text
    cut = text[: MAX_MESSAGE - 40]
    cut = cut[: cut.rfind("\n")] if "\n" in cut else cut
    open_tags = re.findall(r"<(b|i|code)>", cut)
    closed = re.findall(r"</(b|i|code)>", cut)
    for tag in reversed(open_tags[len(closed):]):
        cut += f"</{tag}>"
    return cut + "\n\n<i>(shortened to fit Telegram)</i>"


# ------------------------------------------------------------------ plain-language helpers
def pct(x: float) -> str:
    return f"{round(x * 100)}%"


def bar(x: float, width: int = 10) -> str:
    filled = max(0, min(width, round(x * width)))
    return "■" * filled + "□" * (width - filled)


def sureness(x: float) -> str:
    if x >= 0.9:
        return "very sure"
    if x >= 0.75:
        return "fairly sure"
    if x >= 0.6:
        return "somewhat sure"
    return "not very sure"


def judge_short(name: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", name or "") or "the AI judge"


def claim_problem(text: str) -> Optional[str]:
    """A reason not to check this message at all, or None."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text)
    if len(words) < MIN_CLAIM_WORDS:
        return ("That's a bit short for me to check. Please send a <b>full sentence</b> that states something, "
                "for example:\n\n<i>Vitamin D supplements reduce the risk of bone fractures.</i>")
    if len(text) > MAX_CLAIM_CHARS:
        return (f"That message is quite long ({len(text):,} characters). I work best with <b>one claim at a time</b>, "
                f"in a sentence or two. Could you send just the main claim you want checked?")
    return None


def claim_tips(text: str) -> List[str]:
    """Friendly advice that doesn't stop the check."""
    tips = []
    t = text.strip()
    if t.endswith("?") or re.match(r"(?i)^(does|do|is|are|can|could|will|should|did|was|were|has|have)\b", t):
        tips.append("I work best with <b>statements</b>, not questions. For example, instead of "
                    "<i>Does coffee cause cancer?</i> try <i>Coffee causes cancer.</i> I've checked your "
                    "message as written.")
    if len(re.findall(r"[.!?](\s|$)", t.rstrip(".!? ") + ".")) > 2:
        tips.append("Your message has several sentences. If they make different claims, send them "
                    "<b>one at a time</b> for a clearer answer.")
    if re.search(r"(?i)\b(i|my|me)\b", t):
        tips.append("I check general scientific claims, not personal situations. Rephrase it as a general "
                    "statement for best results.")
    return tips


def quote(e: dict, limit: int) -> str:
    """The best-matching passage, without the paper title the pipeline puts in front of it."""
    text, title = " ".join(str(e.get("passage") or "").split()), " ".join(str(e.get("title") or "").split())
    if title and text.startswith(title):
        text = text[len(title):].strip()
    return esc(clip(text, limit)) if text else ""


def counts(result: dict) -> Dict[str, int]:
    thr = (result.get("settings") or {}).get("min_confidence", 0.5)
    c = {"agree": 0, "disagree": 0, "uncounted": 0}
    for e in result.get("evidence", []):
        if e["judgement"] == "SUPPORT" and e["nli_score"] >= thr:
            c["agree"] += 1
        elif e["judgement"] == "CONTRADICT" and e["nli_score"] >= thr:
            c["disagree"] += 1
        else:
            c["uncounted"] += 1
    return c


# ------------------------------------------------------------------ message builders
def format_result(r: dict, tips: Optional[List[str]] = None, corpus_size: Optional[int] = None) -> str:
    v = VERDICTS.get(r["verdict"], VERDICTS["UNKNOWN"])
    thr = (r.get("settings") or {}).get("min_confidence", 0.5)
    lines = []
    for tip in tips or []:
        lines += [f"<b>Tip:</b> {tip}", ""]
    lines += [f"{v['icon']} <b>{v['title'].upper()}</b>", "",
              "<b>Your claim</b>", f"<i>{esc(r['claim'])}</i>", "",
              "<b>What this means</b>", v["meaning"], ""]

    if r["verdict"] in ("SUPPORTED", "CONTRADICTED"):
        shown = r.get("confidence_calibrated")
        if shown is None:
            shown = r["confidence"]
        lines += ["<b>How sure am I?</b>", f"<code>{bar(shown)}</code> {pct(shown)}, {sureness(shown)}"]
        if r.get("confidence_calibrated") is not None:
            lines.append(f"<i>The AI's raw score was {pct(r['confidence'])}. I adjust it so that \"80%\" really "
                         f"means right about 8 times in 10.</i>")
        lines.append("")
    elif r["verdict"] == "CONFLICTING":
        c = counts(r)
        lines += ["<b>The split</b>", f"{c['agree']} paper(s) agree, {c['disagree']} disagree.", ""]

    ev = r.get("evidence", [])
    decisive = [e for e in ev if e["judgement"] != "NEUTRAL" and e["nli_score"] >= thr]
    if decisive:
        lines.append("<b>What the papers say</b>")
        for i, e in enumerate(decisive[:3], 1):
            icon, word = JUDGEMENTS[e["judgement"]]
            lines.append(f"{i}. {icon} <b>{word}</b> · {esc(clip(e['title'], 110))}")
            if quote(e, 230):
                lines.append(f"    “{quote(e, 230)}”")
        if len(ev) > len(decisive[:3]):
            lines.append(f"<i>…plus {len(ev) - len(decisive[:3])} more. Tap <b>All papers</b> to see them.</i>")
        lines.append("")
    elif ev:
        lines += ["<b>Closest papers I found</b>",
                  *[f"• {esc(clip(e['title'], 110))}" for e in ev[:3]],
                  "<i>These are about the same topic but don't clearly agree or disagree.</i>", ""]
    else:
        lines += ["<i>I found no papers related to this claim at all.</i>", ""]

    library = f"{corpus_size:,} " if corpus_size else ""
    lines += ["<b>Keep in mind</b>", KEEP_IN_MIND.get(r["verdict"], ""),
              f"I only search a library of {library}biomedical paper summaries and get about 6 in 10 claims right "
              f"on a standard test. <b>This is not medical advice.</b>"]
    if r.get("cached"):
        lines.append("<i>(You or someone else checked this exact claim recently, so this answer is instant.)</i>")
    return fit("\n".join(lines))


def format_papers(r: dict) -> str:
    ev = r.get("evidence", [])
    if not ev:
        return "I didn't find any papers for this claim."
    thr = (r.get("settings") or {}).get("min_confidence", 0.5)
    lines = [f"<b>All {len(ev)} papers I read</b>", f"<i>for: {esc(clip(r['claim'], 150))}</i>", ""]
    for i, e in enumerate(ev, 1):
        icon, word = JUDGEMENTS.get(e["judgement"], ("–", e["judgement"]))
        weak = e["judgement"] != "NEUTRAL" and e["nli_score"] < thr
        status = f"{word}, but too unsure to count" if weak else word
        lines += [f"<b>{i}. {esc(clip(e['title'], 150))}</b>",
                  f"{icon} {status} · AI certainty {pct(e['nli_score'])} · paper #{e['doc_id']}"]
        if quote(e, 300):
            lines.append(f"“{quote(e, 300)}”")
        lines.append("")
    lines.append("<i>\"AI certainty\" is how sure the model is about agree / disagree / doesn't say for that one "
                 f"paper. A paper only counts towards the verdict at {pct(thr)} or more.</i>")
    return fit("\n".join(lines))


def format_why(r: dict) -> str:
    c, ev = counts(r), r.get("evidence", [])
    thr = (r.get("settings") or {}).get("min_confidence", 0.5)
    v = VERDICTS.get(r["verdict"], VERDICTS["UNKNOWN"])
    conclusion = {
        "SUPPORTED": "only agreeing papers counted, so the verdict is <b>Supported</b>.",
        "CONTRADICTED": "only disagreeing papers counted, so the verdict is <b>Contradicted</b>.",
        "CONFLICTING": "papers counted on <b>both</b> sides, so the verdict is <b>Mixed</b>.",
        "UNKNOWN": "no paper counted on either side, so the verdict is <b>Not enough evidence</b>.",
    }[r["verdict"] if r["verdict"] in VERDICTS else "UNKNOWN"]
    lines = [f"<b>How I reached \"{v['title']}\"</b>", "",
             "<b>1. Search.</b> I looked through the library for the papers whose summaries share the most "
             f"important words with your claim, and kept the top {len(ev)}.",
             "",
             f"<b>2. Read.</b> An AI model ({esc(judge_short(r.get('judge', '')))}) read each summary next to your "
             f"claim and decided whether the paper <i>agrees</i>, <i>disagrees</i> or <i>doesn't say</i>. A paper "
             f"only counts if the model is at least {pct(thr)} sure.",
             "",
             f"<b>3. Count.</b> {c['agree']} agreed, {c['disagree']} disagreed, {c['uncounted']} didn't count.",
             "",
             "<b>4. Decide.</b> The rule is simple:",
             "• only agreeing papers → Supported",
             "• only disagreeing papers → Contradicted",
             "• both → Mixed",
             "• neither → Not enough evidence",
             "",
             f"Here, {conclusion}"]
    if r.get("elapsed_seconds"):
        lines += ["", f"<i>This took {r['elapsed_seconds']:.1f} seconds.</i>"]
    return "\n".join(lines)


def format_counter(c: dict) -> str:
    how = {"antonym": f"swapped “{esc((c.get('changed') or ['', ''])[0])}” for “{esc((c.get('changed') or ['', ''])[1])}”",
           "negate": f"added “{esc((c.get('changed') or ['', ''])[1])}”",
           "remove-negation": f"removed “{esc((c.get('changed') or ['', ''])[0])}”",
           "prefix": "added “It is not the case that”"}.get(c.get("method"), "reversed it")
    label, explain = STATUS.get(c["status"], ("Can't tell", ""))
    short = lambda v: VERDICTS.get(v, VERDICTS["UNKNOWN"])  # noqa: E731
    return fit("\n".join([
        f"<b>Opposite-claim check: {label}</b>", "",
        "A good fact-checker should <b>not</b> agree with both a claim and its opposite. So I flipped your "
        "claim and checked that too.", "",
        f"<b>Your claim</b>\n<i>{esc(c['claim'])}</i>\n→ {short(c['verdict'])['icon']} {short(c['verdict'])['short']}", "",
        f"<b>The opposite</b> (I {how})\n<i>{esc(c['counter_claim'])}</i>\n→ "
        f"{short(c['counter_verdict'])['icon']} {short(c['counter_verdict'])['short']}", "",
        f"<b>What this tells us</b>\n{explain}",
    ]))


def format_progress(claim: str, stage: str, papers: Optional[List[dict]] = None, judged: int = 0,
                    corpus_size: Optional[int] = None) -> str:
    library = f"{corpus_size:,} research summaries" if corpus_size else "the research library"
    head = f"<b>Checking your claim</b>\n<i>{esc(clip(claim, 300))}</i>\n"
    if stage == "loading":
        return head + ("\nThe AI model is <b>waking up</b>. The first check after a restart takes about a "
                       "minute; after that it's a few seconds. Hang tight, I'll keep going automatically.")
    if stage == "searching" or not papers:
        return head + f"\n<b>Step 1 of 2</b> · Searching {library} for related papers…"
    found = "\n".join(f"{i}. {esc(clip(p['title'], 90))}" for i, p in enumerate(papers, 1))
    return head + (f"\n✔ <b>Step 1 of 2</b> · Found {len(papers)} related paper(s):\n{found}\n\n"
                   f"<b>Step 2 of 2</b> · Reading each one to see if it agrees or disagrees… "
                   f"<code>{judged}/{len(papers)}</code>")


def gold_line(r: dict, gold: str) -> str:
    right = r["verdict"] == gold
    return (f"\n\n<b>Dataset answer:</b> {VERDICTS[gold]['short']} (decided by expert annotators).\n"
            + ("✔ I got this one right." if right else "✘ I got this one wrong. That happens; see the "
                                                        "\"Keep in mind\" note above."))


HELP_TEXT = """<b>How to use SciVerify</b>

<b>Just type a claim</b> and send it. That's it. I'll search scientific papers and tell you whether they agree.

<b>Good claims</b> state one specific thing:
• <i>Aspirin reduces the risk of heart attack.</i>
• <i>Smoking increases the risk of lung cancer.</i>
• <i>Vitamin C prevents the common cold.</i>

<b>Less good</b>
• Questions: <i>Is coffee bad?</i> (try <i>Coffee increases blood pressure.</i>)
• Very vague: <i>Sugar is bad.</i>
• Several claims in one message

<b>Commands</b>
/example · check a random claim from a real research dataset
/quiz · guess the answer yourself, then see what the papers say
/verdicts · what the four possible answers mean
/how · how I work, step by step
/judge · choose which AI model reads the papers
/about · my limits, and what happens to your messages
/help · this message

<b>Under every answer</b> you'll find buttons to see all the papers, why I decided what I did, and an
\"opposite claim\" test of whether I can be trusted on that claim."""

VERDICTS_TEXT = """<b>The four answers I can give</b>

✔ <b>Supported</b>
The papers I found back up the claim.

✘ <b>Contradicted</b>
The papers I found say the opposite.

⇄ <b>Mixed evidence</b>
Some papers agree, some disagree. Scientists may still be debating it, or the answer depends on details (dose, age, type of study).

? <b>Not enough evidence</b>
I couldn't find papers that clearly agree or disagree. <b>This is not the same as "false"</b>. It usually means my library doesn't cover the topic.

<b>"How sure am I?"</b> is shown for Supported and Contradicted. It's adjusted (calibrated) so that "80% sure" means I'm right about 8 times in 10 on test claims."""

HOW_TEXT = """<b>How SciVerify works</b>

<b>1. Search</b>
I have a library of about 5,000 summaries of biomedical research papers. I find the ones that share the most important words with your claim, like a search engine.

<b>2. Read</b>
An AI model trained for "does this text agree with that sentence?" reads each summary next to your claim. For each paper it says <i>agrees</i>, <i>disagrees</i> or <i>doesn't say</i>, and how sure it is.

<b>3. Decide</b>
A simple, transparent rule turns those into one answer: only agreeing papers → Supported; only disagreeing → Contradicted; both → Mixed; neither → Not enough evidence.

<b>4. Show my work</b>
Every answer lists the papers and the exact sentences I relied on, so you can check me.

<b>Extra honesty checks</b>
• My confidence is <i>calibrated</i>, so the percentages mean what they say.
• The <b>Check the opposite</b> button flips your claim and checks that too. If I agree with both, you shouldn't trust me on that one."""

ABOUT_TEXT = """<b>About SciVerify</b>

A student research project that checks scientific claims against published research summaries (the SciFact dataset).

<b>Limits, honestly</b>
• I only know about 5,000 biomedical papers, mostly medicine and biology. I'm not the whole internet.
• On a standard test I get about 6 in 10 claims fully right. Use me as a starting point, not a final answer.
• I read paper <i>summaries</i>, not full papers.
• <b>I am not a doctor. Nothing I say is medical advice.</b>

<b>Your messages</b>
Your claims are sent only to the SciVerify server run by whoever set up this bot. Nothing is shared with anyone else. I remember your last few results only so the buttons under them keep working."""

START_TEXT = """<b>Hi{name}! I'm SciVerify.</b>

I check <b>scientific claims</b> against real research papers and tell you, in plain words, whether the research agrees.

<b>Try it now:</b> just send me a sentence like
<i>Exercise improves memory in older adults.</i>

I'll show you:
• my verdict (supported, contradicted, mixed, or not enough evidence)
• how sure I am
• the actual papers and sentences I used

Not sure what to ask? Tap a button below."""


# ------------------------------------------------------------------ API client
class ServerDown(Exception):
    pass


class BadClaim(Exception):
    pass


class SciVerifyClient:
    """Async wrapper around the SciVerify HTTP API."""

    def __init__(self, base_url: str, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.base = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base, transport=transport,
                                      timeout=httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=30.0))

    async def _call(self, method: str, path: str, **kw) -> dict:
        try:
            resp = await self.http.request(method, path, **kw)
        except httpx.TransportError as exc:
            raise ServerDown(str(exc)) from exc
        if resp.status_code == 422:
            detail = resp.json().get("detail")
            raise BadClaim(detail if isinstance(detail, str) else "That claim couldn't be checked.")
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        return await self._call("GET", "/status")

    async def judges(self) -> dict:
        return await self._call("GET", "/judges")

    async def warmup(self, model: Optional[str] = None) -> dict:
        return await self._call("POST", "/warmup", params={"model": model} if model else None)

    async def example(self, label: Optional[str] = None) -> dict:
        params = {"n": 1, "seed": random.randrange(10**9)}
        if label:
            params["label"] = label
        claims = (await self._call("GET", "/examples", params=params))["claims"]
        if not claims:
            raise ServerDown("The dataset isn't available on the server.")
        return claims[0]

    async def verify(self, claim: str, model: Optional[str] = None) -> dict:
        return await self._call("POST", "/verify", json={"claim": claim, "model": model})

    async def counterclaim(self, claim: str, model: Optional[str] = None) -> dict:
        return await self._call("POST", "/counterclaim", json={"claim": claim, "model": model})

    async def verify_stream(self, claim: str, model: Optional[str],
                            on_event: Callable[[dict], Awaitable[None]]) -> dict:
        """Calls on_event for each progress event and returns the final result."""
        try:
            async with self.http.stream("POST", "/verify/stream", json={"claim": claim, "model": model}) as resp:
                if resp.status_code == 422:
                    detail = json.loads(await resp.aread()).get("detail")
                    raise BadClaim(detail if isinstance(detail, str) else "That claim couldn't be checked.")
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event["event"] == "error":
                        raise RuntimeError(event.get("detail", "unknown server error"))
                    if event["event"] == "done":
                        return event["result"]
                    await on_event(event)
        except httpx.TransportError as exc:
            raise ServerDown(str(exc)) from exc
        raise RuntimeError("The server stopped before finishing.")

    async def close(self) -> None:
        await self.http.aclose()


def server_down_text(api_url: str) -> str:
    return ("<b>I can't reach the SciVerify server right now.</b>\n\n"
            "It might be restarting. Please try again in a minute.\n\n"
            "<i>If you run this bot:</i> start the server with\n"
            f"<code>uvicorn api.main:app --port 8000</code>\n(the bot is looking for it at {esc(api_url)})")


def public_web_url(base: str) -> Optional[str]:
    """Telegram rejects buttons that link to localhost, so only offer the web app when it's public."""
    host = urlparse(base).hostname or ""
    return None if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or not host else base.rstrip("/")


# ------------------------------------------------------------------ Telegram handlers
def build_application(token: str, api_url: str, web_url: Optional[str] = None,
                      client: Optional[SciVerifyClient] = None):
    from telegram import (BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions,
                          ReplyKeyboardRemove, Update)
    from telegram.constants import ChatAction, ParseMode
    from telegram.error import BadRequest
    from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, Defaults,
                              MessageHandler, filters)

    api = client or SciVerifyClient(api_url)
    web = public_web_url(web_url or api_url)
    Ctx = ContextTypes.DEFAULT_TYPE

    def start_buttons():
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("Try an example claim", callback_data="example"),
             InlineKeyboardButton("Quiz me", callback_data="quiz")],
            [InlineKeyboardButton("What do the answers mean?", callback_data="verdicts"),
             InlineKeyboardButton("How does it work?", callback_data="how")],
            [InlineKeyboardButton("Tips for good claims", callback_data="help")],
        ])

    def result_buttons(rid: int, r: dict, extra: Optional[list] = None):
        rows = [[InlineKeyboardButton("All papers", callback_data=f"papers:{rid}"),
                 InlineKeyboardButton("Why this verdict?", callback_data=f"why:{rid}")]]
        if r.get("evidence"):
            rows.append([InlineKeyboardButton("Check the opposite", callback_data=f"opp:{rid}")])
        if web:
            q = urlencode({"claim": r["claim"]})
            rows.append([InlineKeyboardButton("Open in the web app", url=f"{web}/#/verify?{q}")])
        if extra:
            rows.append(extra)
        return InlineKeyboardMarkup(rows)

    def remember(ctx: Ctx, result: dict) -> int:
        store = ctx.chat_data.setdefault("results", {})
        rid = ctx.chat_data.get("next_id", 1)
        ctx.chat_data["next_id"] = rid + 1
        store[rid] = result
        for old in sorted(store)[:-KEEP_RESULTS]:
            del store[old]
        return rid

    async def corpus_size(ctx: Ctx) -> Optional[int]:
        if not ctx.bot_data.get("corpus_size"):
            try:
                ctx.bot_data["corpus_size"] = (await api.status()).get("corpus_size")
            except Exception:  # noqa: BLE001
                return None
        return ctx.bot_data.get("corpus_size")

    async def safe_edit(msg, text: str, **kw) -> None:
        try:
            await msg.edit_text(text, **kw)
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                raise

    async def run_check(chat, ctx: Ctx, claim: str, tips: Optional[List[str]] = None,
                        gold: Optional[str] = None, extra_button=None) -> Optional[dict]:
        """Verify with live progress in one message, then turn that message into the answer."""
        size = await corpus_size(ctx)
        msg = await chat.send_message(format_progress(claim, "searching", corpus_size=size))
        await chat.send_action(ChatAction.TYPING)
        papers: List[dict] = []
        state = {"judged": 0, "last": 0.0}

        async def on_event(ev: dict) -> None:
            if ev["event"] == "loading":
                await safe_edit(msg, format_progress(claim, "loading"))
            elif ev["event"] == "retrieved":
                papers[:] = ev["papers"]
                await safe_edit(msg, format_progress(claim, "read", papers, 0, size))
                state["last"] = time.monotonic()
            elif ev["event"] == "judged":
                state["judged"] += 1
                if time.monotonic() - state["last"] > 1.2:     # Telegram limits edits per chat
                    state["last"] = time.monotonic()
                    await safe_edit(msg, format_progress(claim, "read", papers, state["judged"], size))
                    await chat.send_action(ChatAction.TYPING)

        try:
            r = await api.verify_stream(claim, ctx.chat_data.get("judge"), on_event)
        except ServerDown:
            await safe_edit(msg, server_down_text(api.base))
            return None
        except BadClaim as exc:
            await safe_edit(msg, f"I couldn't check that: {esc(exc)}\n\nTry sending a full sentence.")
            return None
        rid = remember(ctx, r)
        text = format_result(r, tips, size)
        if gold:
            text = fit(text + gold_line(r, gold))
        await safe_edit(msg, text, reply_markup=result_buttons(rid, r, extra_button))
        return r

    # ---- commands
    async def cmd_start(update: Update, ctx: Ctx) -> None:
        user = update.effective_user
        name = f", {esc(user.first_name)}" if user and user.first_name else ""
        await update.effective_chat.send_message(START_TEXT.format(name=name), reply_markup=start_buttons())

    async def cmd_help(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message(HELP_TEXT, reply_markup=start_buttons())

    async def cmd_verdicts(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message(VERDICTS_TEXT)

    async def cmd_how(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message(HOW_TEXT)

    async def cmd_about(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message(ABOUT_TEXT)

    async def cmd_example(update: Update, ctx: Ctx) -> None:
        chat = update.effective_chat
        try:
            ex = await api.example(label=random.choice(["SUPPORTED", "CONTRADICTED"]))
        except ServerDown:
            await chat.send_message(server_down_text(api.base))
            return
        await chat.send_message("Here's a real claim from the <b>SciFact</b> research dataset. Experts have already "
                                "labelled it, so you can see if I get it right.")
        await run_check(chat, ctx, ex["claim"], gold=ex["gold_verdict"],
                        extra_button=[InlineKeyboardButton("Another example", callback_data="example")])

    async def cmd_quiz(update: Update, ctx: Ctx) -> None:
        chat = update.effective_chat
        try:
            ex = await api.example(label=random.choice(list(QUIZ_CHOICES)))
        except ServerDown:
            await chat.send_message(server_down_text(api.base))
            return
        qid = ctx.chat_data.get("next_quiz", 1)
        ctx.chat_data["next_quiz"] = qid + 1
        ctx.chat_data.setdefault("quizzes", {})[qid] = ex
        score = ctx.user_data.get("quiz", {"right": 0, "total": 0})
        tally = f"\n\n<i>Your score so far: {score['right']}/{score['total']}</i>" if score["total"] else ""
        await chat.send_message(
            "<b>Quiz time!</b>\nHere's a real claim from a scientific dataset. What do you think the research "
            f"says?\n\n<i>{esc(ex['claim'])}</i>{tally}",
            reply_markup=InlineKeyboardMarkup(
                [[InlineKeyboardButton(label, callback_data=f"ans:{qid}:{key}")] for key, label in QUIZ_CHOICES.items()]))

    async def cmd_judge(update: Update, ctx: Ctx) -> None:
        chat = update.effective_chat
        try:
            data = await api.judges()
        except ServerDown:
            await chat.send_message(server_down_text(api.base))
            return
        current = ctx.chat_data.get("judge") or data["default"]
        lines = ["<b>Choose who reads the papers</b>",
                 "Different AI models can do the \"reading\" step. They find the same papers but may judge them "
                 "differently.", ""]
        buttons = []
        for j in data["judges"]:
            mark = " (current)" if j["key"] == current else ""
            lines.append(f"<b>{esc(j['name'])}</b>{mark}\n{esc(j.get('description') or '')}\n")
            buttons.append([InlineKeyboardButton(("✔ " if mark else "") + judge_short(j["name"]),
                                                 callback_data=f"judge:{j['key']}")])
        await chat.send_message(fit("\n".join(lines)), reply_markup=InlineKeyboardMarkup(buttons))

    # ---- plain messages
    async def on_text(update: Update, ctx: Ctx) -> None:
        chat, text = update.effective_chat, (update.message.text or "").strip()
        low = re.sub(r"[^a-z ]", "", text.lower()).strip()
        if low in GREETINGS:
            await cmd_start(update, ctx)
            return
        if low in THANKS:
            await chat.send_message("You're welcome! Send another claim any time, or try /quiz.")
            return
        problem = claim_problem(text)
        if problem:
            await chat.send_message(problem, reply_markup=start_buttons())
            return
        await run_check(chat, ctx, text, tips=claim_tips(text))

    async def on_other(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message(
            "I can only read <b>text</b> for now. Please type the claim you'd like me to check.",
            reply_markup=ReplyKeyboardRemove())

    async def on_unknown_command(update: Update, ctx: Ctx) -> None:
        await update.effective_chat.send_message("I don't know that command. Here's what I can do:")
        await cmd_help(update, ctx)

    # ---- buttons
    async def on_button(update: Update, ctx: Ctx) -> None:
        q = update.callback_query
        await q.answer()
        chat, data = update.effective_chat, q.data or ""
        simple = {"example": cmd_example, "quiz": cmd_quiz, "verdicts": cmd_verdicts, "how": cmd_how,
                  "help": cmd_help}
        if data in simple:
            await simple[data](update, ctx)
            return
        kind, _, rest = data.partition(":")

        if kind == "judge":
            ctx.chat_data["judge"] = rest
            try:
                info = next((j for j in (await api.judges())["judges"] if j["key"] == rest), None)
                await api.warmup(rest)
            except ServerDown:
                await chat.send_message(server_down_text(api.base))
                return
            if info is None:
                await chat.send_message("That model isn't available any more. Use /judge to pick another.")
                return
            note = "" if info.get("ready") else ("\n\nIt's loading now. Your next check may take about a minute; "
                                                 "after that it's quick.")
            await safe_edit(q.message, f"✔ Switched to <b>{esc(info['name'])}</b>.\n{esc(info.get('description') or '')}"
                                       f"{note}\n\nSend a claim to try it. Use /judge to switch again.")
            return

        if kind == "ans":
            qid, _, guess = rest.partition(":")
            ex = ctx.chat_data.get("quizzes", {}).pop(int(qid), None)
            if ex is None:
                await chat.send_message("That quiz question has expired. Here's a fresh one!")
                await cmd_quiz(update, ctx)
                return
            gold = ex["gold_verdict"]
            score = ctx.user_data.setdefault("quiz", {"right": 0, "total": 0})
            score["total"] += 1
            score["right"] += guess == gold
            verdict = ("✔ <b>Correct!</b>" if guess == gold else
                       f"✘ <b>Not quite.</b> The experts said <b>{QUIZ_CHOICES[gold]}</b>.")
            await safe_edit(q.message, f"<b>Quiz</b>\n<i>{esc(ex['claim'])}</i>\n\nYou said <b>{QUIZ_CHOICES[guess]}</b>. "
                                       f"{verdict}\nYour score: {score['right']}/{score['total']}\n\n"
                                       "Now let's see whether <b>I</b> get it right…")
            await run_check(chat, ctx, ex["claim"], gold=gold,
                            extra_button=[InlineKeyboardButton("Next question", callback_data="quiz")])
            return

        r = ctx.chat_data.get("results", {}).get(int(rest)) if rest.isdigit() else None
        if r is None:
            await chat.send_message("Sorry, I no longer remember that answer (I keep only the most recent ones, "
                                    "and forget everything when I restart). Please send the claim again.")
            return
        if kind == "papers":
            await chat.send_message(format_papers(r))
        elif kind == "why":
            await chat.send_message(format_why(r))
        elif kind == "opp":
            msg = await chat.send_message("Flipping your claim into its opposite and checking that too. This takes "
                                          "about as long as the first check…")
            await chat.send_action(ChatAction.TYPING)
            try:
                c = await api.counterclaim(r["claim"], r.get("judge_key"))
            except ServerDown:
                await safe_edit(msg, server_down_text(api.base))
                return
            except BadClaim as exc:
                await safe_edit(msg, f"I couldn't build an opposite for that claim: {esc(exc)}")
                return
            await safe_edit(msg, format_counter(c))

    async def on_error(update: object, ctx: Ctx) -> None:
        log.exception("Error while handling an update", exc_info=ctx.error)
        if isinstance(update, Update) and update.effective_chat:
            try:
                await update.effective_chat.send_message(
                    "Sorry, something went wrong on my side. Please try again in a moment. If it keeps happening, "
                    "try a shorter or simpler claim.")
            except Exception:  # noqa: BLE001
                pass

    async def post_init(app) -> None:
        await app.bot.set_my_commands([
            BotCommand("start", "Welcome and quick start"),
            BotCommand("example", "Check a random claim from a research dataset"),
            BotCommand("quiz", "Guess first, then see what the papers say"),
            BotCommand("verdicts", "What my four answers mean"),
            BotCommand("how", "How I work, step by step"),
            BotCommand("judge", "Choose which AI model reads the papers"),
            BotCommand("about", "My limits and your privacy"),
            BotCommand("help", "Tips for writing good claims"),
        ])
        try:
            await api.warmup()
            log.info("SciVerify API reachable at %s; model warming up.", api.base)
        except ServerDown:
            log.warning("SciVerify API not reachable at %s. Start it with: uvicorn api.main:app --port 8000",
                        api.base)

    async def post_shutdown(app) -> None:
        await api.close()

    app = (Application.builder().token(token)
           .defaults(Defaults(parse_mode=ParseMode.HTML, link_preview_options=LinkPreviewOptions(is_disabled=True)))
           .concurrent_updates(True)
           .post_init(post_init).post_shutdown(post_shutdown).build())
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("verdicts", cmd_verdicts))
    app.add_handler(CommandHandler("how", cmd_how))
    app.add_handler(CommandHandler("about", cmd_about))
    app.add_handler(CommandHandler("example", cmd_example))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("judge", cmd_judge))
    app.add_handler(MessageHandler(filters.COMMAND, on_unknown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.StatusUpdate.ALL, on_other))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_error_handler(on_error)
    return app


SETUP_HELP = """
No Telegram bot token found.

  1. Open Telegram and message @BotFather
  2. Send /newbot, pick a name and a username ending in "bot"
  3. BotFather replies with a token like 123456789:AAH...
  4. Create a file called .env in the SciVerify folder containing:

         TELEGRAM_BOT_TOKEN=123456789:AAH...

  5. Run again:  python -m bots.telegram_bot
"""


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print(SETUP_HELP, file=sys.stderr)
        sys.exit(1)
    api_url = os.environ.get("SCIVERIFY_API", "http://127.0.0.1:8000")
    app = build_application(token, api_url, os.environ.get("SCIVERIFY_WEB_URL"))
    log.info("SciVerify bot running. Open your bot in Telegram and send /start. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
