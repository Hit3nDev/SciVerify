"""Counter-claim consistency check.

A judge that really reads the evidence should give opposite answers to a claim and to its
opposite. One that only matches keywords will happily "support" both. So for a claim we also
build its opposite, verify that too, and compare the two verdicts:

    claim SUPPORTED  + opposite CONTRADICTED  -> consistent   (or the other way round)
    claim SUPPORTED  + opposite SUPPORTED     -> inconsistent: the verdict is unreliable
    either side UNKNOWN or CONFLICTING        -> inconclusive

The opposite is built with three small, readable rules, in this order:
  1. a claim that already says "not" / "no" has that negation removed;
  2. otherwise a direction word from a short antonym list is swapped ("increases" ->
     "decreases"), which is exactly how the SciFact annotators wrote contradicting claims;
  3. otherwise "not" is inserted at the first auxiliary / copula, or the first verb becomes
     "does not <verb>" ("do not <verb>" after a plural subject).
If none applies, the claim is wrapped as "It is not the case that ...", which is always
grammatical. Every result says which rule was used, so the check is never a black box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

_PAIRS = [
    ("increase", "decrease"), ("increases", "decreases"), ("increased", "decreased"), ("increasing", "decreasing"),
    ("higher", "lower"), ("high", "low"), ("more", "less"), ("greater", "smaller"), ("larger", "smaller"),
    ("promotes", "inhibits"), ("promote", "inhibit"), ("promoted", "inhibited"),
    ("activates", "inhibits"), ("activate", "inhibit"), ("activated", "inhibited"),
    ("enhances", "reduces"), ("enhance", "reduce"), ("enhanced", "reduced"),
    ("improves", "worsens"), ("improve", "worsen"), ("improved", "worsened"),
    ("better", "poorer"), ("worse", "better"), ("poor", "good"),
    ("positive", "negative"), ("positively", "negatively"),
    ("upregulated", "downregulated"), ("upregulates", "downregulates"), ("upregulation", "downregulation"),
    ("up-regulated", "down-regulated"), ("up-regulation", "down-regulation"),
    ("elevated", "reduced"), ("elevates", "reduces"), ("elevation", "reduction"),
    ("faster", "slower"), ("longer", "shorter"), ("stronger", "weaker"),
    ("protective", "harmful"), ("beneficial", "harmful"), ("resistant", "sensitive"), ("resistance", "sensitivity"),
    ("gain", "loss"), ("accelerates", "delays"), ("accelerate", "delay"),
    ("essential", "dispensable"), ("required", "dispensable"), ("necessary", "unnecessary"),
    ("sufficient", "insufficient"), ("frequent", "rare"), ("common", "rare"),
    ("reduces", "increases"), ("reduce", "increase"), ("reduced", "increased"), ("reduction", "increase"),
    ("decreases", "increases"), ("decrease", "increase"), ("decreased", "increased"),
    ("lower", "higher"), ("low", "high"), ("less", "more"), ("fewer", "more"),
    ("lowers", "raises"), ("lowered", "raised"), ("raises", "lowers"), ("raised", "lowered"),
    ("inhibits", "promotes"), ("inhibit", "promote"), ("inhibited", "promoted"), ("inhibition", "promotion"),
    ("suppresses", "promotes"), ("suppress", "promote"), ("suppressed", "promoted"),
    ("impairs", "improves"), ("impair", "improve"), ("impaired", "improved"),
    ("worsens", "improves"), ("worsen", "improve"), ("worsened", "improved"),
    ("poorer", "better"), ("negative", "positive"), ("negatively", "positively"),
    ("downregulated", "upregulated"), ("downregulates", "upregulates"), ("downregulation", "upregulation"),
    ("down-regulated", "up-regulated"), ("down-regulation", "up-regulation"),
    ("slower", "faster"), ("shorter", "longer"), ("weaker", "stronger"),
    ("harmful", "protective"), ("sensitive", "resistant"), ("sensitivity", "resistance"),
    ("loss", "gain"), ("delays", "accelerates"), ("delay", "accelerate"),
    ("dispensable", "essential"), ("unnecessary", "necessary"), ("insufficient", "sufficient"), ("rare", "common"),
]
ANTONYMS = {}
for _a, _b in _PAIRS:
    ANTONYMS.setdefault(_a, _b)

_AUX = re.compile(r"\b(is|are|was|were|can|could|may|might|will|would|should|must|does|do|did)\b(?!\s+not\b)",
                  re.IGNORECASE)
_HAVE = re.compile(r"\b(has|have|had)\b(?!\s+not\b)(?:\s+(\w+))?", re.IGNORECASE)
_PARTICIPLE_HINTS = {"been", "shown", "found", "led", "made", "become", "gone", "done", "seen", "taken",
                     "given", "grown", "undergone", "proven", "risen", "fallen", "begun"}
_NOT = re.compile(r"\b(does|do|did)\s+not\s+|\bnot\s+|\bno\s+", re.IGNORECASE)
_VERB_S = re.compile(r"\b([A-Za-z]{4,}?)(ies|es|s)\b")
# Word shapes that are nouns, not third-person verbs ("elements", "mixtures", "biomaterials").
_NOUN_SUFFIXES = ("ment", "tion", "sion", "ure", "ity", "ism", "ance", "ence", "ogy", "ide", "ine", "ase",
                  "ist", "ics", "ium", "ial", "oma", "ome", "ite", "ule", "ole", "ane", "ene", "one", "ate")
_BEFORE_NOUN = {"of", "in", "on", "to", "with", "for", "by", "from", "the", "a", "an", "these", "those",
                "their", "its", "and", "or", "than", "among", "between", "into", "via", "as", "at", "all",
                "most", "many", "some", "two", "three", "several", "both", "various", "other", "such"}
_FUNCTION_WORDS = {"of", "in", "on", "to", "with", "for", "by", "from", "and", "or", "than", "that", "which",
                   "the", "an", "as", "at", "into", "via", "after", "before", "during", "between", "among",
                   "over", "under", "are", "were", "have", "had", "can", "may", "will", "not", "who", "whose",
                   "but", "when", "where", "while", "also", "only", "often", "both", "either", "more", "less"}
_NOT_VERBS = {"this", "is", "was", "has", "does", "as", "its", "vs", "plus", "versus", "thus", "various",
              "numerous", "previous", "analysis", "diagnosis", "prognosis", "metastasis", "apoptosis", "status",
              "levels", "cells", "patients", "mice", "rats", "studies", "genes", "proteins", "effects", "outcomes",
              "results", "species", "tissues", "tumors", "tumours", "diseases", "cancers", "women", "humans",
              "individuals", "adults", "children", "infants", "participants", "subjects", "trials", "doses",
              "days", "weeks", "months", "years", "hours", "minutes", "times", "rates", "risks", "factors",
              "receptors", "antibodies", "molecules", "pathways", "mutations", "variants", "markers", "samples",
              "controls", "cases", "countries", "regions", "areas", "populations", "animals", "models",
              "process", "across", "unless", "less", "mass", "loss", "stress", "express", "progress", "access"}


@dataclass
class CounterClaim:
    text: str
    method: str                                # antonym | negate | remove-negation | prefix
    changed: Optional[Tuple[str, str]] = None   # (from, to) when a word was swapped


def _match_case(src: str, repl: str) -> str:
    return repl.capitalize() if src[:1].isupper() else repl


def build_counterclaim(claim: str) -> CounterClaim:
    text = claim.strip()
    if not text:
        raise ValueError("The claim is empty.")

    neg = _NOT.search(text)
    if neg:
        hit = neg.group(0)
        repl = ""
        if hit.lower().startswith("no "):
            repl = "a "
        elif hit.lower().startswith(("does ", "do ", "did ")):
            repl = hit.split()[0] + " "
        out = text[: neg.start()] + _match_case(hit, repl) + text[neg.end():]
        return CounterClaim(out[:1].upper() + out[1:], "remove-negation", (hit.strip(), repl.strip() or "(removed)"))

    for m in re.finditer(r"[A-Za-z][A-Za-z-]*", text):
        word = m.group(0)
        anti = ANTONYMS.get(word.lower())
        if anti and word.lower() in ("lower", "raise") and text[: m.start()].rstrip().lower().endswith(" to"):
            anti = "raise" if word.lower() == "lower" else "lower"      # "to lower X" is a verb
        if anti:
            out = text[: m.start()] + _match_case(word, anti) + text[m.end():]
            return CounterClaim(out, "antonym", (word, anti))

    aux = _AUX.search(text)
    have = _HAVE.search(text)
    first = min((x for x in (aux, have) if x), key=lambda x: x.start(), default=None)
    if first is aux and aux is not None:
        out = text[: aux.end()] + " not" + text[aux.end():]
        return CounterClaim(out, "negate", (aux.group(1), aux.group(1) + " not"))
    if first is have and have is not None:
        verb, nxt = have.group(1), (have.group(2) or "")
        if nxt.lower() in _PARTICIPLE_HINTS or nxt.lower().endswith(("ed", "en")):
            out = text[: have.start(1) + len(verb)] + " not" + text[have.start(1) + len(verb):]
            return CounterClaim(out, "negate", (verb, verb + " not"))
        do = {"has": "does not have", "have": "do not have", "had": "did not have"}[verb.lower()]
        out = text[: have.start(1)] + _match_case(verb, do) + text[have.start(1) + len(verb):]
        return CounterClaim(out, "negate", (verb, do))

    for m in _VERB_S.finditer(text):
        word, stem, suffix = m.group(0), m.group(1), m.group(2)
        if m.start() == 0 or word[:1].isupper():
            continue
        prev = re.findall(r"[A-Za-z-]+", text[: m.start()])
        following = re.match(r"\s+([A-Za-z]+)", text[m.end():])
        nxt = following.group(1) if following else ""
        looks_like_noun = (word.lower() in _NOT_VERBS or (stem + ("e" if suffix == "es" else "")).lower().endswith(_NOUN_SUFFIXES)
                           or (prev and prev[-1].lower() in _BEFORE_NOUN))
        if looks_like_noun:
            # A plural subject followed by its verb: "biomaterials show" -> "biomaterials do not show".
            if (nxt and nxt.islower() and len(nxt) >= 3 and not nxt.endswith(("s", "ed", "ing", "ly"))
                    and nxt not in _FUNCTION_WORDS):
                out = text[: following.start(1) + m.end()] + "do not " + text[following.start(1) + m.end():]
                return CounterClaim(out, "negate", (nxt, "do not " + nxt))
            continue
        if suffix == "ies":
            base = stem + "y"
        elif suffix == "es" and stem.endswith(("s", "x", "z", "ch", "sh")):
            base = stem
        else:
            base = stem + ("e" if suffix == "es" else "")
        out = text[: m.start()] + "does not " + base + text[m.end():]
        return CounterClaim(out, "negate", (word, "does not " + base))

    body = text[:1].lower() + text[1:]
    return CounterClaim("It is not the case that " + body, "prefix", None)


def judge_consistency(verdict: str, counter_verdict: str) -> Tuple[str, str]:
    """(status, plain-English note) for a claim's verdict and its opposite's verdict."""
    decided = {"SUPPORTED", "CONTRADICTED"}
    if verdict in decided and counter_verdict in decided:
        if verdict != counter_verdict:
            return "consistent", ("The judge gives opposite answers to the claim and to its opposite, "
                                  "which is what reading the evidence should produce.")
        return "inconsistent", (f"The judge says {verdict} for both the claim and its opposite. "
                                "It is reacting to the words, not the meaning; do not trust this verdict.")
    if "CONFLICTING" in (verdict, counter_verdict):
        return "inconclusive", "At least one side came back CONFLICTING, so the two verdicts cannot be compared."
    return "inconclusive", "At least one side came back UNKNOWN, so the two verdicts cannot be compared."
