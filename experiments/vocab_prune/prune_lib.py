"""Pure helpers for the vocabulary pruning experiment.

Nothing here touches torch or the network, so the test suite runs in a second.
"""

import json
import math
import re
from collections import Counter
from functools import cache
from urllib.parse import urlsplit

OUTSIDE = "O"
SPECIAL_PIECES = ("<s>", "<pad>", "</s>", "<unk>", "<mask>")
IGNORE = -100
SPACE_MARK = "▁"


def bioes_labels(entity_types: list[str]) -> list[str]:
    labels = [OUTSIDE]
    for entity_type in entity_types:
        labels += [f"{prefix}-{entity_type}" for prefix in "BIES"]
    return labels


def effective_spans(tokens: list[str], offsets: list[tuple[int, int]]) -> list[tuple[int, int] | None]:
    """Character span each token really covers, or None for tokens that carry no text.

    The Metaspace pre-tokenizer emits a bare space-mark token before CJK runs, and the
    fast tokenizer gives it the offset of the next character. Treating it as covering
    that character would put a B tag on a space, so it covers nothing.
    """
    spans = []
    for token, (start, end) in zip(tokens, offsets, strict=True):
        if end <= start or token == SPACE_MARK:
            spans.append(None)
        else:
            spans.append((start, end))
    return spans


def token_tags(spans: list[tuple[int, int] | None], entities: list[dict]) -> list[str | None]:
    """BIOES tag per token. None marks a token the loss and the decoder skip."""
    tags: list[str | None] = [None if span is None else OUTSIDE for span in spans]
    for entity in entities:
        inside = [index for index, span in enumerate(spans) if span and _overlaps(span, entity)]
        _tag_entity(tags, inside, entity["label"])
    return tags


def _overlaps(span: tuple[int, int], entity: dict) -> bool:
    return span[0] < entity["end"] and entity["start"] < span[1]


def _tag_entity(tags: list[str | None], inside: list[int], label: str) -> None:
    if not inside:
        return
    if len(inside) == 1:
        tags[inside[0]] = f"S-{label}"
        return
    tags[inside[0]] = f"B-{label}"
    for index in inside[1:-1]:
        tags[index] = f"I-{label}"
    tags[inside[-1]] = f"E-{label}"


def boundary_misaligned(spans: list[tuple[int, int] | None], entity: dict) -> bool:
    """True when the token grid cannot represent the entity's exact boundaries."""
    covered = [span for span in spans if span and _overlaps(span, entity)]
    if not covered:
        return True
    return covered[0][0] != entity["start"] or covered[-1][1] != entity["end"]


def allowed_transition(previous: str, current: str) -> bool:
    """Hard BIOES constraints. The same rule set the hook's decoder applies."""
    previous_prefix, previous_type = _split(previous)
    current_prefix, current_type = _split(current)
    if previous_prefix in ("B", "I"):
        return current_prefix in ("I", "E") and current_type == previous_type
    return current_prefix in ("O", "B", "S")


def _split(tag: str) -> tuple[str, str | None]:
    if tag == OUTSIDE:
        return OUTSIDE, None
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def viterbi(log_probs: list[list[float]], labels: list[str]) -> list[int]:
    """Best label path under the hard BIOES constraints. Biases are all zero."""
    count = len(labels)
    allowed = [[allowed_transition(labels[a], labels[b]) for b in range(count)] for a in range(count)]
    can_start = [_split(label)[0] in ("O", "B", "S") for label in labels]
    can_end = [_split(label)[0] in ("O", "E", "S") for label in labels]
    scores = [log_probs[0][b] if can_start[b] else -math.inf for b in range(count)]
    back: list[list[int]] = []
    for row in log_probs[1:]:
        step_scores, step_back = _viterbi_step(scores, row, allowed)
        scores, back = step_scores, back + [step_back]
    final = [scores[b] if can_end[b] else -math.inf for b in range(count)]
    path = [max(range(count), key=final.__getitem__)]
    for step_back in reversed(back):
        path.append(step_back[path[-1]])
    return path[::-1]


def _viterbi_step(scores: list[float], row: list[float], allowed: list[list[bool]]) -> tuple[list[float], list[int]]:
    count = len(row)
    step_scores, step_back = [], []
    for b in range(count):
        candidates = [scores[a] if allowed[a][b] else -math.inf for a in range(count)]
        best = max(range(count), key=candidates.__getitem__)
        step_scores.append(candidates[best] + row[b])
        step_back.append(best)
    return step_scores, step_back


def decode_spans(tags: list[str], spans: list[tuple[int, int]], confidence: list[float] | None = None) -> list[dict]:
    """Turn a valid BIOES tag path over real tokens into character spans.

    With confidence, the probability the model gave each chosen tag, a span's score is its
    least sure token: one doubtful token is enough to doubt the whole span."""
    found, start, least = [], None, 1.0

    def close(span_start: int, span_end: int, entity_type: str, score: float) -> None:
        span = {"start": span_start, "end": span_end, "label": entity_type}
        found.append(span if confidence is None else {**span, "score": score})

    for index, (tag, (token_start, token_end)) in enumerate(zip(tags, spans, strict=True)):
        prefix, entity_type = _split(tag)
        sure = confidence[index] if confidence else 1.0
        if prefix == "S":
            close(token_start, token_end, entity_type, sure)
        elif prefix == "B":
            start, least = token_start, sure
        elif prefix in ("I", "E"):
            least = min(least, sure)
            if prefix == "E" and start is not None:
                close(start, token_end, entity_type, least)
                start = None
    return found


def merge_adjacent(spans: list[dict], text: str, max_gap: int = 1) -> list[dict]:
    """Label-agnostic merge of spans separated by at most max_gap whitespace characters.

    Taxonomies disagree on granularity: one set tags GIVEN_NAME and SURNAME apart,
    another tags the whole name. The merge lets a label-agnostic score compare them.
    """
    merged: list[dict] = []
    for span in sorted(spans, key=lambda item: item["start"]):
        gap = text[merged[-1]["end"] : span["start"]] if merged else None
        if merged and len(gap) <= max_gap and not gap.strip():
            merged[-1] = {"start": merged[-1]["start"], "end": max(merged[-1]["end"], span["end"]), "label": "PII"}
        else:
            merged.append({"start": span["start"], "end": span["end"], "label": "PII"})
    return merged


def span_counts(gold: list[dict], predicted: list[dict], typed: bool) -> Counter:
    """True positives, false positives and false negatives for exact span match."""
    key = (
        (lambda span: (span["start"], span["end"], span["label"]))
        if typed
        else (lambda span: (span["start"], span["end"]))
    )
    gold_keys, predicted_keys = Counter(map(key, gold)), Counter(map(key, predicted))
    hits = sum((gold_keys & predicted_keys).values())
    return Counter(tp=hits, fp=sum(predicted_keys.values()) - hits, fn=sum(gold_keys.values()) - hits)


def overlap_counts(gold: list[dict], predicted: list[dict]) -> Counter:
    """Lenient match: a span counts when it shares at least one character with the other side."""
    gold_hit = sum(1 for span in gold if any(_spans_overlap(span, other) for other in predicted))
    predicted_hit = sum(1 for span in predicted if any(_spans_overlap(span, other) for other in gold))
    return Counter(tp_gold=gold_hit, fn=len(gold) - gold_hit, tp_pred=predicted_hit, fp=len(predicted) - predicted_hit)


def _spans_overlap(first: dict, second: dict) -> bool:
    return first["start"] < second["end"] and second["start"] < first["end"]


def f1_report(counts: Counter) -> dict:
    true_positive_for_precision = counts.get("tp_pred", counts["tp"])
    true_positive_for_recall = counts.get("tp_gold", counts["tp"])
    precision = _ratio(true_positive_for_precision, true_positive_for_precision + counts["fp"])
    recall = _ratio(true_positive_for_recall, true_positive_for_recall + counts["fn"])
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, **dict(counts)}


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def select_pieces(vocab: list[tuple[str, float]], frequencies: Counter, target_size: int) -> list[int]:
    """Old ids to keep, in original order.

    Kept unconditionally: special pieces and every single-character piece, so the pruned
    Unigram model can still segment any text the original could and adds no new unknowns.
    The rest of the budget goes to multi-character pieces by corpus frequency.
    """
    forced = {index for index, (piece, _) in enumerate(vocab) if piece in SPECIAL_PIECES or len(piece) == 1}
    if len(forced) > target_size:
        raise ValueError(f"{len(forced)} forced pieces exceed the target of {target_size}")
    ranked = [index for index, _ in frequencies.most_common() if index not in forced]
    chosen = forced | set(ranked[: target_size - len(forced)])
    if len(chosen) < target_size:
        raise ValueError(f"corpus covers only {len(chosen)} pieces, below the target of {target_size}")
    return sorted(chosen)


def with_punctuation_split(tokenizer_json: str) -> str:
    """Isolate every punctuation character as its own pre-token, after Metaspace.

    The XLM-R vocabulary holds pieces such as ':09' and '06)', and the normalizer turns
    a full-width colon into ':'. A span boundary inside such a piece cannot be tagged,
    which made 6.6% of the Traditional held-out spans unreachable. Splitting after
    Metaspace adds no extra space marks. The mark itself is a symbol, not punctuation.
    """
    spec = json.loads(tokenizer_json)
    split = {"type": "Split", "pattern": {"Regex": r"\p{P}"}, "behavior": "Isolated", "invert": False}
    spec["pre_tokenizer"]["pretokenizers"].append(split)
    return json.dumps(spec, ensure_ascii=False)


def pruned_tokenizer_json(tokenizer_json: str, kept_ids: list[int]) -> str:
    """Rewrite a Unigram tokenizer.json so it holds only kept_ids, renumbered densely."""
    spec = json.loads(tokenizer_json)
    if spec["model"]["type"] != "Unigram":
        raise ValueError(f"expected a Unigram model, got {spec['model']['type']}")
    vocab = spec["model"]["vocab"]
    new_id = {old: new for new, old in enumerate(kept_ids)}
    spec["model"]["vocab"] = [vocab[old] for old in kept_ids]
    spec["model"]["unk_id"] = new_id[spec["model"]["unk_id"]]
    for added in spec["added_tokens"]:
        added["id"] = new_id[added["id"]]
    _renumber_post_processor(spec.get("post_processor"), new_id)
    return json.dumps(spec, ensure_ascii=False)


def _renumber_post_processor(post_processor: dict | None, new_id: dict[int, int]) -> None:
    if not post_processor:
        return
    for token in post_processor.get("special_tokens", {}).values():
        token["ids"] = [new_id[old] for old in token["ids"]]


@cache
def _bioes_tables(labels: tuple[str, ...]):
    """Transition scores and the legal first and last tags. Built once per label set: the
    5,329 Python calls cost 0.84 ms, 15% of a request."""
    import numpy

    allowed = numpy.array([[allowed_transition(a, b) for b in labels] for a in labels])
    can_start = numpy.array([_split(label)[0] in ("O", "B", "S") for label in labels])
    can_end = numpy.array([_split(label)[0] in ("O", "E", "S") for label in labels])
    return numpy.where(allowed, 0.0, -numpy.inf), can_start, can_end


def transition_matrix(labels: list[str]):
    """Additive transition scores: 0 where allowed, -inf where BIOES forbids it."""
    return _bioes_tables(tuple(labels))[0]


def viterbi_numpy(log_probs, labels: list[str], transitions) -> list[int]:
    """Vectorised twin of viterbi(). The test suite pins the two to the same answer.

    Transitions are 0 or -inf, so a greedy path that BIOES allows scores the most of any
    path and is the Viterbi answer. 83% of chunks take that shortcut."""
    import numpy

    _, can_start, can_end = _bioes_tables(tuple(labels))
    greedy = log_probs.argmax(axis=1)
    legal = can_start[greedy[0]] and can_end[greedy[-1]] and not numpy.isinf(transitions[greedy[:-1], greedy[1:]]).any()
    if legal:
        return greedy.tolist()
    scores = numpy.where(can_start, log_probs[0], -numpy.inf)
    back = []
    for row in log_probs[1:]:
        candidates = scores[:, None] + transitions
        best = candidates.argmax(axis=0)
        scores = candidates[best, numpy.arange(len(labels))] + row
        back.append(best)
    path = [int(numpy.where(can_end, scores, -numpy.inf).argmax())]
    for best in reversed(back):
        path.append(int(best[path[-1]]))
    return path[::-1]


SENTENCE_ENDS = "。！？\n"


def chunk_text(text: str, max_chars: int) -> list[tuple[int, str]]:
    """Pack whole sentences into chunks of at most max_chars, as (offset, chunk) pairs.

    A single sentence longer than max_chars is cut hard at max_chars.
    """
    sentences, start = [], 0
    for index, character in enumerate(text):
        if character in SENTENCE_ENDS:
            sentences.append((start, index + 1))
            start = index + 1
    if start < len(text):
        sentences.append((start, len(text)))
    pieces = [(s, min(s + max_chars, e)) for s, e in sentences for s in range(s, e, max_chars)]
    chunks: list[tuple[int, int]] = []
    for piece_start, piece_end in pieces:
        if chunks and piece_end - chunks[-1][0] <= max_chars:
            chunks[-1] = (chunks[-1][0], piece_end)
        else:
            chunks.append((piece_start, piece_end))
    return [(chunk_start, text[chunk_start:chunk_end]) for chunk_start, chunk_end in chunks]


JOIN_TOUCHING = {"secret"}
CALIBRATION_EPSILON = 1e-6


def score_logit(score: float) -> float:
    score = min(max(score, CALIBRATION_EPSILON), 1 - CALIBRATION_EPSILON)
    return math.log(score / (1 - score))


def calibrated_score(maps: dict, label: str, score: float) -> float:
    """Platt map p = sigmoid(a * logit(score) + b), per label, else the global map "*"."""
    a, b = maps.get(label, maps["*"])
    return 1 / (1 + math.exp(-(a * score_logit(score) + b)))


def calibrated_spans(spans: list[dict], maps: dict | None) -> list[dict]:
    """Spans with probability scores; without maps, the raw scores stay."""
    if maps is None:
        return spans
    return [{**span, "score": calibrated_score(maps, span["label"], span["score"])} for span in spans]


def join_touching(spans: list[dict]) -> list[dict]:
    """Two secret spans that touch are one secret. The model sometimes ends a long key at a
    subword boundary, and a key longer than a chunk ends at the chunk edge. Names can touch
    in Chinese (張三李四), so no other label joins. The joined span keeps the lower score."""
    joined: list[dict] = []
    for span in sorted(spans, key=lambda item: item["start"]):
        previous = joined[-1] if joined else None
        touching = previous is not None and previous["label"] == span["label"] and previous["end"] == span["start"]
        if touching and span["label"] in JOIN_TOUCHING:
            joined[-1] = {**previous, "end": span["end"]}
            if "score" in span:
                joined[-1]["score"] = min(previous["score"], span["score"])
        else:
            joined.append(span)
    return joined


# The guideline marks a street-level address, never a city or district alone. Only Chinese
# has a closed set of street-level characters; street names in other languages carry no
# common marker (Jalan Muhibbah, Intrarea Tabacu), so the check covers Han spans only.
HAN = re.compile(r"[\u4e00-\u9fff]")
STREET_PART = re.compile(r"[0-9０-９路街道巷弄段號号樓楼室幢栋棟座]|單元|单元")
# Documentation, package, standards and reserved hosts point at no private person.
PUBLIC_HOST_SUFFIXES = (
    "example.com",
    "example.org",
    "example.net",
    ".example",
    ".test",
    ".invalid",
    "localhost",
    "readthedocs.io",
    "pypi.org",
    "npmjs.com",
    "w3.org",
    "schema.org",
    # Licence, standards, language and vendor documentation hosts: library code links them.
    "apache.org",
    "opensource.org",
    "gnu.org",
    "ietf.org",
    "openid.net",
    "wikipedia.org",
    "wikimedia.org",
    "python.org",
    "golang.org",
    "rust-lang.org",
    "nodejs.org",
    "kubernetes.io",
    "k8s.io",
    "docker.com",
    "learn.microsoft.com",
    "developers.google.com",
    "cloud.google.com",
    "googleapis.com",
    "amazonaws.com",
    "developer.apple.com",
    "developer.mozilla.org",
    "stackoverflow.com",
    "stackexchange.com",
    "cdnjs.cloudflare.com",
    "jsdelivr.net",
    "unpkg.com",
    "platform.openai.com",
)
PUBLIC_HOST_PREFIXES = ("docs.", "schemas.")
# A user page on a code host is one path segment (github.com/name); a repository has two.
REPO_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "huggingface.co")
# Code reads: a call, an attribute, an assignment target or a definition. Prose words such
# as "from" are not cues: "a message from jake_99" names a user. A call has no space
# before its parenthesis; "user cpohl (email ...)" does.
IDENTIFIER_AFTER = re.compile(r"^(?:\(|\s*=(?!=))")
IDENTIFIER_BEFORE = re.compile(r"(?:\.|\b(?:def|class)\s+)$")
PERSONAL_PATH = re.compile(r"/(?:users?|u)/[^/]+|/~[^/]+|/@[^/]+", re.IGNORECASE)
ROLE_MAILBOXES = {
    "noreply",
    "no-reply",
    "support",
    "info",
    "admin",
    "press",
    "service",
    "security",
    "abuse",
    "postmaster",
}
DEFAULT_ACCOUNTS = {"root", "admin", "postgres", "ubuntu", "ec2-user", "guest", "nobody", "www-data", "sa"}


def public_url(value: str) -> bool:
    """A URL on a public host, with no personal path and no user:password part. A span that
    does not parse (an unclosed [ reads as a broken IPv6 host) is not shown public, so it stays."""
    try:
        parts = urlsplit(value if "://" in value else f"http://{value}")
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if "://" not in value and "." not in host:
        return True  # docs/source/page.md is a path, not a URL
    segments = [segment for segment in parts.path.split("/") if segment]
    repository = host.endswith(REPO_HOSTS) and len(segments) >= 2
    public = host.endswith(PUBLIC_HOST_SUFFIXES) or host.startswith(PUBLIC_HOST_PREFIXES) or repository
    return public and parts.username is None and not PERSONAL_PATH.search(parts.path)


def shared_email(value: str) -> bool:
    """An organisation's shared mailbox. Reserved domains stay: synthetic sources give
    private people example.com addresses, and 42% of dev-split gold emails use one."""
    return value.lower().partition("@")[0] in ROLE_MAILBOXES


def _policy_drops(span: dict, value: str) -> bool:
    if span["label"] == "address":
        return bool(HAN.search(value)) and not STREET_PART.search(value)
    if span["label"] == "url":
        return public_url(value)
    if span["label"] == "email":
        return shared_email(value)
    return span["label"] == "username" and value.lower() in DEFAULT_ACCOUNTS


def _code_identifier(span: dict, text: str) -> bool:
    """A username span that code reads as an identifier: called, an attribute, assigned or
    defined. The value's own shape is no cue: jeanette.blair and nx.Graph look alike."""
    if span["label"] != "username":
        return False
    before, after = text[: span["start"]], text[span["end"] :]
    return bool(IDENTIFIER_AFTER.match(after)) or bool(IDENTIFIER_BEFORE.search(before))


# Tool output: cat -n, grep -n and editor views put a line number at the start of a line,
# before a tab, an arrow, a bar or a colon; a traceback or a review cites path.py:325.
# A number of at most 6 digits in those places is a line number whatever the model read.
LINE_START = re.compile(r"(?:^|\n)[ \t]*$")
GUTTER_VALUE = re.compile(r"^\d{1,6}(?:[\t→|]|$)")
GUTTER_AFTER = re.compile(r"^[\t→|:]")
PATH_LINE_BEFORE = re.compile(r"\.[A-Za-z]{1,6}:$")
LINE_NUMBER = re.compile(r"^\d{1,6}$")
# A JSON key ("reason": ...), a Rust or C++ path (Mode::Json) and an all-caps heading that
# ends its line with a colon are names of things in code or a template, not of people.
JSON_KEY_BEFORE = re.compile(r"\"$")
JSON_KEY_AFTER = re.compile(r"^\"\s*:")
JSON_KEY_VALUE = re.compile(r"^[a-z_][A-Za-z0-9_]*$")  # "reason", not "Gary Lin" in a reviewer map
SCOPE = "::"
HEADING_VALUE = re.compile(r"^[A-Z][A-Z ]*[A-Z]$")
HEADING_AFTER = re.compile(r"^:[ \t]*(?:\n|$)")
# Hashes: a short git hash opening a git log line, and a UUID inside a file path (session
# folders). A bare hex or UUID value stays: many API keys and webhook tokens are one.
GIT_HASH = re.compile(r"^(?=[0-9a-f]*[a-f])[0-9a-f]{7,12}$")  # a hex letter: a 10-digit ID is no hash
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
ID_LABELS = {"secret", "national_id", "account_number", "phone", "passport", "driver_license"}


def _line_reference(span: dict, text: str) -> bool:
    value, before, after = text[span["start"] : span["end"]], text[: span["start"]], text[span["end"] :]
    gutter = LINE_START.search(before) and (
        "\t" in value[:7] and GUTTER_VALUE.match(value) or LINE_NUMBER.match(value) and GUTTER_AFTER.match(after)
    )
    return bool(gutter) or bool(LINE_NUMBER.match(value) and PATH_LINE_BEFORE.search(before))


def _code_name(span: dict, text: str) -> bool:
    if span["label"] not in {"person", "username"}:
        return False
    value, before, after = text[span["start"] : span["end"]], text[: span["start"]], text[span["end"] :]
    json_key = JSON_KEY_VALUE.match(value) and JSON_KEY_BEFORE.search(before) and JSON_KEY_AFTER.match(after)
    heading = (
        span["label"] == "person"
        and HEADING_VALUE.match(value)
        and LINE_START.search(before)
        and HEADING_AFTER.match(after)
    )
    return bool(json_key or heading or before.endswith(SCOPE) or after.startswith(SCOPE))


def _hash(span: dict, text: str) -> bool:
    if span["label"] not in ID_LABELS:
        return False
    value, before, after = text[span["start"] : span["end"]], text[: span["start"]], text[span["end"] :]
    git_line = GIT_HASH.match(value) and LINE_START.search(before) and after.startswith(" ")
    token = before.split()[-1] if before.split() else ""
    in_path = UUID.search(value) and "://" not in token and (before.endswith(("/", "-")) or after.startswith("/"))
    return bool(git_line or in_path)


def tool_rules(spans: list[dict], text: str) -> list[dict]:
    """Drop the shapes of tool output: a line number, a JSON key, a scoped code name, an
    all-caps heading, a git hash and a UUID in a file path."""
    return [s for s in spans if not (_line_reference(s, text) or _code_name(s, text) or _hash(s, text))]


# Developer text names people by handle in fields the model reads as code: a roster's
# slack or github key, author= and reviewer= in a log, and an @mention in a table cell.
# login, username and handle keys stay out: in a config file they name service accounts.
# The key stands alone (not gf-github:pr-create), a colon or a bare = follows it
# (author = x is code), and the value is no call (currentUser()).
HANDLE = r"([A-Za-z0-9][A-Za-z0-9._-]{1,38})"
HANDLE_KEY = re.compile(
    r"(?i)(?<![\w./-])(?:slack|github|gitlab|author|reviewer|assignee|telegram|twitter)"
    r"[\"']?(?:\s*:|=)\s*[\"']?@?" + HANDLE + r"(?![A-Za-z0-9._\-\[<{(])"
)
HANDLE_CELL = re.compile(r"\|\s*@" + HANDLE + r"\s*(?=\|)")
NOT_A_HANDLE = {
    "null", "none", "nil", "true", "false", "tbd", "todo", "unknown", "me", "self", "user", "username", "login",
    "example", "test", "admin", "root", "anonymous", "ghost", "human", "unassigned", "nobody", "reviewer",
    "assignee", "author", "user_name", "handle", "dependabot", "renovate", "github-actions", "coderabbit",
    "coderabbitai", "codecov", "sonarcloud", "deepsource", "sourcery-ai",
}  # fmt: skip
BOT_HANDLE = re.compile(r"(?i)(?:^|[-_.])bot$|^copilot\b")


def _handle(value: str) -> bool:
    return value.lower() not in NOT_A_HANDLE and not value.isdigit() and not BOT_HANDLE.search(value)


def username_keys(spans: list[dict], text: str) -> list[dict]:
    """Add a username span for a handle in a username field the spans do not cover."""
    added = []
    for pattern in (HANDLE_KEY, HANDLE_CELL):
        for match in pattern.finditer(text):
            span = {"start": match.start(1), "end": match.end(1), "label": "username", "score": 1.0}
            taken = [*spans, *added]
            if _handle(match.group(1)) and not any(
                s["start"] < span["end"] and span["start"] < s["end"] for s in taken
            ):
                added.append(span)
    return sorted([*spans, *added], key=lambda s: (s["start"], s["end"]))


def post_rules(spans: list[dict], text: str) -> list[dict]:
    """Drop the spans the guideline excludes by form: a Chinese place name with no street
    part, a public-host, repository or documentation URL, a path that is not a URL, a shared
    mailbox, a default account, a code identifier read as a username, and a username inside
    a dropped URL. The model flags these by shape; training them away cost recall."""
    dropped = [
        span for span in spans if _policy_drops(span, text[span["start"] : span["end"]]) or _code_identifier(span, text)
    ]
    dropped_urls = [span for span in dropped if span["label"] == "url"]
    inside_url = [
        span
        for span in spans
        if span["label"] == "username"
        and any(u["start"] <= span["start"] and span["end"] <= u["end"] for u in dropped_urls)
    ]
    return [span for span in spans if span not in dropped and span not in inside_url]


NAME_DOTS = re.compile(r"[·・‧•]")
# A public figure is rarely written next to a private phone or ID; a private namesake is.
PRIVATE_CONTEXT_CHARS = 200
NOT_CONTEXT = {"person", "date"}


def name_key(value: str) -> str:
    """One spelling per name: one kind of middle dot, single spaces, no case."""
    return " ".join(NAME_DOTS.sub("·", value).split()).casefold()


def full_name(value: str) -> bool:
    """Two Han characters or two words. A lone given name or surname never matches."""
    han = len(HAN.findall(value))
    return han >= 2 if han else len(value.split()) >= 2


def _near_private_pii(span: dict, others: list[dict]) -> bool:
    return any(
        other["start"] < span["end"] + PRIVATE_CONTEXT_CHARS and span["start"] - PRIVATE_CONTEXT_CHARS < other["end"]
        for other in others
    )


def _part_of(key: str, public_keys: list[str]) -> bool:
    """A surname or given name of a public name met earlier in the text."""
    if HAN.search(key):
        return any(key in public for public in public_keys)
    return any(set(key.split()) <= set(public.split()) for public in public_keys)


def public_filter(spans: list[dict], text: str, public_names: set[str]) -> list[dict]:
    """Drop person spans that name a listed public figure: a full-name match with no other
    PII nearby, and later partial mentions of the same person. The model reads form and
    cannot know who is famous; the list can."""
    others = [span for span in spans if span["label"] not in NOT_CONTEXT]
    persons = sorted((span for span in spans if span["label"] == "person"), key=lambda span: span["start"])
    public, public_keys = [], []
    for span in persons:
        key = name_key(text[span["start"] : span["end"]])
        listed = full_name(key) and key in public_names and not _near_private_pii(span, others)
        if listed or (public_keys and _part_of(key, public_keys)):
            public.append(span)
        if listed:
            public_keys.append(key)
    return [span for span in spans if span not in public]


# Public offices, in Traditional and Simplified. The guideline excludes politicians and
# officials; a name next to one of these titles is one. Private titles (經理, 老師) are out.
OFFICES_HAN = [
    "總統",
    "总统",
    "總理",
    "总理",
    "首相",
    "主席",
    "議員",
    "议员",
    "立委",
    "部長",
    "部长",
    "外長",
    "外长",
    "國務卿",
    "国务卿",
    "發言人",
    "发言人",
    "市長",
    "市长",
    "縣長",
    "县长",
    "州長",
    "州长",
    "省長",
    "省长",
    "議長",
    "议长",
    "大使",
    "特首",
    "局長",
    "局长",
    "署長",
    "署长",
    "司長",
    "司长",
    "總書記",
    "总书记",
    "委員長",
    "委员长",
    "國王",
    "国王",
    "女王",
    "王儲",
    "王储",
    "總督",
    "总督",
    "教宗",
    "將軍",
    "将军",
    "上將",
    "上将",
    "司令",
]
OFFICES_LATIN = (
    "President|Vice President|Senator|Sen\\.|Rep\\.|Representative|Congressman|Congresswoman|Governor|"
    "Mayor|Prime Minister|Minister|Secretary of State|Chancellor|King|Queen|Prince|Princess|Pope|Ambassador"
)
OFFICE_BEFORE = re.compile(f"(?:{'|'.join(OFFICES_HAN)})\\s*$|\\b(?:{OFFICES_LATIN})\\s+$")
OFFICE_AFTER = re.compile(f"^\\s*[副前]?(?:{'|'.join(OFFICES_HAN)})")
OFFICE_WINDOW = 12


def _titled(span: dict, text: str) -> bool:
    before = text[max(0, span["start"] - OFFICE_WINDOW) : span["start"]]
    after = text[span["end"] : span["end"] + OFFICE_WINDOW]
    return bool(
        OFFICE_BEFORE.search(before) or (HAN.search(text[span["start"] : span["end"]]) and OFFICE_AFTER.search(after))
    )


def _same_person(key: str, public_keys: list[str]) -> bool:
    """The same key, or a part of a public full name (a later surname alone)."""
    return key in public_keys or _part_of(key, [public for public in public_keys if full_name(public)])


def office_filter(spans: list[dict], text: str) -> list[dict]:
    """Drop person spans next to a public-office title (議員郭榮鏗, Senator John Smith), unless
    private PII sits nearby, and every other mention of the same name in the text."""
    others = [span for span in spans if span["label"] not in NOT_CONTEXT]
    persons = [span for span in spans if span["label"] == "person"]
    public_keys = [
        name_key(text[span["start"] : span["end"]])
        for span in persons
        if _titled(span, text) and not _near_private_pii(span, others)
    ]
    if not public_keys:
        return spans
    public = [span for span in persons if _same_person(name_key(text[span["start"] : span["end"]]), public_keys)]
    return [span for span in spans if span not in public]


# Words of the news register, Traditional and Simplified. A row that carries NEWS_MARKERS
# distinct ones is a news report, and the guideline excludes the public people news names.
NEWS_WORDS = [
    "記者",
    "记者",
    "報導",
    "报导",
    "報道",
    "报道",
    "中央社",
    "新華社",
    "新华社",
    "路透",
    "法新社",
    "美聯社",
    "美联社",
    "綜合報導",
    "综合报道",
    "發言人",
    "发言人",
    "表示",
    "指出",
    "強調",
    "强调",
    "呼籲",
    "呼吁",
    "聲明",
    "声明",
    "受訪",
    "受访",
    "專電",
    "专电",
    "電",
    "新聞",
    "新闻",
    "據報",
    "据报",
    "消息人士",
    "官員",
    "官员",
    "政府",
    "當局",
    "当局",
    "立法院",
    "國會",
    "国会",
    "議會",
    "议会",
]
NEWS_PATTERN = re.compile("|".join(sorted(set(NEWS_WORDS), key=len, reverse=True)))
NEWS_MARKERS = 4


def _news_register(text: str) -> bool:
    return len(set(NEWS_PATTERN.findall(text))) >= NEWS_MARKERS


def news_filter(spans: list[dict], text: str) -> list[dict]:
    """In a Han news report, drop person spans with no private PII within 200 characters.
    A reporter, an official or a spokesperson is public; a victim listed with a phone is
    not. Other registers are left alone, so a chat that names a journalist keeps its names."""
    if not _news_register(text):
        return spans
    others = [span for span in spans if span["label"] not in NOT_CONTEXT]
    return [
        span
        for span in spans
        if not (
            span["label"] == "person"
            and HAN.search(text[span["start"] : span["end"]])
            and not _near_private_pii(span, others)
        )
    ]


def place_filter(spans: list[dict], text: str, place_names: set[str]) -> list[dict]:
    """Drop Han person spans that name a listed place (德國, 伯恩州). The list leaves out
    every string that is also a person's name. Latin text is left alone: full names such
    as Anna Maria and Maria Helena are also towns, and the dev split holds both."""
    return [
        span
        for span in spans
        if not (
            span["label"] == "person"
            and HAN.search(key := name_key(text[span["start"] : span["end"]]))
            and full_name(key)
            and key in place_names
        )
    ]


def shifted(spans: list[dict], offset: int) -> list[dict]:
    return [{**span, "start": span["start"] + offset, "end": span["end"] + offset} for span in spans]
