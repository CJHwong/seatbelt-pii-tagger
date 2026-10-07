# /// script
# requires-python = ">=3.12"
# dependencies = ["rstr>=3.2"]
# ///
"""Generate secret training rows in code-shaped text, with provider values the rules certify.

usage: uv run --script scripts/rule_secrets.py [--train 60000] [--heldout 2000] [--seed 13]

Writes data/synthetic/rule_secrets/train.jsonl and heldout.jsonl, in the row format
build_training_set.py reads: {"text", "spans": [{start, end, label, value, rule}], "script"}.

The agent-seatbelt rules engine (pinned clone in data/external/agent-seatbelt) holds 271
provider-secret rules. For each rule, rstr samples a string from the rule's regex, and the
rule's own "value" group gives the secret inside it. A row is kept only when the rules engine,
run on the final text, reports a secret over every provider value in it. Generic secrets
(passwords, hex and base64 tokens, JWTs, private keys) are labelled by construction, because
many real ones carry too little entropy for any rule.

The rules decide nothing else. Their URL, email and phone rules flag far more than the
guideline marks, so no row copies them.

Rule ids held out by experiments/compare/seatbelt.py never reach train.jsonl. heldout.jsonl
holds only those providers, so it tests providers the model never saw.
"""

import argparse
import base64
import json
import random
import re
import string
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import rstr

ROOT = Path(__file__).resolve().parent.parent
HOOKS = ROOT / "data" / "external" / "agent-seatbelt" / "hooks-pii"
OUT = ROOT / "data" / "synthetic" / "rule_secrets"
sys.path[:0] = [str(HOOKS), str(ROOT / "experiments" / "compare")]
from pii_rules import deterministic_spans
from pii_secret_patterns import PORTED_RULES

from seatbelt import held_out

LOOKAROUND = re.compile(r"\(\?<?[=!](?:[^()]|\([^()]*\))*\)")
UNSAFE_IN_TOKEN = set("\"'\\\n\r\t ")
PROVIDER_SHARE = 0.55
NO_SECRET_SHARE = 0.2
ROW_ATTEMPTS = 8

SOURCES = {rule_id: source for rule_id, source, _, _, _ in PORTED_RULES}
COMPILED = {rule_id: re.compile(source) for rule_id, source in SOURCES.items()}

SECRET_KEYS = [
    "api_key", "api_token", "access_token", "auth_token", "client_secret", "secret_key", "password",
    "db_password", "mysql_password", "redis_password", "smtp_password", "shared_secret", "private_token",
    "webhook_secret", "session_secret", "jwt_secret", "passwd", "admin_password", "app_secret",
]  # fmt: skip
PLACEHOLDERS = {
    "<YOUR_API_KEY>", "${API_KEY}", "xxxxxxxxxxxxxxxx", "your_password_here", "REPLACE_ME", "********",
    "<token>", "TODO", "{{ secret }}", "%(password)s", "os.environ['API_KEY']", "process.env.TOKEN",
}  # fmt: skip


class Entry(NamedTuple):
    key: str | None
    value: str
    secret: bool
    rule: str | None = None
    raw: tuple[int, int] | None = None  # the secret's offsets when value is a whole rule line


def rule_ids(held: bool) -> list[str]:
    return [rule_id for rule_id in SOURCES if held_out(rule_id) == held]


def provider_value(rule_id: str, generator: random.Random) -> tuple[str, int, int] | None:
    """A string the rule's regex matches, and the offsets of the rule's secret group in it."""
    try:
        candidate = rstr.Rstr(generator).xeger(LOOKAROUND.sub("", SOURCES[rule_id]))
    except (ValueError, re.error, IndexError):
        return None
    match = COMPILED[rule_id].search(candidate)
    if not match or "\n" in candidate or "\r" in candidate:
        return None
    group = "value" if "value" in match.re.groupindex and match.group("value") else 0
    start, end = match.start(group), match.end(group)
    return (candidate, start, end) if end > start else None


def provider_entry(rule_id: str, generator: random.Random) -> Entry | None:
    produced = provider_value(rule_id, generator)
    if produced is None:
        return None
    line, start, end = produced
    if (start, end) == (0, len(line)) and not UNSAFE_IN_TOKEN & set(line):
        return Entry(generator.choice(SECRET_KEYS), line, True, rule_id)
    return Entry(None, line, True, rule_id, (start, end))


def password(generator: random.Random) -> str:
    words = ["summer", "taipei", "dragon", "admin", "sunshine", "coffee", "tiger", "office", "happy", "winter"]
    style = generator.randrange(4)
    if style == 0:
        return generator.choice(words).capitalize() + str(generator.randint(1, 2030)) + generator.choice("!@#$%&*")
    if style == 1:
        pool = string.ascii_letters + string.digits + "!@#$%^&*-_"
        return "".join(generator.choice(pool) for _ in range(generator.randint(10, 20)))
    if style == 2:
        return "-".join(generator.choice(words) for _ in range(3)) + str(generator.randint(10, 99))
    return generator.choice(words) + str(generator.randint(100, 999999))


def token(generator: random.Random) -> str:
    kind = generator.randrange(4)
    if kind == 0:
        return "".join(generator.choice("0123456789abcdef") for _ in range(generator.choice([32, 40, 64])))
    if kind == 1:
        raw = bytes(generator.randrange(256) for _ in range(generator.choice([24, 30, 48])))
        return base64.b64encode(raw).decode()
    if kind == 2:
        part = lambda n: (
            base64.urlsafe_b64encode(bytes(generator.randrange(256) for _ in range(n))).decode().rstrip("=")
        )
        return "eyJ" + part(24) + "." + part(48) + "." + part(32)
    return "".join(generator.choice(string.ascii_letters + string.digits) for _ in range(generator.randint(24, 48)))


def private_key(generator: random.Random) -> str:
    kind = generator.choice(["RSA PRIVATE KEY", "PRIVATE KEY", "EC PRIVATE KEY", "OPENSSH PRIVATE KEY"])
    body = base64.b64encode(bytes(generator.randrange(256) for _ in range(generator.choice([180, 360, 600])))).decode()
    lines = [body[i : i + 64] for i in range(0, len(body), 64)]
    return "\n".join([f"-----BEGIN {kind}-----", *lines, f"-----END {kind}-----"])


PUBLIC_VALUES: dict[str, Callable[[random.Random], str]] = {
    "host": lambda g: g.choice(["db.internal", "api.example.com", "10.0.3.12", "cache-01.svc.local", "localhost"]),
    "port": lambda g: str(g.choice([5432, 3306, 6379, 8080, 443, 9200])),
    "region": lambda g: g.choice(["ap-northeast-1", "us-east-1", "eu-west-2", "asia-east1"]),
    "timeout": lambda g: str(g.choice([30, 60, 120, 5000])),
    "bucket": lambda g: g.choice(["prod-assets", "logs-archive", "build-cache"]),
    "log_level": lambda g: g.choice(["INFO", "DEBUG", "WARN"]),
    "user": lambda g: g.choice(["app", "postgres", "deploy", "svc_reader", "root"]),
    "request_id": lambda g: str(uuid.UUID(int=g.getrandbits(128))),
    "commit": lambda g: "".join(g.choice("0123456789abcdef") for _ in range(40)),
    "checksum": lambda g: "sha256:" + "".join(g.choice("0123456789abcdef") for _ in range(64)),
    "build": lambda g: str(g.randint(100, 99999)),
    "endpoint": lambda g: g.choice(["https://api.example.com/v2", "https://hooks.example.org/in"]),
}


class Builder:
    def __init__(self) -> None:
        self.text, self.spans = "", []

    def add(self, piece: str) -> None:
        self.text += piece

    def value(self, entry: Entry) -> None:
        """Write an entry's value; label it when it is a secret and not a placeholder."""
        if entry.raw:
            start, end = entry.raw
            self.add(entry.value[:start])
            self.mark(entry.value[start:end], entry.rule)
            self.add(entry.value[end:])
        elif entry.secret and entry.value not in PLACEHOLDERS:
            self.mark(entry.value, entry.rule)
        else:
            self.add(entry.value)

    def mark(self, value: str, rule: str | None) -> None:
        span = {"start": len(self.text), "end": len(self.text) + len(value), "label": "secret", "value": value}
        self.spans.append({**span, "rule": rule} if rule else span)
        self.add(value)


def camel(key: str) -> str:
    head, *rest = key.split("_")
    return head + "".join(part.capitalize() for part in rest)


def line_format(prefix: Callable[[str], str], suffix: Callable[[str], str], header: str = "", indent: str = ""):
    """A context that writes one line per entry. A whole rule line is written as it is."""

    def render(entries, generator: random.Random) -> tuple[str, list[dict]]:
        builder = Builder()
        builder.add(header)
        for entry in (Entry(*e) for e in entries):
            builder.add(indent)
            if entry.raw is None:
                builder.add(prefix(entry.key))
            builder.value(entry)
            if entry.raw is None:
                builder.add(suffix(entry.key))
            builder.add("\n")
        return builder.text.rstrip("\n"), builder.spans

    return render


def json_object(entries, generator: random.Random) -> tuple[str, list[dict]]:
    builder = Builder()
    builder.add("{\n")
    for index, entry in enumerate(Entry(*e) for e in entries):
        builder.add(f'  "{entry.key}": "')
        builder.value(entry)
        builder.add('"' + ("," if index < len(entries) - 1 else "") + "\n")
    builder.add("}")
    return builder.text, builder.spans


def curl(entries, generator: random.Random) -> tuple[str, list[dict]]:
    builder = Builder()
    builder.add(f"curl -s -X POST https://api.example.com/v1/{generator.choice(['deploy', 'events', 'sync'])} \\\n")
    for entry in (Entry(*e) for e in entries):
        if entry.secret:
            builder.add('  -H "Authorization: ' + generator.choice(["Bearer ", "token ", ""]))
            builder.value(entry)
            builder.add('" \\\n')
        else:
            builder.add(f'  -d "{entry.key}=')
            builder.value(entry)
            builder.add('" \\\n')
    return builder.text.rstrip(" \\\n"), builder.spans


def database_url(entries, generator: random.Random) -> tuple[str, list[dict]]:
    builder = Builder()
    for entry in (Entry(*e) for e in entries):
        if entry.secret and entry.raw is None:
            scheme = generator.choice(["postgres", "mysql", "mongodb", "redis"])
            builder.add(f"DATABASE_URL={scheme}://{generator.choice(['app', 'svc_rw', 'admin'])}:")
            builder.value(entry)
            builder.add(f"@{PUBLIC_VALUES['host'](generator)}:{PUBLIC_VALUES['port'](generator)}/main\n")
        else:
            builder.add("" if entry.raw else f"{(entry.key or '').upper()}=")
            builder.value(entry)
            builder.add("\n")
    return builder.text.rstrip("\n"), builder.spans


PROSE = {
    "en": ["Here is the {key} for staging: {value}", "use this {key} until Friday -> {value}", "{key} = {value} (rotate after the demo)"],
    "zh-Hant": ["staging 的 {key} 是 {value}，用完記得換掉", "先用這組 {key}：{value}，別貼到群組", "{key} 我放這裡 {value} 明天再改"],
    "zh-Hans": ["测试环境的 {key} 是 {value}，别外传", "先用这个 {key}：{value}，用完就删", "{key} 给你 {value} 下班前记得换"],
}  # fmt: skip


def prose(language: str):
    def render(entries, generator: random.Random) -> tuple[str, list[dict]]:
        builder = Builder()
        for entry in (Entry(*e) for e in entries):
            template = generator.choice(PROSE[language])
            before, after = template.split("{value}")
            builder.add(before.format(key=entry.key or "config"))
            builder.value(entry)
            builder.add(after.format(key=entry.key or "config") + "\n")
        return builder.text.rstrip("\n"), builder.spans

    render.__name__ = f"prose_{language}"
    return render


CONTEXTS = {
    "env": line_format(lambda k: f"{k.upper()}=", lambda k: ""),
    "export": line_format(lambda k: f'export {k.upper()}="', lambda k: '"'),
    "yaml": line_format(lambda k: f"{k}: ", lambda k: ""),
    "k8s": line_format(lambda k: f"{k}: ", lambda k: "", "apiVersion: v1\nkind: Secret\nmetadata:\n  name: app\nstringData:\n", "  "),
    "python": line_format(lambda k: f'{k.upper()} = "', lambda k: '"'),
    "js": line_format(lambda k: f"const {camel(k)} = '", lambda k: "';"),
    "go": line_format(lambda k: f'{camel(k)} := "', lambda k: '"'),
    "toml": line_format(lambda k: f'{k} = "', lambda k: '"', "[service]\n"),
    "docker": line_format(lambda k: f"ENV {k.upper()}=", lambda k: "", "FROM python:3.12-slim\n"),
    "xml": line_format(lambda k: f"<{camel(k)}>", lambda k: f"</{camel(k)}>", "<config>\n", "  "),
    "log": line_format(lambda k: f"2026-03-02T10:15:22Z INFO worker {k}=", lambda k: ""),
    "json": json_object,
    "curl": curl,
    "database_url": database_url,
    "prose_en": prose("en"),
    "prose_zh-Hant": prose("zh-Hant"),
    "prose_zh-Hans": prose("zh-Hans"),
}  # fmt: skip
LINE_CONTEXTS = ["env", "export", "log", "database_url", "prose_en", "prose_zh-Hant", "prose_zh-Hans"]
SCRIPT_OF_CONTEXT = {"prose_zh-Hant": "zh-Hant", "prose_zh-Hans": "zh-Hans"}


def rules_detect(text: str, span: dict) -> bool:
    return any(
        found["label"] == "secret" and found["start"] < span["end"] and span["start"] < found["end"]
        for found in deterministic_spans(text)
    )


def secret_entry(allowed: list[str], generator: random.Random, provider_only: bool) -> Entry | None:
    if provider_only or generator.random() < PROVIDER_SHARE:
        return provider_entry(generator.choice(allowed), generator)
    key = generator.choice(SECRET_KEYS)
    value = password(generator) if "pass" in key or "pwd" in key else token(generator)
    return Entry(key, value, True)


def entries_for_row(allowed: list[str], generator: random.Random, provider_only: bool) -> list[Entry] | None:
    entries = [
        Entry(k, PUBLIC_VALUES[k](generator), False)
        for k in generator.sample(sorted(PUBLIC_VALUES), generator.randint(0, 4))
    ]
    if generator.random() < NO_SECRET_SHARE:
        entries.append(Entry(generator.choice(SECRET_KEYS), generator.choice(sorted(PLACEHOLDERS)), True))
    else:
        for _ in range(generator.choice([1, 1, 1, 2])):
            entry = secret_entry(allowed, generator, provider_only)
            if entry is None:
                return None
            entries.append(entry)
    generator.shuffle(entries)
    return entries


def private_key_row(generator: random.Random) -> dict:
    key = private_key(generator)
    lead = generator.choice(["", "# deploy key, do not commit\n", "cat ~/.ssh/id_rsa\n"])
    return {
        "text": lead + key,
        "spans": [{"start": len(lead), "end": len(lead) + len(key), "label": "secret", "value": key}],
    }


def one_row(allowed: list[str], generator: random.Random, provider_only: bool) -> dict | None:
    if not provider_only and generator.random() < 0.03:
        return private_key_row(generator)
    entries = entries_for_row(allowed, generator, provider_only)
    if entries is None:
        return None
    has_raw = any(entry.raw for entry in entries)
    context = generator.choice(LINE_CONTEXTS if has_raw else sorted(CONTEXTS))
    text, spans = CONTEXTS[context](entries, generator)
    if not all(rules_detect(text, span) for span in spans if span.get("rule")):
        return None
    row = {"text": text, "spans": spans}
    return {**row, "script": SCRIPT_OF_CONTEXT[context]} if context in SCRIPT_OF_CONTEXT else row


def generate(count: int, held: bool, seed: int):
    """count rows; held picks the held-out providers and provider secrets only."""
    generator, allowed, made, attempts = random.Random(seed), rule_ids(held), 0, 0
    while made < count:
        attempts += 1
        if attempts > count * ROW_ATTEMPTS:
            raise RuntimeError(f"only {made} of {count} rows after {attempts} attempts")
        row = one_row(allowed, generator, provider_only=held)
        if row is not None:
            made += 1
            yield row


def write(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=60_000)
    parser.add_argument("--heldout", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    print("train", write(OUT / "train.jsonl", generate(args.train, held=False, seed=args.seed)))
    print("heldout", write(OUT / "heldout.jsonl", generate(args.heldout, held=True, seed=args.seed + 1)))


if __name__ == "__main__":
    main()
