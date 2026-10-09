"""Social-media noise for training rows, applied as batches are built (--augment SHARE).

The sources are almost all punctuated prose. Weibo posts run on with little punctuation,
so punctuation became a shortcut. Latin rows are left alone. Over two runs each,
lowercasing moved WNUT person recall by -0.002 and cost Argilla 0.013, and hashtags and
emoji on Latin rows still cost Argilla 0.010, with no WNUT gain in either arm.

Every transform keeps each entity and ignore span on the same characters. No transform
inserts an @handle: an unlabelled handle would teach that usernames are not PII.
"""

import random
import unicodedata

HAN_SCRIPTS = ("zh", "zh-Hans", "zh-Hant")
SOCIAL_MARKS = ("#日常", "#心情", "#分享", "#tbt", "#fyi", "😂", "🙏", "❤️", "😭", "👍", "🔥", "~~", "!!!", "哈哈哈")


def _spans(row: dict) -> list[dict]:
    return row["entities"] + row["ignore"]


def _protected(row: dict) -> list[bool]:
    inside = [False] * len(row["text"])
    for span in _spans(row):
        inside[span["start"] : span["end"]] = [True] * (span["end"] - span["start"])
    return inside


def _moved(row: dict, text: str, position: list[int]) -> dict:
    """The row with new text. position[i] is where old character i lands, for i up to len."""

    def shift(span: dict) -> dict:
        return {**span, "start": position[span["start"]], "end": position[span["end"]]}

    return {
        **row,
        "text": text,
        "entities": [shift(s) for s in row["entities"]],
        "ignore": [shift(s) for s in row["ignore"]],
    }


def lowercased(row: dict) -> dict:
    """Lowercase the whole row. A character whose lowercase has another length (İ) would
    move every offset after it, so such a row stays as it is."""
    lower = row["text"].lower()
    return {**row, "text": lower} if len(lower) == len(row["text"]) else row


def _is_punctuation(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


def without_punctuation(row: dict) -> dict:
    """Drop punctuation and spaces outside every span, as in a run-on post."""
    inside = _protected(row)
    kept, position = [], []
    for index, character in enumerate(row["text"]):
        position.append(len(kept))
        if inside[index] or not _is_punctuation(character):
            kept.append(character)
    position.append(len(kept))
    return _moved(row, "".join(kept), position)


def with_social_marks(row: dict, picker: random.Random) -> dict:
    """Put a hashtag, emoji or filler after one or two clause ends outside spans."""
    inside = _protected(row)
    text = row["text"]
    edges = [i + 1 for i, c in enumerate(text) if _is_punctuation(c) and not inside[i]] + [len(text)]
    span_ends = {span["end"] for span in _spans(row)}
    edges = [edge for edge in edges if edge not in span_ends and (edge == len(text) or not inside[edge])]
    if not edges:
        return row
    chosen = sorted(picker.sample(edges, min(len(edges), picker.randint(1, 2))))
    pieces, position, previous = [], [], 0
    added = 0
    for edge in chosen:
        pieces.append(text[previous:edge])
        position += [index + added for index in range(previous, edge)]
        mark = picker.choice(SOCIAL_MARKS)
        pieces.append(mark)
        added += len(mark)
        previous = edge
    pieces.append(text[previous:])
    position += [index + added for index in range(previous, len(text) + 1)]
    return _moved(row, "".join(pieces), position)


def augmented(row: dict, picker: random.Random, share: float) -> dict:
    """With probability share, give a Chinese row one transform. Latin rows stay as they are."""
    if row["script"] not in HAN_SCRIPTS or picker.random() >= share:
        return row
    return without_punctuation(row) if picker.random() < 0.5 else with_social_marks(row, picker)


# Capitalized words that are not personal names. None of them is a product the planted
# transcript set (scratch-research/plant.py) uses as a negative, so that set stays a test.
TOOL_SWAP_WORDS = (
    "Nginx Webpack Ansible Airflow Jupyter Pandas Tableau Figma Notion Vercel Heroku Celery Django Flask "
    "Hadoop Mongo Bazel Gradle Maven Cargo Rollup Babel Ember Angular Astro Remix Fastly Okta Twilio"
).split()
# Plain and invented words only. R62 swapped in tool names too, and its tool-name false alarms
# rose: a tool name labelled person teaches that tool names can be people.
PLAIN_SWAP_WORDS = (
    "Window Garden Orange Silver Market Bridge Harbor Copper Meadow Summit Lantern Cobalt Velvet Granite "
    "Trelvo Quandix Morvane Zeltrin Ospery Calvix Durnell Feyra Brisko Nalthor"
).split()
SWAP_WORDS = TOOL_SWAP_WORDS + PLAIN_SWAP_WORDS


def context_swapped(row: dict, picker: random.Random, words: list[str] = SWAP_WORDS) -> dict:
    """Fill every person span with one or two capitalized words that are not names, keeping the
    person label. The words carry no name signal, so the label has to come from the context.
    Ghaddar et al. (TACL 2021) call the opposite habit name regularity bias: a name-like word
    is tagged person whatever the sentence says."""
    people = sorted((s for s in row["entities"] if s["label"] == "person"), key=lambda s: s["start"])
    if not people:
        return row
    text, pieces, position, previous, added = row["text"], [], [], 0, 0
    for span in people:
        if span["start"] < previous:
            continue
        filler = " ".join(picker.sample(words, 2 if " " in text[span["start"] : span["end"]] else 1))
        pieces.append(text[previous : span["start"]])
        position += [index + added for index in range(previous, span["start"])]
        position += [span["start"] + added + min(i, len(filler)) for i in range(span["end"] - span["start"])]
        pieces.append(filler)
        added += len(filler) - (span["end"] - span["start"])
        previous = span["end"]
    pieces.append(text[previous:])
    position += [index + added for index in range(previous, len(text) + 1)]
    return _moved(row, "".join(pieces), position)


def swapped(row: dict, picker: random.Random, share: float, words: list[str] = SWAP_WORDS) -> dict:
    """With probability share, fill a Latin row's person spans with words that are not names."""
    if row["script"] in HAN_SCRIPTS or picker.random() >= share:
        return row
    return context_swapped(row, picker, words)


# Developer-context sources, where a lone first name or handle is common in real transcripts
# and rare in training: 53 of r61's 55 judged dev misses are single-token values.
DEV_SOURCES = ("synthetic/dev-transcripts", "synthetic/llm-devwork", "synthetic/llm-agentwork")
LONE_FIRST = (
    "Anika Bastian Callum Delphine Eitan Fatima Gideon Hiro Isla Jasper Keiko Lorenzo Maeve Nikolai "
    "Orla Quentin Rania Silas Thea Umar Vera Wesley Ximena Yusuf Zoe Adaeze Bjorn Carmen Dario Esme "
    "Farid Gwen Hamza Imogen Jonas Kira Leon Mila Nils Odette Paulo Ronja Soren Talia Uriel Valentina"
).split()
LONE_WORDS = ("dev", "ops", "code", "hub", "lab", "io", "x")
# Each template holds one {v}. The planted evaluation sets use other templates.
LONE_TEMPLATES = (
    ("person", "  owner: {v}"),
    ("person", "- [ ] {v}: rotate the staging certificates"),
    ("person", "    // {v} asked for this guard, keep it"),
    ("person", "assigned_to={v}"),
    ("person", "{v} approved these changes"),
    ("person", "merged by {v} 2 hours ago"),
    ("person", "on-call this week: {v}"),
    ("person", "{v}: can we ship the hotfix before standup?"),
    ("username", "drwxr-xr-x   5 {v}  staff   160 Oct  3 10:12 build"),
    ("username", "{v}      48213   0.0  0.3  python worker.py"),
    ("username", "@{v} commented 3 days ago"),
    ("username", "/srv/home/{v}/projects/app/main.py:42: warning: unused import"),
)


def lone_value(label: str, picker: random.Random) -> str:
    first = picker.choice(LONE_FIRST)
    if label == "person":
        return first.lower() if picker.random() < 0.3 else first
    shapes = (first.lower(), f"{first.lower()}{picker.randint(2, 99)}", f"{first.lower()}-{picker.choice(LONE_WORDS)}")
    return picker.choice(shapes)


def _insertion_point(row: dict, picker: random.Random) -> int:
    """A random line start that no span covers or ends at."""
    inside, ends = _protected(row), {span["end"] for span in _spans(row)}
    starts = [0] + [i + 1 for i, c in enumerate(row["text"]) if c == "\n" and i + 1 not in ends and not (i + 1 < len(inside) and inside[i + 1])]
    return picker.choice(starts)


def _inserted(row: dict, at: int, line: str) -> dict:
    text = row["text"][:at] + line + row["text"][at:]
    position = [i if i < at else i + len(line) for i in range(len(row["text"]) + 1)]
    return _moved(row, text, position)


def with_lone_name(row: dict, picker: random.Random) -> dict:
    """Insert one line holding a lone first name or handle at a line start no span covers."""
    at = _insertion_point(row, picker)
    label, template = picker.choice(LONE_TEMPLATES)
    value = lone_value(label, picker)
    line = template.replace("{v}", value) + "\n"
    moved = _inserted(row, at, line)
    start = at + line.index(value)
    return {**moved, "entities": sorted(moved["entities"] + [{"start": start, "end": start + len(value), "label": label}], key=lambda s: s["start"])}


def lone_named(row: dict, picker: random.Random, share: float) -> dict:
    """With probability share, give a developer-context row one lone name or handle line."""
    if row.get("source") not in DEV_SOURCES or picker.random() >= share:
        return row
    return with_lone_name(row, picker)


# Name-like tool names for contrastive lines: the same developer registers as LONE_TEMPLATES,
# no label. R63 alone raised false alarms on tool names. These words appear in no planted
# evaluation list and in neither swap list.
LONE_TOOLS = (
    "Hermes Apollo Phoenix Crystal Ray Jax Meteor Falcon Electron Deno Chai Sinon Nomad Consul Relay "
    "Keras Tornado Unity Quarkus Grails Spring Argo Istio Envoy Caddy Kong Eureka Puma Chef Salt Gerrit "
    "Loki Tempo Thanos Cortex Jaeger Zipkin Solr Druid Trino Hive Storm Heron"
).split()
# A wider list for R66: more name-like tool and project names, also none in an evaluation,
# swap or name list. R65's 44 words did not stop false alarms on unseen tool names.
WIDE_LONE_TOOLS = LONE_TOOLS + (
    "Percy Drone Flux Helm Tilt Rancher Falco Clair Sonar Mend Atlantis Kitchen Locust Gatling Artillery "
    "Vegeta Siege Varnish Squid Tyk Ambassador Contour Skipper Fabio Cube Pulsar Rabbit Neon Ghost Wagtail "
    "Plone Drupal Odoo Frappe Matrix Element Synapse Conduit Jitsi Restic Borg Velero Stash Rook Ceph Lustre "
    "Bower Grunt Gulp Parcel Vitest Playwright Puppeteer Appium Detox Storybook Buildkite Concourse Pulumi "
    "Crossplane Molecule Superset Metabase Redash Dremio Patroni Vitess Citus Hasura Strapi Directus "
    "Mattermost Zulip Nextcloud Syncthing Longhorn Minio Mosquitto Zookeeper Etcd Podman Kaniko Cosign Trivy"
).split()
TOOL_TEMPLATES = (
    "Restarting {t} service... done",
    "  image: {l}:2.4.1",
    "{t} 3.2.1 installed successfully",
    "    // {t} drops the header here, keep the retry",
    "merged by CI: bump {t} to 2.0",
    "on-call runbook: restart {t} first",
    "@{l}/core@4.1.0",
    "drwxr-xr-x   5 root  staff   160 Oct  3 10:12 {l}",
    "{t}: connection refused, retrying in 5s",
    "/opt/{l}/bin/{l} --config /etc/{l}.yaml",
)


def with_tool_line(row: dict, picker: random.Random, tools: list[str] = LONE_TOOLS) -> dict:
    """Insert one line that holds a name-like tool name, unlabelled, at a line start no span covers."""
    at = _insertion_point(row, picker)
    tool = picker.choice(tools)
    line = picker.choice(TOOL_TEMPLATES).replace("{t}", tool).replace("{l}", tool.lower()) + "\n"
    return _inserted(row, at, line)


def tool_lined(row: dict, picker: random.Random, share: float, tools: list[str] = LONE_TOOLS) -> dict:
    """With probability share, give a developer-context row one unlabelled tool-name line."""
    if row.get("source") not in DEV_SOURCES or picker.random() >= share:
        return row
    return with_tool_line(row, picker, tools)
