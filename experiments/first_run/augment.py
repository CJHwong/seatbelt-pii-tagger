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
