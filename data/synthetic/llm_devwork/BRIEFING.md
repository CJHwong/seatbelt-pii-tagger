# Briefing: LLM-written developer work prose for a PII tagger

## Why these rows exist

A small token tagger (6 layers, 35M parameters) finds personal data (PII) in English,
Simplified Chinese and Traditional Chinese. It runs on the text software engineers and
their coding agents produce. There it misses a coworker's given name in plain prose:
"ask Priya", "thanks to dan for the fix", "Wen is out today". The model saw names in
forms, chat and news, but rarely in the prose of engineering work, where most words
around a name are code, tools and tickets. These rows teach it that register.

## Hard rules

1. Write every row yourself. Do not read any file in this repository except this
   briefing, `benchmark/GUIDELINE.md`, the three check scripts named below and your own
   output files. Do not open `benchmark/benchmark.jsonl`, `benchmark/raw/`, any `blind`
   folder, any `runs/` folder, any test or evaluation data, or any dataset file. Do not
   read anything under `~/.claude` or `~/.codex`.
2. Do not fetch anything from the internet.
3. Write each row by hand. Do not write code, scripts or loops that generate rows, and do
   not fill slots in a sentence pattern. A row that differs from another only in its
   marked values or digits is a template, and `skeletons.py` rejects it.
4. Every value is fictional. Invent names, handles and emails. Do not use a real private
   person's data. Do not reuse the examples in the guideline or in this briefing.
5. Real public figures, companies, products and open-source projects may appear only
   unmarked, as the guideline says.

## Row format

One JSON object per line, UTF-8, in a `.jsonl` file:

```json
{"id": "D1-0001", "register": "standup notes", "script": "latin", "text": "..."}
```

- `id`: your writer code, a dash, a 4-digit number. Unique.
- `register`: a short English name of the genre.
- `script`: `latin` (English), `zh-Hant` or `zh-Hans`.
- `text`: the row, with every PII span marked inline as `[[type:value]]`. The value is
  the exact text as it appears. Use `\n` for line breaks. No nested marks. No `[[` or
  `]]` anywhere else.

## Registers

Write the prose engineers and their agents write around code. Mix these, and invent
others of the same kind:

- standup and status notes, weekly updates, retro notes, meeting minutes with action items
- Slack or Teams threads in an engineering channel, DMs about a bug, handoff messages
- pull-request descriptions and review comments, commit messages, release notes
- incident timelines and postmortems, on-call handoffs, runbook notes
- ticket comments (Jira, Linear, GitHub issues), sprint planning notes
- an assistant's summary to its user: "I asked Mona's question in the thread", "the
  failing test came from Ravi's change"
- Markdown tables and bullet lists in reports: owner columns, reviewer lists, rotations

About 80% English (`latin`). About 20% Traditional Chinese work chat (`zh-Hant`), the
mixed Chinese and English a Taiwanese team writes ("這個 PR 等 [[person:Kevin]] review").

## What to mark

`benchmark/GUIDELINE.md` is the labelling policy. Read it in full. In this register:

- **person**: a private person's name: a given name alone, a nickname, a lowercase name,
  a full name, a surname used alone as a name. Mark every mention, including repeats,
  possessives (mark `Ravi` in "Ravi's change") and names at the start of a line or
  inside a table cell.
- **username**: a person's handle: a GitHub login, a Slack @handle (mark it without the
  @), a personal account name in a reviewer or author field.
- **email**: a person's work or personal address.
- **secret**: a live-looking credential pasted by mistake, rarely (at most 1 row in 20).
- Do not mark: bots and service accounts (dependabot, renovate, a CI account), an AI
  assistant's persona name, teams and channels, repositories, packages, functions,
  variables, branches, ticket keys, file paths, hostnames, versions, dates, public
  figures, companies and products, including those named like people (Jenkins, Ruby,
  Ada, Julia, Django, Hugo, Sentry, Grafana).

## What makes a good row

- Names sit inside ordinary work sentences, surrounded by code words, ticket keys,
  paths and tool names. Most names are a given name alone, often lowercase in chat.
- Short rows and long rows. Several people per row is good.
- Vary everything across rows: names from many cultures (Taiwanese, Chinese, Indian,
  Vietnamese, Filipino, European, American, African, Latin American), the products and
  tools, the topics, the lengths. No person value may appear in more than 2 rows.
- About 85% of rows contain at least one marked name or handle. About 15% contain none:
  write those about bots, tools, products named like people and tickets, with no person
  in them.
- Never write a row where a private person's name appears unmarked. Unmarked names teach
  the tagger that names in prose are not PII.

## Steps

1. Read `benchmark/GUIDELINE.md`.
2. Write your rows in parts of about 50 rows each:
   `data/synthetic/llm_devwork/raw/<WRITER>_partN.jsonl`.
3. After each part, run these from the repository root and fix every reported row:
   - `python3 data/synthetic/llm_social/validate.py data/synthetic/llm_devwork/raw/<WRITER>_part*.jsonl` (0 bad rows)
   - `python3 data/synthetic/llm_social/skeletons.py data/synthetic/llm_devwork/raw/<WRITER>_part*.jsonl` (0 rows that share a skeleton)
   - `python3 data/synthetic/llm_social/near_duplicates.py 0.5 data/synthetic/llm_devwork/raw/<WRITER>_part*.jsonl` (0 rows beyond the first of a cluster)
   Count your person values with a one-off command: none may appear in more than 2 rows.
4. When done, report the row count, the validate summary line and the registers you used.

## Checker steps (checkers only)

You check one writer's rows. You are a different model from the writer.

1. Read `benchmark/GUIDELINE.md` and this briefing in full.
2. Read every row of the writer's `raw/<WRITER>_part*.jsonl` files.
3. For each row, check every mark and the text around it:
   - Each mark has the right type, and the value is exactly the PII and nothing more.
   - The row has no missed PII. A private person's name left unmarked is the worst
     error in this set: mark it.
   - The row has no marked non-PII: bots, assistant personas, products and tools named
     like people, public figures, teams, repositories.
   - A row is unfixable when it reads as a template or is nonsense.
4. Write the corrected rows, keeping their ids, to
   `data/synthetic/llm_devwork/reviewed/<WRITER>.reviewed.jsonl`. Drop unfixable rows.
5. Write `data/synthetic/llm_devwork/reviewed/<WRITER>.changes.md`: one line per changed
   or dropped row (the id and what changed), then counts: rows read, rows changed, rows
   dropped, and marks added, removed and retyped.
6. Run `validate.py` on your reviewed file until it reports 0 bad rows.
7. Report the counts and the validate summary line.
