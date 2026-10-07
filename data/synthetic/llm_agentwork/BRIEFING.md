# Briefing: LLM-written coding-agent sessions for a PII tagger

## Why these rows exist

A small token tagger (6 layers, 35M parameters) finds personal data (PII) in English,
Simplified Chinese and Traditional Chinese. It runs on what a coding agent reads and
writes: the user's prompts, the agent's replies, its tool calls and the tool output. There
it makes three mistakes, all about short words:

1. **It calls a product a person.** A capitalized word in engineering prose reads like a
   name to it: a SaaS product, a cloud service, a library, an AI model, a company, a
   shipping carrier, a status word in a table (Green, Blocked), a verb in a heading (Lint,
   Deploy). None of these is PII.
2. **It misses a coworker named by one word.** "ping Mateo", "waiting on yuki", "Arjun's
   branch". A lone given name, often lowercase, among tool names.
3. **It misses a handle inside a command or a payload.** A reviewer in `gh pr edit
   --add-reviewer`, a Slack user in a search query, an assignee in a Jira JSON body, a
   login in an API response, a mention in a commit trailer.

These rows put real people and look-alike words side by side, so the tagger learns the
difference from context.

## Hard rules

1. Write every row yourself. Do not read any file in this repository except this
   briefing, `benchmark/GUIDELINE.md`, the three check scripts named below and your own
   output files. Do not open `benchmark/benchmark.jsonl`, `benchmark/raw/`, any `blind`
   folder, any `runs/` folder, anything under `data/` except your own output folder, any
   test or evaluation data, or any dataset file. Do not read anything under `~/.claude`
   or `~/.codex`, and do not read any real transcript or log on this machine.
2. Do not fetch anything from the internet.
3. Write each row by hand. Do not write code, scripts or loops that generate rows, and do
   not fill slots in a sentence pattern. A row that differs from another only in its
   marked values or digits is a template, and `skeletons.py` rejects it.
4. Every private value is fictional. Invent names, handles, emails and IDs. Do not use a
   real private person's data. Do not reuse the examples in the guideline or here.
   Do not copy any name, handle, company, organization or repository from your own
   instructions or session context. Those are real, and they would enter the training set.
5. Real public companies, products, services, open-source projects and public figures
   may appear, always unmarked.

## Row format

One JSON object per line, UTF-8, in a `.jsonl` file:

```json
{"id": "A1-0001", "register": "agent tool call", "script": "latin", "text": "..."}
```

- `id`: your writer code, a dash, a 4-digit number. Unique.
- `register`: a short English name of the genre.
- `script`: `latin` (English), `zh-Hant` or `zh-Hans`.
- `text`: the row, with every PII span marked inline as `[[type:value]]`. The value is
  the exact text as it appears. Use `\n` for line breaks. No nested marks. No `[[` or
  `]]` anywhere else; if code needs `[[`, write it differently.

## Registers

Each writer has a main area (given in your task). All of them are coding-agent sessions:

- **user prompts to an agent**: terse, casual, sometimes lowercase or with typos; they name
  tools, tickets and teammates ("ask wen if the Datadog alert is hers")
- **agent replies**: summaries, plans and reports in Markdown, dense with product and
  service names, with the teammates the work involves
- **tool calls**: shell commands (`gh`, `git`, `kubectl`, `aws`, `curl`, `jq`), JSON
  arguments of tools such as a Slack search, a Jira query, a calendar lookup, a GitHub API
  call, written the way an agent emits them
- **tool output**: CLI tables, API JSON (users, reviewers, assignees, authors), Slack
  message lists, Jira issue bodies, CI logs, `git log` and `git blame` lines
- **Markdown status reports**: tables with owner, reviewer or assignee columns beside
  status words, product columns and ticket keys

Length: 2 lines to about 30 lines.

## What to mark

`benchmark/GUIDELINE.md` is the labelling policy. Read it in full. In this register:

- **person**: a private person's name: a given name alone, lowercase or not, a nickname, a
  full name, a surname alone. Every mention, including possessives (`Arjun` in "Arjun's")
  and names in table cells.
- **username**: a person's handle or login: in a command argument, a JSON field
  (`"login"`, `"user"`, `"assignee"`, `"reviewer"`, `"author"`), an @mention (mark it
  without the @), a Slack member ID that belongs to a person, a `Co-authored-by` trailer.
- **email**: a person's address.
- **secret**: a live-looking credential, rarely (at most 1 row in 20).
- **ignore** (`[[ignore:value]]`): a login inside a home-directory path such as
  `/Users/<login>/`.

## What not to mark, and put several in every row

- products, services and tools: issue trackers, CI systems, clouds and their services,
  observability and LLM tooling, databases, editors, AI models and assistants
- companies, banks, carriers, vendors, customers named as companies, teams, channels
- words that look like names but are not: status values (Green, Amber, Red, Blocked),
  heading verbs (Deploy, Rollback, Publish), table headers, branch and job names,
  feature flags, product names that are also given names (Jenkins, Ruby, Julia, Ada)
- bots and service accounts, shared mailboxes, default accounts, an AI agent's persona
- public figures (a famous founder, a politician in a news snippet the user pasted)
- repositories, packages, functions, variables, paths, hostnames, ticket keys, hashes,
  versions, placeholders such as `<user>` or `${REVIEWER}`

## What makes a good row

- A real person sits next to look-alikes in the same row: `--add-reviewer <handle>`
  beside `--label Deploy`; "Mateo saw it in Grafana"; an assignee login next to a
  `"status": "Blocked"` field.
- Most names are one word. Many handles are lowercase, with digits, dots or dashes.
- Vary names across cultures (Taiwanese, Chinese, Indian, Vietnamese, Filipino, Korean,
  Japanese, European, American, African, Latin American), the products, the topics and
  the lengths. No person, handle or email may appear in more than 2 rows.
- About 80% of rows contain at least one mark. About 20% contain none: dense with
  products, companies, status words and bots, and no person.
- Never leave a private person's name or handle unmarked. That teaches the opposite.
- About 20% of rows are Traditional Chinese (`zh-Hant`): the mixed Chinese and English a
  Taiwanese team writes to its agent ("幫我問一下 [[person:阿哲]] Sentry 那個 alert").

## Steps

1. Read `benchmark/GUIDELINE.md`.
2. Write your rows in parts of about 50 rows each:
   `data/synthetic/llm_agentwork/raw/<WRITER>_partN.jsonl`.
3. After each part, run these from the repository root and fix every reported row:
   - `python3 data/synthetic/llm_social/validate.py data/synthetic/llm_agentwork/raw/<WRITER>_part*.jsonl` (0 bad rows)
   - `python3 data/synthetic/llm_social/skeletons.py data/synthetic/llm_agentwork/raw/<WRITER>_part*.jsonl` (0 rows that share a skeleton)
   - `python3 data/synthetic/llm_social/near_duplicates.py 0.5 data/synthetic/llm_agentwork/raw/<WRITER>_part*.jsonl` (0 rows beyond the first of a cluster)
   Count your person, username and email values with a one-off command: none may appear
   in more than 2 rows.
4. When done, report the row count, the validate summary line and the registers you used.

## Checker steps (checkers only)

You check one writer's rows. You are a different model from the writer.

1. Read `benchmark/GUIDELINE.md` and this briefing in full.
2. Read every row of the writer's `raw/<WRITER>_part*.jsonl` files.
3. For each row, check every mark and the text around it:
   - Each mark has the right type, and the value is exactly the PII and nothing more.
   - The row has no missed PII. A private person's name or handle left unmarked is the
     worst error in this set: mark it.
   - The row has no marked look-alike: a product, a company, a status word, a bot, a
     persona, a public figure, a placeholder, a code identifier. Unmark it.
   - A row is unfixable when it reads as a template or is nonsense.
4. Write the corrected rows, keeping their ids, to
   `data/synthetic/llm_agentwork/reviewed/<WRITER>.reviewed.jsonl`. Drop unfixable rows.
5. Write `data/synthetic/llm_agentwork/reviewed/<WRITER>.changes.md`: one line per changed
   or dropped row (the id and what changed), then counts: rows read, rows changed, rows
   dropped, and marks added, removed and retyped.
6. Run `validate.py` on your reviewed file until it reports 0 bad rows.
7. Report the counts and the validate summary line.
