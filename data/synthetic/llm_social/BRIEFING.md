# Briefing: LLM-written social-media training rows for a PII tagger

## Why these rows exist

A small token tagger (6 layers, 35M parameters) finds personal data (PII) in English,
Simplified Chinese and Traditional Chinese. Its training text is mostly templates and
form-like synthetic rows. It is weakest on real casual text: Weibo posts, tweets, chats,
where names are nicknames, given names, @handles, lowercase, and run into the text with
little punctuation. These rows give it varied, realistic casual text with correct labels.

## Hard rules

1. Write every row yourself. Do not read any file in this repository except this
   briefing, `benchmark/GUIDELINE.md`, `validate.py` and your own output files. Do not
   open `benchmark/benchmark.jsonl`, `benchmark/raw/`, any `blind` folder, any test or
   evaluation data, or any dataset file. Rows that copy test text spoil the experiment.
2. Do not fetch anything from the internet.
3. Write each row by hand. Do not write code, scripts or loops that generate rows, and do
   not fill slots in a sentence pattern. A row that differs from another only in its marked
   values, hashtags or digits is a template, and `skeletons.py` rejects it.
4. Every value is fictional. Invent names, numbers, handles and addresses. Do not use a
   real private person's data. Do not reuse the examples in the guideline.
5. Real public figures, brands, shops, places and hotlines may appear only unmarked, as
   the guideline says. Use them sparingly (see "Negatives" below).

## Row format

One JSON object per line, UTF-8, in a `.jsonl` file:

```json
{"id": "W1-0001", "register": "LINE group chat", "script": "zh-Hant", "text": "..."}
```

- `id`: your writer code, a dash, a 4-digit number. Unique.
- `register`: a short English name of the genre, such as `PTT post`, `Weibo comment`.
- `script`: `zh-Hant`, `zh-Hans` or `latin` (for English).
- `text`: the row, with every PII span marked inline as `[[type:value]]`. The value is the
  exact text as it appears. Use `\n` for line breaks. No nested marks. No `[[` or `]]`
  anywhere else.

## What to mark

`benchmark/GUIDELINE.md` is the labelling policy. Read it in full. It is written for
Taiwanese text. The same types apply to the other two scripts, with these local forms:

- **Simplified Chinese (mainland):**
  - `national_id`: an 18-character 身份证号 (6-digit region, 8-digit birth date,
    3 digits, a check digit or X).
  - `phone`: an 11-digit mobile starting with 1 (13x to 19x), with or without +86 or
    spaces.
  - `username`: a 微博 @handle or 昵称 when it names an account, a 微信号, a QQ号, a
    小红书号, a 抖音号. Mark the handle without the @ when it is written as `@name`. For
    example `@[[username:momo不吃香菜]]`.
  - `license_plate`: forms like 粤B·D12345 or 沪A 8K2L9.
  - `address`: the street level and below, with the province, city and district when
    written next to it: 路, 号, 小区, 栋, 单元, 室.
  - Do not use the Taiwan-only types: `nhi_card`, `household_no`, `medical_license`,
    `military_id`, `company_id`. Mark a 统一社会信用代码 or other identifier with no type
    as `ignore`.
- **English:**
  - `national_id`: a US SSN, a UK National Insurance number, or similar.
  - `phone`: a person's number, in any local or international format.
  - `username`: @handles, Discord tags, Reddit u/ names (mark the name without the `u/`),
    gamer tags, Venmo or Cash App handles.
  - Do not use the Taiwan-only types listed above.
- **All scripts:**
  - `person` includes a given name alone, a nickname and a lowercase name ("jake",
    "mei", "小雨"), when it points at a private person.
  - A name written only as `@handle` is a `username`, not a `person`.
  - A person's name in a display name next to a handle is a `person`.
  - Mark every mention, including repeats.

## What makes a good row

- It reads like a real person typing in that register. Short and long rows. Typos, slang,
  emoji, no punctuation, run-on sentences, mixed English, 注音文, full-width digits,
  spaces inside numbers. Casual names: given names, nicknames, 小名, lowercase English.
- Several people per row is good. PII sits in the middle of sentences, not only in
  "name: X" lines. Avoid form-like layouts; the training set has enough forms.
- Vary everything across rows: names (common and rare surnames, two- and three-character
  given names, English names), number formats, topics, lengths, which types appear.
  Do not repeat a name or value across rows.
- About 85% of rows contain at least one PII span. About 15% contain none.

## Negatives

Earlier experiments showed a trap. Rows where a person's name in prose goes unmarked teach
the tagger that names in prose are not PII, and its recall on private names fell. So:

- Do not write rows that are lists or discussions of celebrities, politicians or
  historical people.
- A public figure, brand, shop or place may appear in a row that also mentions private
  people, as normal chatter does ("昨天看五月天演唱會 跟[[person:阿哲]]...").
- Rows with no PII should be about things the guideline excludes: order and tracking
  numbers, hotlines and 0800 numbers, shop addresses, appointment dates, company
  websites, prices, product codes. Use `[[ignore:...]]` for real identifiers with no type
  (a member number, a student ID, an employee ID).

## Steps

1. Read `benchmark/GUIDELINE.md`.
2. Write your rows in parts of about 50 rows each:
   `data/synthetic/llm_social/raw/<WRITER>_partN.jsonl`.
3. After each part, run
   `python3 data/synthetic/llm_social/validate.py data/synthetic/llm_social/raw/<WRITER>_part*.jsonl`
   from the repository root. Fix every bad row before you go on. Then run
   `python3 data/synthetic/llm_social/skeletons.py data/synthetic/llm_social/raw/<WRITER>_part*.jsonl`.
   It must report 0 rows that share a skeleton. Then run
   `python3 data/synthetic/llm_social/near_duplicates.py 0.5 data/synthetic/llm_social/raw/<WRITER>_part*.jsonl`.
   It must report 0 rows beyond the first of a cluster. A row that retells another row's
   situation with new details is a near-duplicate, and `prepare.py` drops it.
   Mark every mention of a person, including repeats and chat speaker labels, not only the
   first. Count your person values with a one-off command: none may appear in more than 2
   rows.
4. When done, report the row count, the validate summary line and the registers you used.

## Checker steps (checkers only)

You check one writer's rows. You are a different model from the writer.

1. Read `benchmark/GUIDELINE.md` and this briefing in full.
2. Read every row of the writer's `raw/<WRITER>_part*.jsonl` files.
3. For each row, check every mark and the text around it:
   - Each mark has the right type, and the value is exactly the PII and nothing more.
   - The row has no missed PII.
   - The row has no marked non-PII: public figures, titles such as 陳先生, shared
     mailboxes, hotlines, appointment dates, order numbers.
   - A row is unfixable when it copies a known real text, reads as a template, or is
     nonsense.
4. Write the corrected rows, keeping their ids, to
   `data/synthetic/llm_social/reviewed/<WRITER>.reviewed.jsonl`. Drop unfixable rows.
5. Write `data/synthetic/llm_social/reviewed/<WRITER>.changes.md`. It has one line per
   changed or dropped row (the id and what changed), then counts: rows read, rows
   changed, rows dropped, and marks added, removed and retyped.
6. Run `validate.py` on your reviewed file until it reports 0 bad rows.
7. Report the counts and the validate summary line.
