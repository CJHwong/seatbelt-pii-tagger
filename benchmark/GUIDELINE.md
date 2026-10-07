# Annotation guideline: the Traditional Chinese PII benchmark

This benchmark tests a PII detector on realistic Taiwanese text. It is a test set only.
Nobody trains on it or tunes against it.

## Row format

One JSON object per line, in a `.jsonl` file:

```json
{"id": "A-001", "register": "LINE chat", "text": "欸[[person:陳怡君]]你的新手機是[[phone:0912-345-678]]嗎"}
```

- Mark every span inline as `[[type:value]]`. The value is the exact text as it appears.
- Do not nest marks. Do not put `[[`, `]]` or `:` inside a marked value's type name.
- A value may contain spaces, dashes, full-width characters and line breaks, exactly as written.
- Rows with no PII have no marks. They are as important as rows with PII.
- Use `\n` for line breaks inside `text`.

## What to mark

Mark every mention, including the second and later mentions of the same thing.

| type | mark | do not mark |
|---|---|---|
| `person` | A private person's full name, given name alone, nickname or English name, when it points at that person: 陳怡君, 怡君, 小君, Kevin Lin | Surname plus a title or kinship word: 陳先生, 林小姐, 王媽媽, 張醫師. Public figures, politicians, celebrities, athletes, historical people, fictional characters, brand names |
| `email` | A person's email address | An organisation's shared address: service@, info@, press@ |
| `phone` | A person's mobile or home number, in any format, with or without +886 | Public hotlines, company switchboards, 110, 119, 1999, 0800 numbers |
| `address` | A street-level address of a person: road, lane, alley, number, floor. Include the city and district when they are written next to it | A city or district alone. A landmark, shop, school, hospital or government office and its address |
| `date` | A person's birth date only, in any format, including 民國 years | Every other date: appointments, deadlines, start dates, holidays |
| `url` | A URL that points at one private person: a profile, a personal blog, a personal site | Company, government, news and shop websites |
| `account_number` | A bank account number or a credit card number | An invoice, order, tracking or reference number |
| `secret` | A password, PIN, one-time code, API key, access token, card security code (CVV) | Placeholder text such as `your_password_here` |
| `national_id` | A 身分證字號 (letter plus 9 digits) | |
| `passport` | A passport number | |
| `license_plate` | A private vehicle's plate | Bus route numbers, train numbers, flight numbers |
| `username` | A person's handle or account ID: LINE ID, PTT ID, IG handle, Dcard ID, game ID | |
| `nhi_card` | A 健保卡號 (12 digits) | |
| `company_id` | A 統一編號 (8 digits) of a business | |
| `driver_license` | A driver's licence number | |
| `medical_license` | A physician's or pharmacist's licence number, with its wrapper such as 醫字第…號 when written | |
| `military_id` | A military ID or service number, with its wrapper such as 軍字第…號 when written | |
| `household_no` | A 戶號 (letter plus 7 digits) | |
| `ignore` | A real identifier that fits no type above: a patient chart number, an employee ID, a student ID, a membership number. Also any span where two readings are equally reasonable: two types (a military driving licence), or PII against not PII (a small shop's advertised mobile). A prediction on an `ignore` span counts neither way. | |

## Values

- Every value is fictional. Never use a real private person's data.
- Invent values yourself. Do not reuse the examples in this file.
- Real public figures and real businesses are allowed only in text that does not mark them.
- Make formats realistic and varied. A national ID is a capital letter, then 1 or 2, then 8
  digits. A mobile is 09 plus 8 digits. A 健保卡號 starts with 0000.

## What makes a good row

- It reads like real Taiwanese writing from its register, not like a form filled from a template.
- Vary length: some rows are one line, some are ten.
- Include realistic mess: typos, 注音文, English mixed in, emoji, full-width digits,
  spaces inside numbers, OCR errors in scanned forms.
- Negatives should be hard: text full of names of public figures, shops, places, hotlines,
  order numbers and dates that are not birth dates.
