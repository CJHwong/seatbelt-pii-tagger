---
license: apache-2.0
base_model: microsoft/Multilingual-MiniLM-L12-H384
language:
- en
- zh
pipeline_tag: token-classification
datasets:
- nvidia/Nemotron-PII
- ai4privacy/pii-masking-openpii-1.5m
- Wismut/nym-pii-multilingual-data
- gretelai/synthetic_pii_finance_multilingual
- gretelai/gretel-pii-masking-en-v1
- CJHwong/seatbelt-pii-tagger-data
---

# seatbelt-pii-tagger 2026.10: English, Simplified Chinese and Traditional Chinese

A 35M-parameter token tagger that finds personal data and secrets in text. PII means personally identifiable information. It reads English, Simplified Chinese (zh-Hans) and Traditional Chinese (zh-Hant). It targets developer text: prompts, agent replies, tool calls and tool output from coding agents. The shipped files are an int8 ONNX (Open Neural Network Exchange) graph of 35.6 MB that runs on the CPU, and an fp16 Core ML package of 69 MB that runs on the Apple Neural Engine.

The [technical report](https://github.com/CJHwong/seatbelt-pii-tagger/blob/main/docs/technical_report.md) holds the method and every result. The [code](https://github.com/CJHwong/seatbelt-pii-tagger) rebuilds the model with `reproduce.sh`.

## Intended use

A per-prompt hook that flags personal data and secrets before text leaves the machine. This is a personal learning project, not a product.

Out of scope:

- Languages other than English and Chinese.
- Identifiers outside the 18 types below, such as employee IDs, IP addresses or medical record numbers.
- Company names, occupations, ages and other quasi-identifiers.
- Any use that needs complete recall, such as legal anonymisation.

## Labels

The model tags 18 types with BIOES (Begin, Inside, End, Single, Outside), so the head has 73 classes. A span is personal data only when it points at one private person. Public figures, shared mailboxes, public hotlines, landmark addresses and dates other than birth dates are not personal data.

| type | marks |
|---|---|
| `person` | a private person's name, given name, nickname or English name |
| `email` | a person's email address |
| `phone` | a person's mobile or home number |
| `address` | a person's street-level address |
| `date` | a birth date only |
| `url` | a URL that points at one private person |
| `account_number` | a bank account or credit card number |
| `secret` | a password, PIN, one-time code, API key, access token or card security code |
| `national_id` | a national ID, for example a 身分證字號 or SSN |
| `passport` | a passport number |
| `license_plate` | a private vehicle's plate |
| `username` | a person's handle or account ID |
| `nhi_card` | a Taiwan National Health Insurance card number (健保卡號) |
| `company_id` | a Taiwan business number (統一編號) |
| `driver_license` | a driver's licence number |
| `medical_license` | a physician's or pharmacist's licence number |
| `military_id` | a military ID or service number |
| `household_no` | a Taiwan household number (戶號) |

## How to run it

The model runs in the `tagger` mode of the [agent-seatbelt](https://github.com/CJHwong/agent-seatbelt) PII hook. The hook splits text into windows of at most 400 characters, runs each window alone, and decodes the token scores into character spans. A rule chain then drops public figures, places, documentation URLs, shared mailboxes and code identifiers.

In an agent-seatbelt checkout, start the server. It downloads the files from this repository on first use:

```bash
uv run hooks-pii/server/server.py --mode tagger --port 9123
```

The server picks the network's runtime by itself:

- On Apple Silicon, it runs `model.mlpackage` on the Neural Engine. The package is fp16 and pads each window to 64, 128, 256 or 512 tokens. A window takes 0.4 ms at the median, against 1.4 ms for the int8 graph on the CPU (Apple M5 Pro, 400 benchmark windows, lower is better). Its spans match the PyTorch fp32 network's on all but 93 of 33,479 comparison rows.
- Elsewhere, or when Core ML fails to load, it runs `model.int8.onnx` on the CPU. Set `PII_TAGGER_FP32=1` to run `model.onnx` instead.

To run your own export, set `PII_TAGGER_DIR` to a folder that holds the same files.

`model.safetensors` and `config.json` hold the same network as PyTorch weights, for `transformers`: `BertForTokenClassification.from_pretrained` loads them, with the 73 label names in the config. The hook does not use them.

## Results

"Ours" marks a set built for this model. "Public" marks a published dataset.

| set | made by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|
| tw_bench | public | typed F1 (higher is better) | **0.847** | **0.847** | 0.684 | 0.336 | 0.376 | 0.593 |
| zh_bench | public | any overlap F1 (higher is better) | **0.998** | **0.998** | 0.937 | 0.717 | 0.717 | 0.596 |
| CredData | public | clean lines flagged, of 1,000 (lower is better) | 160 | 155 | 195 | **51** | 189 | 759 |
| nemotron_test | public | typed F1 (higher is better) | **0.979** | 0.976 | 0.794 | 0.675 | 0.764 | 0.890 |
| kiji_en | public | typed F1 (higher is better) | **0.861** | 0.858 | 0.823 | 0.619 | 0.643 | 0.805 |
| Secrets suite, 4 sets | 3 ours, 1 public (CredData) | mean typed F1 (higher is better) | 0.941 | **0.942** | 0.769 | 0.000 | 0.762 |  |
| Our core suite, 8 sets | ours | mean typed F1 (higher is better) | **0.942** | 0.941 | 0.759 | 0.399 | 0.420 | 0.519 |
| transcript_dev | ours, private data | F1 (higher is better) | 0.701 | **0.752** | 0.325 | 0.282 | 0.169 | 0.119 |
| transcript_test | ours, private data | F1 (higher is better) | 0.739 | **0.750** | 0.414 | 0.321 | 0.187 | 0.167 |

What each column is:

- **raw**: this model alone.
- **+ chain**: this model plus the rules agent-seatbelt runs after it.
- **OpenAI Privacy Filter**: openai/privacy-filter alone.
- **Redact model**: desert-ant-labs/redact by [Desert Ant Labs](https://desertant.com) alone, with no rules.
- **Redact hook**: the same Redact model as agent-seatbelt's `redact` mode runs it. That mode adds agent-seatbelt's own regex rules, including 271 secret-token patterns, to the model's spans. These are not Desert Ant Labs' rules. The secret scores in this column come from them: the Redact model alone scores 0.000 on the Secrets suite.
- **NVIDIA GLiNER-PII**: nvidia/gliner-PII at its card's threshold of 0.3. It is English-only, so its Chinese scores show a language limit, not quality. It ran only CredData of the secret sets, so its Secrets suite cell is empty.

What each metric is:

- **F1** combines precision (the share of flagged spans that are right) and recall (the share of real values found): 2 × precision × recall / (precision + recall). 1.0 is perfect.
- **typed F1**: a flagged span counts only if it overlaps a real value of the same type. The comparison uses the types every system shares. The 8 ID types (national ID, passport, driver licence and the 5 Taiwan-only IDs) count as one type, and dates, usernames and licence plates are not scored.
- **any overlap F1**: a flagged span counts if it overlaps a real value, whatever its type.
- **mean typed F1**: the average of typed F1 over the sets of a suite, each set with the same weight.
- **clean lines flagged**: of the lines that hold no secret, the number a system flags.
- **F1 on the transcript sets**: judges decide. Precision is the share of flagged spans a judge called personal data. Recall is the share of real values a flagged span overlaps, whatever its type.
- **Bold** marks the best value in a row.

The transcript sets come from the author's own coding-agent sessions. Their scores are published, and their text is not. Section 8 of the technical report gives every set and one more baseline, the agent-seatbelt rules engine. nemotron_test shares a generator with a training source, so its score is likely optimistic.

## Limits

- After the chain, person recall is low on sets whose gold marks public figures. For example, multinerd_en recall drops from 0.941 raw to 0.523 with the chain (higher is better). This follows the guideline.
- The raw model flags public names in news text: 66.1 flags per 10k characters on wikinews_zhtw (lower is better). The chain brings it to 39.4.
- Casual text is weak. Weibo person recall is 0.478 and WNUT-17 person recall is 0.508 (higher is better).
- Transcript recall is about 0.7: 0.673 raw and 0.714 with the chain on transcript_dev (higher is better). The transcript test set holds only 26 gold entities, so one entity moves recall by about 0.038.
- Training is not deterministic on Apple MPS (Metal Performance Shaders). Copies of one recipe differ, and this model is one copy.
- No public data teaches the Taiwan-only types. Only templates teach them.

## Training data

| source | kind | train rows |
|---|---|---|
| ai4privacy/pii-masking-openpii-1.5m | public dataset (synthetic) | 183,140 |
| Wismut/nym-pii-multilingual-data | public dataset (synthetic) | 526,320 |
| nvidia/Nemotron-PII | public dataset (synthetic) | 98,989 |
| gretelai/synthetic_pii_finance_multilingual | public dataset (synthetic) | 25,719 |
| gretelai/gretel-pii-masking-en-v1 | public dataset (synthetic) | 49,506 |
| Traditional Chinese templates | in-house synthetic | 182,509 |
| secrets from regex rules | in-house synthetic | 58,757 |
| social, developer-work and agent-session rows | in-house synthetic | 5,297 |
| coding-agent transcript windows | in-house, private | 7,740 |

The transcript windows come from the author's own coding-agent sessions, on personal and professional projects. LLM judges labelled them, and every judged personal value was replaced by a fake before training. The windows still hold real text, so they are private and not released.

Two comparison systems also tag personal data: Redact by Desert Ant Labs and NVIDIA GLiNER-PII. No training label comes from either one. Transcript windows that showed a Redact or GLiNER candidate to a judge were judged again without it.

The in-house synthetic sets are in [CJHwong/seatbelt-pii-tagger-data](https://huggingface.co/datasets/CJHwong/seatbelt-pii-tagger-data). In 138 rows of the developer-work and agent-session sets, names and project details copied from the author's sessions are swapped for fakes, so those rows differ from the training copies (see the technical report, section 10).

## Licence and attributions

Apache 2.0. The NOTICE file carries the same attributions as this table.

| work | licence and holder | use |
|---|---|---|
| microsoft/Multilingual-MiniLM-L12-H384 | MIT License, Copyright (c) Microsoft Corporation (github.com/microsoft/unilm LICENSE) | base model: layers 0 to 5 kept, vocabulary pruned from 250,037 to 64,000 XLM-R pieces |
| XLM-R SentencePiece tokenizer (fairseq) | MIT License, Copyright (c) Facebook, Inc. and its affiliates (github.com/facebookresearch/fairseq LICENSE) | tokenizer |
| nvidia/Nemotron-PII | CC BY 4.0, NVIDIA Corporation | training data, relabelled to 18 types |
| ai4privacy/pii-masking-openpii-1.5m | CC BY 4.0, "Ai4Privacy / Ai Suisse SA" | training data, relabelled and filtered to English and Chinese scripts |
| Wismut/nym-pii-multilingual-data | MIT, Wismut | training data, relabelled and filtered |
| gretelai/synthetic_pii_finance_multilingual | Apache 2.0, Gretel | training data |
| gretelai/gretel-pii-masking-en-v1 | Apache 2.0, Gretel | training data |
| gitleaks (MIT, Zachary Rice), betterleaks (MIT), microsoft/security-utilities (MIT), Nosey Parker (Apache 2.0, Praetorian Security) | as listed | regexes that generate the secret training strings |
| Faker | MIT | synthetic values |
| Wikipedia (wikimedia/wikipedia 20231101.zh) | CC BY-SA | Traditional Chinese paragraphs that rank the vocabulary; no Wikipedia text is in the model |
| Wikidata | CC0 | the public persons and places lists |

Evaluation sets: CredData (Yun et al., 2021, https://github.com/Samsung/CredData). Weibo NER, the revised annotation (`weiboNER_2nd_conll` in https://github.com/hltcoe/golden-horse): Peng and Dredze, "Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings", EMNLP 2015; Peng and Dredze, "Improving Named Entity Recognition for Chinese Social Media with Word Segmentation Representation Learning", ACL 2016; He and Sun, "F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media", arXiv:1611.04234, 2016.
