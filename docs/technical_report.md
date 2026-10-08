# seatbelt-pii-tagger 2026.10: a 35M-parameter PII tagger for English, Simplified Chinese and Traditional Chinese

Technical report. Paths are relative to the repository root. Every result comes from `experiments/compare/runs/tables.md`, which `experiments/compare/tables.py` writes from the saved scores. In every table, "ours" marks a set built for this model, and "public" marks a published dataset.

## 1. Summary

seatbelt-pii-tagger is a token classifier that finds personal data and secrets in text. PII means personally identifiable information. It reads English, Simplified Chinese (zh-Hans) and Traditional Chinese (zh-Hant). It targets developer text: prompts, agent replies, tool calls and tool output from coding agents. It also covers chat, forms and Taiwanese text.

- Architecture: a 6-layer BERT-style encoder, hidden size 384, a 64,000-piece vocabulary, 35,449,033 parameters.
- Output: 18 entity types, tagged with BIOES (Begin, Inside, End, Single, Outside), so 73 classes.
- Shipped format: ONNX (Open Neural Network Exchange), int8 weights, 35,607,393 bytes.
- Post-processing: a separate rule chain runs after the model in the [agent-seatbelt](https://github.com/CJHwong/agent-seatbelt) hook. This report scores the raw model and the model plus the chain separately.

Headline results:

| set | made by | metric | raw | + chain | OpenAI Privacy Filter |
|---|---|---|---|---|---|
| Our core suite, 8 sets | ours | mean typed F1 (higher is better) | 0.942 | 0.941 | 0.759 |
| tw_bench (Taiwanese) | public | typed F1 (higher is better) | 0.847 | 0.847 | 0.684 |
| CredData (secrets in code) | public | clean lines flagged, of 1,000 (lower is better) | 160 | 155 | 195 |
| transcript_dev (coding-agent windows) | ours, private data | F1 (higher is better) | 0.701 | 0.752 | 0.325 |

## 2. Intended use and out-of-scope use

**Intended use.** A per-prompt hook that flags personal data and secrets before text leaves the machine. It suits English, zh-Hans and zh-Hant text. This is a personal learning project, not a product.

**What a label means.** The model follows `benchmark/GUIDELINE.md`. A span is personal data only when it points at one private person. The guideline makes these choices:

| choice | effect on the output |
|---|---|
| Public figures are not personal data. Politicians, celebrities, athletes, historical people, fictional characters and brand names stay unmarked. | `person` covers private people only. |
| A surname plus a title or kinship word (陳先生, 王媽媽) is not a name. | Not marked. |
| An organisation's shared mailbox (service@, info@) is not personal. | Not `email`. |
| Public hotlines, switchboards and 0800 numbers are not personal. | Not `phone`. |
| A city or district alone, or the address of a landmark, shop or office, is not personal. | `address` is street-level and private only. |
| `date` means a birth date only. | Appointments, deadlines and other dates stay unmarked. |
| A company, news or government URL is not personal. | `url` covers profiles and personal sites only. |
| Invoice, order and tracking numbers are not accounts. | Not `account_number`. |
| Placeholder text such as `your_password_here` is not a secret. | Not `secret`. |

The raw network does not learn public figures and places well. The chain does that work with name lists (section 4).

**Out of scope.**

- Languages other than English and Chinese. The training build drops other scripts.
- Identifiers outside the 18 types: employee IDs, medical record numbers, tax IDs, IP and MAC addresses, device IDs. Sensitive attributes such as religion or health are also out.
- Company names, occupations, ages and other quasi-identifiers.
- Any use that needs a guarantee of complete recall, such as legal anonymisation. Section 9 gives the measured misses.
- Public-figure filtering without the chain. The raw model flags many public names (section 8).

## 3. Labels

The head has 1 + 4 x 18 = 73 classes: `O` plus `B-`, `I-`, `E-` and `S-` for each type. A Viterbi decoder with hard BIOES constraints turns token scores into character spans `{start, end, label, score}`.

| # | type | what it marks |
|---|---|---|
| 1 | `person` | a private person's full name, given name, nickname or English name |
| 2 | `email` | a person's email address |
| 3 | `phone` | a person's mobile or home number |
| 4 | `address` | a person's street-level address |
| 5 | `date` | a birth date only |
| 6 | `url` | a URL that points at one private person |
| 7 | `account_number` | a bank account or credit card number |
| 8 | `secret` | a password, PIN, one-time code, API key, access token or CVV (card security code) |
| 9 | `national_id` | a national ID, for example a 身分證字號 or SSN |
| 10 | `passport` | a passport number |
| 11 | `license_plate` | a private vehicle's plate |
| 12 | `username` | a person's handle or account ID |
| 13 | `nhi_card` | a Taiwan National Health Insurance card number (健保卡號) |
| 14 | `company_id` | a Taiwan business number (統一編號) |
| 15 | `driver_license` | a driver's licence number |
| 16 | `medical_license` | a physician's or pharmacist's licence number |
| 17 | `military_id` | a military ID or service number |
| 18 | `household_no` | a Taiwan household number (戶號) |

The release file `labels.json` holds the 73 labels in this order. The six Taiwan-only types (13 to 18) have no public training data. Only the template generator teaches them.

## 4. Model

| property | value |
|---|---|
| base model | `microsoft/Multilingual-MiniLM-L12-H384`, first 6 of 12 layers |
| layers, hidden, heads, intermediate | 6, 384, 12, 1,536 |
| vocabulary | 64,000 pieces, pruned from 250,037 |
| parameters | 35,449,033 |
| training length | 256 tokens (`TRAIN_LENGTH` in `experiments/vocab_prune/train_eval.py`) |
| inference window | 400 characters, at most 512 tokens (release file `tagger.json`) |
| chain thresholds | `url` and `username` spans under 0.5 are dropped (`tagger.json`) |

**Vocabulary pruning, and why.** The full XLM-R (cross-lingual RoBERTa) vocabulary costs about 96M embedding parameters at hidden size 384. That is too large for a per-prompt hook. `experiments/vocab_prune/build_pruned.py` ranks pieces by frequency over the training corpus, keeps 64,000, and keeps the embedding rows of those pieces. Every single-character piece is kept, so no text becomes unknown. The ranking corpus also holds 160,904 Traditional Chinese Wikipedia paragraphs, because template text holds too little real Traditional writing. The kept pieces cover 99.73% of the corpus tokens.

**Windowing.** The hook splits text into windows of at most 400 characters and runs each window alone. It then shifts the spans back to their places in the full text.

**Export.** `experiments/export/export_onnx.py` writes the hook files. onnxruntime's BERT optimizer fuses the graph. The int8 file uses dynamic int8 weights. `experiments/export/export_coreml.py` writes `model.mlpackage` for the Apple Neural Engine: fp16, with one function per padded length (64, 128, 256 and 512 tokens) over one shared copy of the weights. Its chained spans differ from the PyTorch fp32 spans on 93 of 33,479 comparison rows. The int8 graph differs on 1,963.

| file | size | note |
|---|---|---|
| `model.onnx` (fp32) | 141,827,559 bytes | through the hook, 0 of 33,479 rows differ from the scored spans (`experiments/export/parity.py`) |
| `model.int8.onnx` (int8, ships) | 35,607,393 bytes | on every set, the int8 scores stay inside the spread between training copies (section 7) |
| `persons.txt.gz`, `places.txt.gz` | 2,497,021 and 2,845,487 bytes | public-name lists the chain uses: 394,057 persons and 558,379 places |

**Latency** (lower is better). I timed the int8 graph through a local agent-seatbelt tagger server (`pii-server.py --mode tagger`) on an 18-core Apple M5 Pro. Each size got one warm-up request, then 30 timed requests over HTTP on the same machine. An 81-character text took a median of 2 ms (p90 2 ms). A 16,000-character text took a median of 92 ms (p90 96 ms). The texts were synthetic, with fake names, emails and IDs.

## 5. Training data

| source | kind | train rows | share of draws |
|---|---|---|---|
| ai4privacy/pii-masking-openpii-1.5m | public dataset (synthetic) | 183,140 | uniform remainder |
| Wismut/nym-pii-multilingual-data | public dataset (synthetic) | 526,320 | uniform remainder |
| nvidia/Nemotron-PII | public dataset (synthetic) | 98,989 | uniform remainder |
| gretelai/synthetic_pii_finance_multilingual | public dataset (synthetic) | 25,719 | uniform remainder |
| gretelai/gretel-pii-masking-en-v1 | public dataset (synthetic) | 49,506 | uniform remainder |
| synthetic/tw-templates | in-house script-generated | 182,509 | uniform remainder |
| synthetic/rule-secrets | in-house script-generated | 58,757 | uniform remainder; zh-Hans capped at 1,250 rows |
| synthetic/llm-social | in-house LLM-written | 2,588 | 0.10 |
| synthetic/llm-devwork | in-house LLM-written | 1,290 | 0.03 |
| synthetic/llm-agentwork | in-house LLM-written | 1,419 | 0.03 |
| synthetic/dev-transcripts | in-house, private: real windows with every judged value faked | 7,740 | 0.05 |

LLM means large language model. The rows are the `kept_train` counts in the build's `data/prepared/stats.json`. The table sums to 1,137,977 rows. The cap removes 2,924 zh-Hans rule-secret rows, so training draws from 1,135,053 rows.

**How the sampler works.** Each draw first picks a script bucket: Latin 0.5, zh-Hans 0.3, zh-Hant 0.2. Inside a bucket, each source with a share takes that share of the draws. All other rows in the bucket share the remaining draws as one pool, one row picked uniformly at a time. So each other source gets draws in proportion to its rows in that bucket. See `--source-share` in `experiments/first_run/train.py`.

**How the in-house sources are made.**

- **tw-templates.** `scripts/synthesize.py` fills `synthetic/templates.yaml` with values from `scripts/tw_values.py`. It writes 200,000 rows. A separate held-out file uses scenarios the train templates never use, and it never trains. The generator records each span as it places the value.
- **rule-secrets.** `scripts/rule_secrets.py` samples a string from each rule regex of the agent-seatbelt rules engine. It puts the value into code-shaped contexts (env, YAML, Kubernetes Secret, Python, JSON, curl and others) and into prose in all three scripts. A row stays only if the rules engine finds every provider value in it. About 20% of rows hold no secret but carry hard negatives such as UUIDs and commit hashes.
- **llm-social, llm-devwork, llm-agentwork.** LLMs wrote rows from a briefing and the guideline: casual chat (social), engineering prose (devwork) and coding-agent sessions (agentwork). The briefings are in `data/synthetic/llm_*/BRIEFING.md`. A checker on a different model reviewed every row. Each `prepare.py` drops duplicates, near-duplicates (Jaccard 0.5) and rows that share a 16-character Han or 32-character Latin stretch with any evaluation set.
- **dev-transcripts (private, not released).** Windows from the author's own coding-agent sessions, on personal and professional projects. The author's identifiers are replaced before a judge reads a window. LLM judges label the candidate spans and list misses. Every judged personal value is then swapped for a fake of its type. The windows still hold real text, so they are not released. A public rebuild trains without them (see the README).

**Provenance of the labels.** Two of the systems in section 8 also tag personal data: Redact by Desert Ant Labs and NVIDIA GLiNER-PII. Redact's licence bars using its outputs to train a competing model. No training label comes from either system. The transcript judges saw candidate spans from several systems. Every window whose candidates included a Redact or GLiNER span was judged again with only the other candidates: 1,373 dev windows and 144 test windows. The 2,007 Redact and GLiNER spans left over were judged for scoring only and never became gold. So the transcript recall of Redact and GLiNER in section 8 is measured against gold they did not help build.

**Leak controls.**

1. `experiments/compare/leak_check.py` compares every training row with every evaluation text. A row leaks when it has the same whitespace-free text, or covers 20% or more of the text's windows of 32 Latin or 16 Han characters. `scripts/build_training_set.py` runs it after every build. The build fails when a training row copies or half covers a transcript window.
2. Transcript training windows are dropped when they share a 32-character stretch with a dev or test window.
3. Evaluation rows that a training row copies are written to `experiments/compare/runs/leaked.json`, and `experiments/compare/score.py` leaves them out of every score.
4. nemotron_test shares a generator with a training source. Its scores are likely optimistic.

## 6. Training recipe

| setting | value |
|---|---|
| starting weights | the pruned build `build-split-tw` |
| steps | 10,000 |
| peak learning rate | 4.2e-4, linear warmup over 6% of steps, then linear decay to 0 |
| optimizer | AdamW, weight decay 0.01 |
| batch | 64, length-sorted pools |
| LP-FT probe | the first 1,000 steps train only the classifier, at learning rate 1e-3 |
| precision | bf16 |
| seed | 7 |
| hardware | Apple M5 Pro, 18 cores, 64 GiB, torch on MPS (Metal Performance Shaders) |
| time per run | 1,572 s (about 26 minutes), 6.36 steps per second |

LP-FT means linear probing, then fine-tuning. `reproduce.sh` holds the full command.

## 7. Evaluation protocol

**Suites.** Each suite row in section 8 is the mean of one metric over its sets. Clean rows flagged are a total, not a mean.

| suite | set | made by |
|---|---|---|
| Our core | our_benchmark, blind_en, blind_zh-Hans, blind_zh-Hant | ours |
| Our core | perturbed_fullwidth, perturbed_spaced, perturbed_ocr, perturbed_long | ours |
| Chinese | tw_bench, zh_bench, weibo, wikinews_zhtw | public |
| English and multilingual | gretel_en, tab_echr, wnut17, argilla, spy, wismut_test_latin | public |
| Secrets | creddata | public |
| Secrets | generated_held_out, seatbelt_secrets_trained, seatbelt_secrets_held_out, seatbelt_clean, seatbelt_positive, seatbelt_test_cases, seatbelt_holdout, seatbelt_final_holdout, seatbelt_challenge | ours |
| Published by the external systems | nemotron_test, kiji_en, gretel_finance_en, wikiann_en, wikiann_zh, multinerd_en, multinerd_zh, conll2003_en, multiconer_en, multiconer_zh | public |
| Transcripts | transcript_dev (3,000 windows, 168 gold entities), transcript_test (312 windows, 26 gold entities) | ours, private data |

The seatbelt_* sets come from the agent-seatbelt test suite. `experiments/compare/sets.py` loads every set.

**Scoring.** Every system reads the same 400-character windows. All labels map into the 8 OpenAI Privacy Filter categories. The ID types fold into `account_number`. `username` and `license_plate` have no shared category, so typed scores leave them out. Dates count on neither side. Some sets score person recall only, because their gold marks public figures. `experiments/compare/compare_lib.py` holds the rules.

**Transcripts.** The windows come from the author's own coding-agent sessions, on personal and professional projects. LLM judges labelled the union of every system's candidate spans. The windows hold real values, so they stay private, and only their aggregate scores are published.

**Run-to-run spread.** Training on MPS is not deterministic, so two runs of one recipe differ. Each recipe was trained 4 times. A change counted only when the mean of 4 copies moved by more than the spread inside the copies. The release is one of the 4 copies of its recipe: the one with the best secret F1 on the fixed dev split, 0.938 (the copies scored 0.934 to 0.938, higher is better). On transcript_dev, the 4 copies scored 0.693 to 0.718 raw F1.

**Which sets steered decisions.** A set that steered a choice gives an optimistic score.

| status | sets |
|---|---|
| Picked the shipped copy | the fixed dev split of the training build, and transcript_dev |
| Kept or dropped a training change | wikinews_zhtw, creddata, argilla clean rows |
| Shown in every comparison, so any of them could steer a choice | every other set outside the last row |
| Never used by any decision | transcript_test, the published group |

## 8. Results

Columns:

- **raw**: the network alone.
- **+ chain**: the network plus the rules agent-seatbelt runs after it.
- **OpenAI Privacy Filter** (openai/privacy-filter, 1.4B parameters).
- **Redact model** (desert-ant-labs/redact by Desert Ant Labs): the model alone, its rules blanked.
- **Redact hook**: the Redact model as agent-seatbelt's `redact` mode runs it. That mode adds agent-seatbelt's own regex rules, including 271 secret-token patterns, to the model's spans. These are not Desert Ant Labs' rules. The secret scores in this column come from them.
- **agent-seatbelt rules**: the agent-seatbelt rules engine alone.
- **NVIDIA GLiNER-PII** (nvidia/gliner-PII), at its card's threshold 0.3. It is English-only, so its Chinese scores show a language limit, not quality. Of the secret sets it ran only on creddata, so the Secrets suite rows leave it blank.

Bold marks the best value in each row.

**Suites**

| suite | metric | sets | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| Our core | mean typed F1 ↑ | 8 | **0.942** | 0.941 | 0.759 | 0.399 | 0.420 | 0.193 | 0.519 |
| Our core | clean rows flagged, of 574 ↓ | 7 | 115 | 101 | 127 | 211 | 218 | **27** | 142 |
| Chinese | mean typed F1 ↑ | 1 | **0.847** | 0.847 | 0.684 | 0.336 | 0.376 | 0.327 | 0.593 |
| Chinese | clean rows flagged, of 40 ↓ | 1 | 18 | 12 | 15 | 18 | 19 | **8** | 13 |
| Chinese | mean any overlap F1 ↑ | 1 | **0.998** | 0.998 | 0.937 | 0.717 | 0.717 | 0.058 | 0.596 |
| Chinese | mean person recall ↑ | 1 | **0.478** | 0.434 | 0.389 | 0.274 | 0.274 | 0.000 | 0.142 |
| Chinese | mean flags per 10k chars ↓ | 1 | 66.1 | 39.4 | 8.5 | 22.7 | 22.7 | **0.0** | 30.8 |
| English and multilingual | mean typed F1 ↑ | 4 | 0.840 | **0.840** | 0.708 | 0.668 | 0.682 | 0.374 | 0.736 |
| English and multilingual | clean rows flagged, of 647 ↓ | 3 | 122 | 121 | 155 | 105 | 111 | **64** | 253 |
| English and multilingual | mean person recall ↑ | 2 | 0.739 | 0.728 | 0.519 | 0.616 | 0.616 | 0.000 | **0.803** |
| Secrets | mean typed F1 ↑ | 4 | 0.941 | **0.942** | 0.769 | 0.000 | 0.762 | 0.772 |  |
| Secrets | clean rows flagged, of 1761 ↓ | 7 | 431 | 425 | 620 | **197** | 601 | 500 |  |
| Secrets | mean row recall ↑ | 1 | 0.727 | 0.727 | 0.909 | 0.515 | **0.939** | 0.818 |  |
| Published by the external systems | mean typed F1 ↑ | 3 | **0.901** | 0.898 | 0.753 | 0.521 | 0.548 | 0.343 | 0.794 |
| Published by the external systems | clean rows flagged, of 415 ↓ | 2 | 34 | 34 | 126 | **24** | 111 | 92 | 198 |
| Published by the external systems | mean person recall ↑ | 7 | 0.669 | 0.475 | 0.340 | 0.484 | 0.484 | 0.000 | **0.766** |

↑ means higher is better, ↓ lower is better. The chain lowers false flags on news text (wikinews_zhtw 66.1 to 39.4) and on tw_bench clean rows (18 to 12). It also lowers person recall on public-figure sets (0.669 to 0.475). Those sets mark public figures as gold, and the guideline does not.

`experiments/compare/runs/tables.md` gives every set of every suite, with the same columns. The two groups below are the ones the suite rows summarise least well.

**Transcripts** (ours, private data: the scores are published, the windows are not)

| set | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|
| transcript_dev | F1 ↑ | 0.701 | **0.752** | 0.325 | 0.282 | 0.169 | 0.062 | 0.119 |
| transcript_dev | recall ↑ | 0.673 | 0.714 | 0.613 | 0.298 | 0.339 | 0.107 | **0.899** |
| transcript_dev | precision ↑ | 0.732 | **0.795** | 0.221 | 0.268 | 0.113 | 0.043 | 0.064 |
| transcript_dev | flags per 10k chars ↓ | 1.9 | **1.9** | 5.3 | 2.2 | 5.4 | 3.4 | 28.1 |
| transcript_test | F1 ↑ | 0.739 | **0.750** | 0.414 | 0.321 | 0.187 | 0.055 | 0.167 |
| transcript_test | recall ↑ | 0.654 | 0.692 | 0.692 | 0.346 | 0.346 | 0.077 | **0.885** |
| transcript_test | precision ↑ | **0.850** | 0.818 | 0.295 | 0.300 | 0.129 | 0.043 | 0.092 |
| transcript_test | flags per 10k chars ↓ | **2.2** | 2.4 | 6.7 | 3.5 | 7.9 | 5.2 | 28.7 |

transcript_test has 26 gold entities, so one entity moves recall by about 0.038.

**Published by the external systems** (all public, never used by a decision)

| set | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|
| nemotron_test | typed F1 ↑ | **0.979** | 0.976 | 0.794 | 0.675 | 0.764 | 0.555 | 0.890 |
| nemotron_test (of 49) | clean rows flagged ↓ | 3 | 3 | 9 | 9 | 10 | **1** | 13 |
| kiji_en | typed F1 ↑ | **0.861** | 0.858 | 0.823 | 0.619 | 0.643 | 0.327 | 0.805 |
| gretel_finance_en | typed F1 ↑ | **0.862** | 0.861 | 0.642 | 0.270 | 0.237 | 0.148 | 0.687 |
| gretel_finance_en (of 366) | clean rows flagged ↓ | 31 | 31 | 117 | **15** | 101 | 91 | 185 |
| wikiann_en | person recall ↑ | 0.907 | 0.667 | 0.578 | 0.775 | 0.775 | 0.000 | **0.934** |
| wikiann_zh | person recall ↑ | **0.599** | 0.443 | 0.314 | 0.221 | 0.221 | 0.000 | 0.428 |
| multinerd_en | person recall ↑ | 0.941 | 0.523 | 0.330 | 0.910 | 0.910 | 0.000 | **0.989** |
| multinerd_zh | person recall ↑ | 0.590 | 0.393 | 0.174 | 0.056 | 0.056 | 0.000 | **0.629** |
| conll2003_en | person recall ↑ | 0.820 | 0.666 | 0.528 | 0.699 | 0.699 | 0.000 | **0.954** |
| multiconer_en | person recall ↑ | 0.438 | 0.337 | 0.305 | 0.724 | 0.724 | 0.000 | **0.745** |
| multiconer_zh | person recall ↑ | 0.388 | 0.295 | 0.149 | 0.004 | 0.004 | 0.000 | **0.680** |

nemotron_test shares a generator with a training source (section 5). The pii-masking-300k licence needs written permission to share results, so this report leaves that set out.

## 9. Limitations and known failure modes

- **Person recall on public-figure sets is low after the chain.** The chain drops names on its public lists, so recall on sets that mark public figures falls. Examples: multinerd_en 0.941 raw to 0.523 chained, wikiann_en 0.907 to 0.667. This follows the guideline. A user who wants every name needs the raw output.
- **The raw model flags public names in news text.** wikinews_zhtw has 66.1 flags per 10k characters raw, against 8.5 for the OpenAI Privacy Filter (lower is better). The chain brings it to 39.4.
- **Casual and social text is weak.** Weibo person recall is 0.478 raw. WNUT-17 person recall is 0.508 raw (higher is better).
- **Transcript recall is about 0.7.** transcript_dev recall is 0.673 raw and 0.714 chained (higher is better).
- **Thin per-type counts on transcripts.** transcript_dev holds 168 gold entities in 3,000 windows, and transcript_test holds 26 in 312 windows. Per-type transcript scores for the rare types carry almost no information.
- **Run-to-run spread.** The release is one training copy of its recipe (section 7). Another copy of the same recipe scores a little differently.
- **Secret row recall on the seatbelt positives is 0.727,** below the OpenAI Privacy Filter (0.909) and the Redact hook (0.939) (higher is better).
- **Taiwan-only types rest on templates only.** No public data teaches `nhi_card`, `company_id`, `medical_license`, `military_id` or `household_no`.
- **Chinese name order in training.** Wismut writes 91% of Chinese names given name first, which real text does not do.
- **Optimistic scores on steered sets.** CredData, Argilla and wikinews_zhtw steered choices, so their scores are optimistic (section 7).

## 10. Licence and data provenance

**Model licence.** Apache 2.0, with a NOTICE file that carries the attributions in section 12. Every public training source is MIT, Apache 2.0 or CC BY 4.0 (Creative Commons Attribution 4.0). None is non-commercial or share-alike. Apache 2.0 can carry the MIT notice of the base model in NOTICE.

**LLM-written sources and labels.** Large language models wrote the in-house LLM-written sources and every transcript label.

**Redact and GLiNER.** Section 5, "Provenance of the labels", explains why no training label and no transcript gold comes from these systems. Their scores in section 8 are measurements, not training input.

**Released synthetic data.** The synthetic training sets ship in the data repository [CJHwong/seatbelt-pii-tagger-data](https://huggingface.co/datasets/CJHwong/seatbelt-pii-tagger-data). Before release, 530 real values from the transcript sets were matched whole-word against them: every judged personal value, plus the author's identifiers. No identifier, secret, email or phone number matched. The LLM writers had copied some names, project names and ticket prefixes from their context. The released copies swap those letter for letter, so every span offset holds. So the released copies differ from the training copies in 33 of 1,299 llm-devwork rows and 105 of 1,437 llm-agentwork rows. Every other set ships unchanged. The model is a token classifier and does not generate text.

**Evaluation sets.** Section 8 publishes aggregate scores only, never rows. The pii-masking-300k set is left out, because its licence needs written permission to share results. CredData and Weibo scores are published with the citations in section 12.

## 11. Reproducing

`./reproduce.sh` fetches the data at pinned revisions, builds the training set and the vocabulary, trains, runs the chain, scores every group and exports the ONNX files. The README lists what a public run cannot match. The largest gap is the private transcript source in section 5.

## 12. Acknowledgements

Model and training data:

| # | work | licence and holder | use |
|---|---|---|---|
| 1 | microsoft/Multilingual-MiniLM-L12-H384 | MIT License, Copyright (c) Microsoft Corporation (github.com/microsoft/unilm LICENSE) | base model: layers 0 to 5 kept, vocabulary pruned from 250,037 to 64,000 XLM-R pieces |
| 2 | XLM-R SentencePiece tokenizer (fairseq) | MIT License, Copyright (c) Facebook, Inc. and its affiliates (github.com/facebookresearch/fairseq LICENSE) | tokenizer |
| 3 | nvidia/Nemotron-PII | CC BY 4.0, NVIDIA Corporation | training data, relabelled to 18 types |
| 4 | ai4privacy/pii-masking-openpii-1.5m | CC BY 4.0, "Ai4Privacy / Ai Suisse SA" | training data, relabelled and filtered to English and Chinese scripts |
| 5 | Wismut/nym-pii-multilingual-data | MIT, Wismut | training data, relabelled and filtered |
| 6 | gretelai/synthetic_pii_finance_multilingual | Apache 2.0, Gretel | training data |
| 7 | gretelai/gretel-pii-masking-en-v1 | Apache 2.0, Gretel | training data |
| 8 | gitleaks (MIT, Zachary Rice), betterleaks (MIT), microsoft/security-utilities (MIT), Nosey Parker (Apache 2.0, Praetorian Security) | as listed | regexes that generate the secret training strings |
| 9 | Faker | MIT | synthetic values |
| 10 | Wikipedia (wikimedia/wikipedia 20231101.zh) | CC BY-SA | Traditional Chinese paragraphs that rank the vocabulary; no Wikipedia text is in the model |
| 11 | Wikidata | CC0 | the public persons and places lists |

Evaluation sets and comparison systems:

- CredData: JaeKu Yun, ShinHyung Choi, YuJeong Lee, Oleksandra Sokol, WooChul Shim, Arkadiy Melkonyan and Dmytro Kuzmenko. Project CredData: A Dataset of Credentials for Research. https://github.com/Samsung/CredData, 2021.
- Weibo NER, the revised annotation (`data/weiboNER_2nd_conll.*` in https://github.com/hltcoe/golden-horse, CC BY-SA 3.0):
  - Nanyun Peng and Mark Dredze. "Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings." EMNLP 2015.
  - Nanyun Peng and Mark Dredze. "Improving Named Entity Recognition for Chinese Social Media with Word Segmentation Representation Learning." ACL 2016.
  - Hangfeng He and Xu Sun. "F-Score Driven Max Margin Neural Network for Named Entity Recognition in Chinese Social Media." arXiv:1611.04234, 2016.
- Redact by Desert Ant Labs (https://desertant.com), model desert-ant-labs/redact, Desert Ant Labs Source-Available License 1.0. It appears in the results as a benchmarked system only.

`datasets.yaml` records the licence of every evaluation set.
