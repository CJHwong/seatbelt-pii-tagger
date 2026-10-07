# seatbelt-pii-tagger

A 35M-parameter token tagger that finds personal data and secrets in English, Simplified Chinese and Traditional Chinese text. It is the `tagger` mode of the [agent-seatbelt](https://github.com/CJHwong/agent-seatbelt) PII hook.

This repository holds the code and the recipe that train it. The model, its card and the release data are on Hugging Face:

- Model: [CJHwong/seatbelt-pii-tagger](https://huggingface.co/CJHwong/seatbelt-pii-tagger), release `2026.10`
- Release data: [CJHwong/seatbelt-pii-tagger-data](https://huggingface.co/datasets/CJHwong/seatbelt-pii-tagger-data), the in-house synthetic sets and the public names lists

[docs/model_card.md](docs/model_card.md) and [docs/technical_report.md](docs/technical_report.md) give the labels, the method and every result.

## Reproduce the release

You need [uv](https://docs.astral.sh/uv/) and git.

```sh
./reproduce.sh                    # the release recipe, 10,000 steps
TRAIN_STEPS=200 ./reproduce.sh    # a quick check of the whole pipeline
```

The script fetches every dataset at its pinned revision, builds the training set and the vocabulary, trains the run `release`, applies the rule chain, scores every evaluation group and exports the ONNX files. The results go to `experiments/compare/runs/tables.md` and `experiments/export/build/release/`.

Training uses Apple MPS (Metal Performance Shaders) when present, else the CPU. The release took 26 minutes on an Apple M5 Pro. MPS training is not deterministic, so two runs of one recipe differ a little.

## What a public run cannot match

1. **Private transcript windows.** The release trained on 7,740 windows from the author's own coding-agent sessions. They hold real text, so they are not published. A public run skips that source, so its scores differ from the release. The transcript evaluation sets are private too, and their rows are missing from a public `tables.md`.
2. **Scrubbed rows.** In 138 rows of the two LLM-written work sets, the published copies swap names and project details for fakes. The release trained on the copies before that swap.
3. **Generated sets ship as data.** `scripts/rule_secrets.py` and `scripts/synthesize.py` give the same output only under a fixed `PYTHONHASHSEED`. The release copies therefore come from the data repository, not from a new run of the generators.
4. **CredData** needs its own downloader, on Linux. `fetch_data.py` prints the step. The evaluation skips CredData until its data is present.
5. **ai4privacy/pii-masking-300k** needs written permission to share results, so the published tables leave it out. To score it on your machine, fetch it with `uv run scripts/fetch_data.py --only pii-masking-300k`.

## Add your own data

A source in `type_map.yaml` can name a local file with `path:` and `private: true`. The build reads it when the file exists and skips it with a warning when it does not. Put sources you keep out of the shared map in `type_map.local.yaml`. Put extra runs and systems to score in `experiments/compare/systems.local.yaml`. Both files are optional.

## Layout

| path | holds |
|---|---|
| `datasets.yaml` | every dataset and external source, its pinned revision and licence, and the excluded datasets with the reason |
| `type_map.yaml` | how each source's labels map to the 18 types |
| `scripts/` | fetch, verify, generate and build the training set |
| `synthetic/` | the Traditional Chinese templates |
| `data/synthetic/llm_*/` | the briefs and checks for the LLM-written sets |
| `benchmark/`, `blind/` | our evaluation sets |
| `experiments/vocab_prune/` | the 64,000-piece vocabulary |
| `experiments/first_run/` | training |
| `experiments/compare/` | the rule chain, the external systems and scoring |
| `experiments/export/` | the ONNX export |

## Licence

Apache 2.0. [NOTICE](NOTICE) lists the works the model includes or derives from. `datasets.yaml` records the licence of each data source.
