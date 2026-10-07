# seatbelt-pii-tagger against the external systems

↑ means higher is better, ↓ lower is better.

## Suites

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

## Our core

| set | by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| our_benchmark | ours | typed F1 ↑ | 0.948 | **0.950** | 0.730 | 0.331 | 0.356 | 0.216 | 0.587 |
| our_benchmark (of 138) | ours | clean rows flagged ↓ | 24 | 18 | 36 | 61 | 61 | **1** | 29 |
| blind_en | ours | typed F1 ↑ | **0.903** | 0.897 | 0.857 | 0.760 | 0.793 | 0.510 | 0.685 |
| blind_en (of 77) | ours | clean rows flagged ↓ | 20 | 18 | 16 | 23 | 27 | **11** | 36 |
| blind_zh-Hans | ours | typed F1 ↑ | 0.875 | **0.878** | 0.719 | 0.440 | 0.456 | 0.075 | 0.267 |
| blind_zh-Hans (of 110) | ours | clean rows flagged ↓ | 44 | 41 | 46 | 62 | 63 | **8** | 38 |
| blind_zh-Hant | ours | typed F1 ↑ | **0.962** | 0.957 | 0.724 | 0.479 | 0.494 | 0.107 | 0.326 |
| blind_zh-Hant (of 156) | ours | clean rows flagged ↓ | 14 | 12 | 15 | 25 | 27 | **6** | 29 |
| perturbed_fullwidth | ours | typed F1 ↑ | **0.958** | 0.958 | 0.652 | 0.331 | 0.348 | 0.183 | 0.478 |
| perturbed_fullwidth (of 83) | ours | clean rows flagged ↓ | 13 | 12 | 14 | 39 | 39 | **1** | 8 |
| perturbed_spaced | ours | typed F1 ↑ | **0.983** | **0.983** | 0.833 | 0.241 | 0.245 | 0.145 | 0.621 |
| perturbed_spaced (of 5) | ours | clean rows flagged ↓ | **0** | **0** | **0** | 1 | 1 | **0** | 1 |
| perturbed_ocr | ours | typed F1 ↑ | **0.973** | **0.973** | 0.832 | 0.316 | 0.336 | 0.095 | 0.626 |
| perturbed_ocr (of 5) | ours | clean rows flagged ↓ | **0** | **0** | **0** | **0** | **0** | **0** | 1 |
| perturbed_long | ours | typed F1 ↑ | **0.931** | 0.929 | 0.722 | 0.295 | 0.333 | 0.212 | 0.563 |

## Chinese

| set | by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| tw_bench | public | typed F1 ↑ | **0.847** | 0.847 | 0.684 | 0.336 | 0.376 | 0.327 | 0.593 |
| tw_bench (of 40) | public | clean rows flagged ↓ | 18 | 12 | 15 | 18 | 19 | **8** | 13 |
| zh_bench | public | any overlap F1 ↑ | **0.998** | 0.998 | 0.937 | 0.717 | 0.717 | 0.058 | 0.596 |
| weibo | public | person recall ↑ | **0.478** | 0.434 | 0.389 | 0.274 | 0.274 | 0.000 | 0.142 |
| wikinews_zhtw | public | flags per 10k chars ↓ | 66.1 | 39.4 | 8.5 | 22.7 | 22.7 | **0.0** | 30.8 |

## English and multilingual

| set | by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| gretel_en | public | typed F1 ↑ | 0.924 | **0.924** | 0.790 | 0.728 | 0.725 | 0.344 | 0.876 |
| gretel_en (of 209) | public | clean rows flagged ↓ | 76 | 76 | 111 | 71 | 77 | **64** | 92 |
| tab_echr | public | person recall ↑ | 0.971 | 0.971 | 0.529 | 0.721 | 0.721 | 0.000 | **1.000** |
| wnut17 | public | person recall ↑ | 0.508 | 0.485 | 0.508 | 0.510 | 0.510 | 0.000 | **0.606** |
| argilla | public | typed F1 ↑ | **0.903** | **0.903** | 0.781 | 0.742 | 0.793 | 0.369 | 0.798 |
| argilla (of 384) | public | clean rows flagged ↓ | 46 | 45 | 31 | 25 | 25 | **0** | 134 |
| spy | public | typed F1 ↑ | 0.537 | **0.539** | 0.498 | 0.486 | 0.473 | 0.363 | 0.467 |
| wismut_test_latin | public | typed F1 ↑ | **0.996** | 0.996 | 0.761 | 0.716 | 0.739 | 0.419 | 0.804 |
| wismut_test_latin (of 54) | public | clean rows flagged ↓ | **0** | **0** | 13 | 9 | 9 | **0** | 27 |

## Secrets

| set | by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| creddata | public | typed F1 ↑ | 0.806 | **0.809** | 0.568 | 0.000 | 0.308 | 0.316 | 0.495 |
| creddata (of 1000) | public | clean rows flagged ↓ | 160 | 155 | 195 | **51** | 189 | 156 | 759 |
| generated_held_out | ours | typed F1 ↑ | 0.998 | **0.999** | 0.697 | 0.000 | 0.811 | 0.844 |  |
| generated_held_out (of 344) | ours | clean rows flagged ↓ | **0** | **0** | 169 | 57 | 169 | 137 |  |
| seatbelt_secrets_trained | ours | typed F1 ↑ | **0.980** | **0.980** | 0.896 | 0.000 | 0.968 | 0.968 |  |
| seatbelt_secrets_held_out | ours | typed F1 ↑ | **0.981** | **0.981** | 0.915 | 0.000 | 0.962 | 0.962 |  |
| seatbelt_clean (of 54) | ours | clean rows flagged ↓ | 5 | 5 | 5 | 9 | 10 | **3** |  |
| seatbelt_positive | ours | row recall ↑ | 0.727 | 0.727 | 0.909 | 0.515 | **0.939** | 0.818 |  |
| seatbelt_test_cases (of 25) | ours | clean rows flagged ↓ | 20 | 19 | 22 | **11** | 23 | 19 | 20 |
| seatbelt_holdout (of 52) | ours | clean rows flagged ↓ | 40 | 40 | 36 | **2** | 29 | 29 | 45 |
| seatbelt_final_holdout (of 46) | ours | clean rows flagged ↓ | 39 | 39 | 31 | **1** | 21 | 21 | 38 |
| seatbelt_challenge (of 240) | ours | clean rows flagged ↓ | 167 | 167 | 162 | **66** | 160 | 135 | 189 |

## Published by the external systems

| set | by | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|---|
| nemotron_test | public | typed F1 ↑ | **0.979** | 0.976 | 0.794 | 0.675 | 0.764 | 0.555 | 0.890 |
| nemotron_test (of 49) | public | clean rows flagged ↓ | 3 | 3 | 9 | 9 | 10 | **1** | 13 |
| kiji_en | public | typed F1 ↑ | **0.861** | 0.858 | 0.823 | 0.619 | 0.643 | 0.327 | 0.805 |
| gretel_finance_en | public | typed F1 ↑ | **0.862** | 0.861 | 0.642 | 0.270 | 0.237 | 0.148 | 0.687 |
| gretel_finance_en (of 366) | public | clean rows flagged ↓ | 31 | 31 | 117 | **15** | 101 | 91 | 185 |
| wikiann_en | public | person recall ↑ | 0.907 | 0.667 | 0.578 | 0.775 | 0.775 | 0.000 | **0.934** |
| wikiann_zh | public | person recall ↑ | **0.599** | 0.443 | 0.314 | 0.221 | 0.221 | 0.000 | 0.428 |
| multinerd_en | public | person recall ↑ | 0.941 | 0.523 | 0.330 | 0.910 | 0.910 | 0.000 | **0.989** |
| multinerd_zh | public | person recall ↑ | 0.590 | 0.393 | 0.174 | 0.056 | 0.056 | 0.000 | **0.629** |
| conll2003_en | public | person recall ↑ | 0.820 | 0.666 | 0.528 | 0.699 | 0.699 | 0.000 | **0.954** |
| multiconer_en | public | person recall ↑ | 0.438 | 0.337 | 0.305 | 0.724 | 0.724 | 0.000 | **0.745** |
| multiconer_zh | public | person recall ↑ | 0.388 | 0.295 | 0.149 | 0.004 | 0.004 | 0.000 | **0.680** |

## Transcripts (private)

| set | metric | raw | + chain | OpenAI Privacy Filter | Redact model | Redact hook | agent-seatbelt rules | NVIDIA GLiNER-PII |
|---|---|---|---|---|---|---|---|---|
| transcript_dev | f1 ↑ | 0.701 | **0.752** | 0.325 | 0.282 | 0.169 | 0.062 | 0.119 |
| transcript_dev | recall ↑ | 0.673 | 0.714 | 0.613 | 0.298 | 0.339 | 0.107 | **0.899** |
| transcript_dev | precision ↑ | 0.732 | **0.795** | 0.221 | 0.268 | 0.113 | 0.043 | 0.064 |
| transcript_dev | flags per 10k chars ↓ | 1.9 | **1.9** | 5.3 | 2.2 | 5.4 | 3.4 | 28.1 |
| transcript_test | f1 ↑ | 0.739 | **0.750** | 0.414 | 0.321 | 0.187 | 0.055 | 0.167 |
| transcript_test | recall ↑ | 0.654 | 0.692 | 0.692 | 0.346 | 0.346 | 0.077 | **0.885** |
| transcript_test | precision ↑ | **0.850** | 0.818 | 0.295 | 0.300 | 0.129 | 0.043 | 0.092 |
| transcript_test | flags per 10k chars ↓ | **2.2** | 2.4 | 6.7 | 3.5 | 7.9 | 5.2 | 28.7 |
