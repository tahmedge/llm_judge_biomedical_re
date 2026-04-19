# LLM-Judge for Biomedical Relation Extraction

[![ACL 2025](https://img.shields.io/badge/ACL%202025-Main%20Conference-blue)](https://aclanthology.org/2025.acl-long.1238.pdf)

Official repository for the paper [**Improving Automatic Evaluation of Large Language Models (LLMs) in Biomedical Relation Extraction via LLMs-as-the-Judge**](https://aclanthology.org/2025.acl-long.1238.pdf) — *ACL 2025 (Main Conference, Long Paper)*

---

## 📖 Overview

Large Language Models (LLMs) achieve strong zero-shot performance on **biomedical relation extraction**, but evaluating them is notoriously hard. Traditional metrics like string matching and token overlap fail when LLMs produce **synonyms, abbreviations, or paraphrased answers** (e.g., "Hepatic carcinoma" vs. "Liver cancer"). Human evaluation is reliable but slow and expensive.

This paper presents the **first comprehensive study of LLMs-as-the-Judge for biomedical relation extraction**. We benchmark 8 LLM-Judges on responses from 5 LLM-Generators across 3 biomedical RE datasets, and propose two techniques to substantially improve judge reliability:

1. 📋 **Structured Output Formatting** — require LLM-Generators to emit JSON-formatted relations, making them easier to evaluate.
2. 🔄 **Domain Adaptation via Transfer Learning** — fine-tune open-source LLM-Judges on out-of-domain RE judgment data to boost in-domain performance.

---

## ✨ Key Contributions

- 🔬 **First comprehensive benchmark** of LLM-Judges for biomedical relation extraction — 8 judges × 5 generators × 3 datasets.
- ⚠️ **Reveals that LLM-Judges perform poorly** in biomedical RE (typically <50% accuracy), far below human evaluators.
- 📋 **Structured output formatting** improves LLM-Judge performance by ~15% on average across datasets.
- 🔄 **Domain-adaptive fine-tuning** enables small open-source models (3B–7B) to outperform proprietary LLMs.
- 📊 **100+ experiments** analyzing structured output, domain adaptation, and model scaling effects.
- 📦 **36K judgment samples released**: 4K human-annotated + 32K LLM-annotated across 3 datasets.

---

## 🧩 Datasets

| Dataset | Task | Test Size |
|---|---|---|
| **BC5CDR** | Chemical–Disease Relation Extraction | 500 |
| **KD-DTI** | Drug–Target Interaction Extraction | 1,159 |
| **DDI**| Drug–Drug Interaction Extraction | 191 |

---

## 🤖 Models Evaluated

### LLM-Generators (models being evaluated)
GPT-3.5, GPT-4-Turbo, Claude-2, PaLM-2, LLaMA-2-13B-Instruct

### LLM-Judges (evaluators)

**Closed-source (cost-optimized variants):**
- GPT-4o-Mini
- Gemini-1.5-Flash
- Claude-3-Haiku

**Open-source (≤10B parameters, deployable on 1× L4 GPU):**
- LLaMA-3.1-8B-Instruct
- Qwen-2.5-7B-Instruct
- Phi-3.5-Mini-3.8B-Instruct
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-LLaMA-8B

---

## 🔬 Methodology

### 1. Prompting the LLM-Judge

The judge receives the task description, biomedical text, gold relations, and AI-predicted relations, and must output a JSON with:
- `correctly_predicted_relations`
- `total_predicted_relations`

### 2. Structured Output Formatting

Instead of asking the LLM-Generator for free-form relation lists, we require a JSON-structured output:

```json
{"drug": "salbutamol", "side-effect": "tremor"}
```

This makes the generator's output machine-parsable and dramatically improves judge accuracy.

### 3. Domain Adaptation via Transfer Learning

Fine-tune open-source LLM-Judges on **out-of-domain** human-annotated judgment data (e.g., train on BC5CDR → evaluate on KD-DTI) to specialize them for biomedical RE evaluation without needing target-domain labels.

---

## 📊 Major Results

### Existing Benchmarks (Unstructured Responses)

Most LLM-Judges fail to exceed 50% accuracy — only GPT-4o-Mini crosses that threshold but never reaches 60%.

| LLM-Judge | BC5CDR EM ↑ | DDI EM ↑ | KD-DTI EM ↑ |
|---|---|---|---|
| **GPT-4o-Mini** | **48.35** | **59.03** | **53.11** |
| Gemini-1.5-Flash | 42.55 | 47.12 | 40.68 |
| Claude-3-Haiku | 29.50 | 31.15 | 40.27 |
| Qwen-2.5-7B-Instruct | 45.25 | 46.60 | 49.98 |
| LLaMA-3.1-8B-Instruct | 29.45 | 29.32 | 36.73 |
| Phi-3.5-Mini-3.8B | 33.80 | 43.06 | 35.55 |

### Structured Output Formatting Improves Accuracy Substantially

| LLM-Judge | BC5CDR (Struct ↑) | BC5CDR (Unstruct) | KD-DTI (Struct ↑) | KD-DTI (Unstruct) |
|---|---|---|---|---|
| **Gemini-1.5-Flash** | **70.20** | 48.40 | 41.59 | 41.52 |
| Claude-3-Haiku | 67.80 | 36.80 | 59.28 | 50.04 |
| GPT-4o-Mini | 67.20 | 43.20 | **72.39** | 66.35 |
| Qwen-2.5-7B-Instruct | 67.80 | 44.60 | 61.78 | 55.82 |
| LLaMA-3.1-8B-Instruct | 59.20 | 33.00 | 40.38 | 38.22 |
| Phi-3.5-Mini-3.8B | 53.80 | 35.60 | 36.94 | 36.74 |

**Paired t-test confirms statistically significant improvements (p < 0.05) across all three datasets.**

### Domain Adaptation via Transfer Learning

Fine-tuning a small open-source model on out-of-domain data produces **state-of-the-art performance**, surpassing even closed-source models:

| LLM-Judge | Fine-tune On | Evaluate On | EM Accuracy (Δ vs. Zero-Shot) |
|---|---|---|---|
| Qwen-2.5-7B-Instruct | BC5CDR | KD-DTI | **75.75 (+13.97)** |
| Qwen-2.5-7B-Instruct | KD-DTI | BC5CDR | **71.40 (+3.60)** |
| Phi-3.5-Mini-3.8B | BC5CDR | KD-DTI | **69.54 (+32.60)** |
| Phi-3.5-Mini-3.8B | KD-DTI | BC5CDR | **64.80 (+11.00)** |

### Fine-Tuned Small Models Can Beat Larger Zero-Shot Models

On KD-DTI, a fine-tuned **Qwen-2.5-3B** (70.49%) surpasses zero-shot **Qwen-2.5-7B** (61.78%) — making small models viable for resource-constrained deployment.

---

## 🔑 Key Findings

- ⚠️ **Zero-shot LLM-Judges are unreliable** for biomedical RE — domain specificity breaks evaluation paradigms that work on general NLP.
- 📋 **Structured output formatting is essential** — it consistently and significantly improves every judge tested.
- 🔄 **Domain adaptation transfers across biomedical RE datasets** — fine-tuning on BC5CDR helps KD-DTI evaluation and vice versa.
- 💪 **Small open-source models (3B–7B) can outperform proprietary models** when fine-tuned with our approach.
- 🧪 **Reasoning models (DeepSeek-R1-Distilled)** don't automatically outperform instruction-tuned counterparts on this task.
- 🩺 **Biomedical domain models (BioMistral-7B)** failed to follow judging instructions — general-purpose instruction-following matters more than biomedical pretraining for this task.
- 📉 **Reducing model size dramatically hurts zero-shot performance** (e.g., Qwen-2.5 drops 45–74% going from 7B → 1.5B), but fine-tuning recovers most of the gap.

---

## 📏 Evaluation Metrics

- **Exact Match (EM) Accuracy** ↑ — Percentage of judgments where both `correctly_predicted_relations` and `total_predicted_relations` match the human gold label exactly.
- **Root Mean Squared Error (RMSE)** ↓ — Distance between LLM-Judge and human annotation counts.

---

## 📝 Citation

If you use this code, data, or findings, please cite:

```bibtex
@inproceedings{laskar2025improving,
  title     = {Improving Automatic Evaluation of Large Language Models (LLMs) in Biomedical Relation Extraction via LLMs-as-the-Judge},
  author    = {Laskar, Md Tahmid Rahman and Jahan, Israt and Dolatabadi, Elham
               and Peng, Chun and Hoque, Enamul and Huang, Jimmy Xiangji},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {25483--25497},
  year      = {2025},
  address   = {Vienna, Austria},
  publisher = {Association for Computational Linguistics}
}
```

---

## 👥 Authors

Md Tahmid Rahman Laskar¹, Israt Jahan¹, Elham Dolatabadi¹'², Chun Peng¹, Enamul Hoque¹, Jimmy Xiangji Huang¹

¹ York University, Toronto, Canada  ² Vector Institute

📧 Contact: `{tahmedge, jhuang}@yorku.ca`

---

## 🙏 Acknowledgements

This research was supported by the Natural Sciences and Engineering Research Council (NSERC) of Canada, the NSERC Discovery Grant, the York Research Chairs (YRC) program, and Compute Canada.

---

## ⚖️ Ethical Considerations

The proposed LLM-Judge is designed **solely for evaluating LLM-generated responses in biomedical relation extraction** and is **not intended for direct clinical use** or end-user biomedical decision-making. The accuracy of the relation extraction system depends on the LLM-Generator, and the judge only scores predictions — it does not drive medical decisions. For real-world deployment, we recommend a **human-in-the-loop approach** where expert annotators verify the outputs of models that score well under LLM-Judge evaluation. All human annotations were conducted by two authors of the paper.

---

## ⚠️ Limitations

- LLM-Judges still fall short of human evaluators on biomedical RE — deep domain reasoning remains a gap.
- Benchmarking is limited to cost-efficient model variants; larger reasoning models (e.g., OpenAI O3) were not evaluated due to cost.
- Only three biomedical RE datasets were considered; generalization to other biomedical tasks (e.g., event extraction, entity linking) is future work.
- Prompt engineering was performed on a small subset rather than exhaustively — other prompt variants may yield further gains.

---

## 🔗 Related Work

- [Chart LVLM Judge](https://github.com/tahmedge/chart_lvlm_judge) — LVLM-as-a-Judge for chart comprehension (ACL/EMNLP 2025 Industry)
- [MM-JudgeBench](https://github.com/tahmedge/mm-judgebench) — Multilingual & multimodal LVLM judge benchmark (ACL 2026 Findings)
