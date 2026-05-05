# Literature Review & Gap Analysis
**Project:** Suicidality and Depression Detection — Cross-lingual NLP Benchmark  
**Author:** Alina Erkulova, MSc Data Science, ELTE  
**Date:** May 2026

> All claims in this document are verified against primary sources.  
> Papers are only cited where the specific result was confirmed by reading the actual paper or its full HTML/PDF version.

---

## How to read this document

Each section covers one paper. For each paper you get:
- What they did
- Their actual verified results (not from search summaries)
- How it relates to your project
- What you can learn or borrow

At the end: a consolidated gap analysis and action list.

---

## Paper 1 — The Russian VK Dataset (Your Data Source)

**Citation:**  
Narynov, S., Mukhtarkhanuly, D., & Omarov, B. (2020). *Dataset of depressive posts in Russian language collected from social media.* Data in Brief, Elsevier.  
DOI: https://doi.org/10.1016/j.dib.2020.105122  
PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC7016367/

### What they did
This is a **data paper**, not a modelling paper. The authors built the dataset you are using. They:
- Scraped public VKontakte accounts from CIS countries (Russia, Kazakhstan, Ukraine, Belarus, and others) using the VK API and depression-related keywords
- Had posts annotated as depressive or non-depressive by psychiatrists from the Republican Scientific and Practical Center of Mental Health
- Released 32,018 depressive + 32,021 non-depressive posts (64,039 total)
- Did a descriptive analysis: text length distributions, most frequent unigrams/bigrams, author age distribution

### Their actual results
**No classification F1 or accuracy scores are reported.** The paper is purely a dataset contribution. They describe LSTM and BiLSTM as potential classification approaches but do not report training results.

### How this relates to your project
You are using their dataset directly. This is the paper you **must cite** as the data source. Because no classification benchmark exists in this paper, your results (SVM F1=0.9948, XLM-R F1=0.9942) may be among the **first published model benchmarks on this dataset**.

### What you should note in your thesis
1. Cite this as the dataset source (mandatory)
2. The data comes from **multiple CIS countries**, not only Russia — this is relevant to your geographic bias finding (Kazakhstan as a predictive feature)
3. The paper does not report model performance → your results fill that gap
4. The annotation was done by psychiatrists — high quality, but only binary labels

---

## Paper 2 — The C-SSRS Dataset (Your Data Source)

**Citation:**  
Gaur, M., Alambo, A., Sain, J. P., Kursuncu, U., Thirunarayan, K., Kavuluru, R., Sheth, A., Welton, R. S., & Pathak, J. (2019). *Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention.* In Proceedings of the World Wide Web Conference (WWW 2019).  
https://scholarcommons.sc.edu/aii_fac_pub/4/

### What they did
This is the paper that **created the C-SSRS Reddit dataset** you use. They:
- Identified 2,181 Reddit users who discussed suicidal ideation or attempts across the platform
- Had 4 practicing psychiatrists annotate 500 of those users using the Columbia Suicide Severity Rating Scale
- Inter-annotator agreement: 0.79 pairwise, 0.73 group-wise
- Defined a **5-class** scheme: Supportive, No Risk, Low Risk, Moderate Risk, High Risk
- Used domain knowledge (DSM-5, SNOMED-CT, suicide lexicons) + CNN and SVM-L for classification

### Their actual results
- CNN + domain knowledge outperformed SVM-L
- Improvement over prior 4-class state-of-the-art: +4.2% graded recall, -12.5% perceived risk measure
- **No absolute F1 scores are reported** — they report relative improvement over their own baseline

### Critical difference from your setup
The original paper uses **5 classes**. You collapse to **binary** (suicidal vs. non-suicidal). This means:
- You cannot directly compare your F1=0.73 SVM to their numbers
- Your task is simpler (binary), so higher F1 is expected
- You must state this difference explicitly in your thesis

### What you should note in your thesis
1. Cite as the dataset source (mandatory)
2. Acknowledge the binary vs. 5-class difference
3. Your SVM (0.73) and BERT (0.71) on this dataset are respectable for the binary case — the dataset is genuinely hard (indirect, clinical language)
4. The original authors found domain knowledge helpful — this is a direction you did not explore

---

## Paper 3 — Temporal vs. Non-Temporal Assessment on C-SSRS

**Citation:**  
Gaur, M., Aribandi, V., Alambo, A., Kursuncu, U., Thirunarayan, K., Beich, J., Pathak, J., & Sheth, A. (2021). *Characterization of time-variant and time-invariant assessment of suicidality on Reddit using C-SSRS.* PLOS ONE.  
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250448

### What they did
Same dataset as Paper 2 (C-SSRS), but 448 users this time. They asked: does the **order** of posts matter for suicide risk prediction?
- Time-Variant Model (TvarM): LSTM+CNN, processes posts in temporal order
- Time-Invariant Model (TinvM): CNN, aggregates all posts regardless of order
- Both used medical knowledge (DSM-5, SNOMED-CT, ICD-10)

### Their actual results (verified)

| Task | TvarM | TinvM |
|---|---|---|
| Suicidal Ideation | AUC 0.78 | Lower |
| Suicide Behavior | Lower | +26.5% F1 improvement |
| Suicide Attempt | Lower | +21% F1 improvement |

Conclusion: TvarM better for detecting **ideation**, TinvM better for detecting **behavior/attempt**. Authors recommend a hybrid model.

### How this relates to your project
You treat each post as independent — no temporal dimension. This paper shows that for C-SSRS, temporal patterns carry signal. This is a **gap in your approach** you can acknowledge in limitations.

### What you can learn
- Single-post classification (your approach) is a simplification
- For a stronger C-SSRS model: user-level analysis with temporal features would help
- Your binary simplification also removes the granularity that makes temporal models useful

---

## Paper 4 — MentalBERT: Domain-Adapted Transformer

**Citation:**  
Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P., & Cambria, E. (2022). *MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare.* In Proceedings of LREC 2022.  
arXiv: https://arxiv.org/abs/2110.15621

### What they did
- Took standard BERT and RoBERTa and **continued pre-training** on 13,671,785 sentences from 7 mental health Reddit communities: r/depression, r/SuicideWatch, r/Anxiety, r/offmychest, r/bipolar, r/mentalillness, r/mentalhealth
- This produces MentalBERT and MentalRoBERTa
- Evaluated on 8 mental health datasets (depression, suicide, stress, anxiety)
- Available on HuggingFace: `mental/mental-bert-base-uncased`

### Their actual results (verified from full paper)

| Dataset | Task | MentalBERT F1 | MentalRoBERTa F1 | BERT-base F1 |
|---|---|---|---|---|
| UMD | Suicide (Reddit) | **58.26** | — | 58.01 |
| T-SID | Suicide (Twitter) | — | **89.01** | 88.51 |
| eRisk18 T1 | Depression | — | **93.38** | 88.54 |
| CLPsych15 | Depression (Twitter) | — | **69.71** | 62.75 |
| Depression_Reddit | Depression | 94.62 | 95.11 | — |
| SWMH | Multi-disorder | — | **72.16** | 72.03 |
| Dreaddit | Stress | — | — | — |

### Critical observation
On **suicide detection specifically (UMD dataset)**: MentalBERT improves over BERT by only **+0.25 F1 points** (58.26 vs 58.01). That is a negligible difference.  
The larger gains appear on **depression detection** tasks (up to +6.96 F1 on CLPsych15).

### How this relates to your project
MentalBERT is not trained on your specific datasets and its gains on suicide detection are very small. However:
- It is easy to add to your pipeline (one config line: `mentalbert: mental/mental-bert-base-uncased`)
- It gives you a methodological point: *"Does domain-adaptive pre-training help beyond general BERT?"*
- If your results show small/no improvement, that is a valid finding — consistent with what Ji et al. found on the UMD suicide dataset

### What you can add
Adding MentalBERT is low effort (same `run_bert_experiment` function, new model ID) and directly responds to the supervisor's request to "use methods from recent papers."

---

## Paper 5 — Deep Learning with Label Correction

**Citation:**  
Haque, A., Reddi, V., & Giallanza, T. (2021). *Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction.* arXiv:2102.09427.  
https://arxiv.org/abs/2102.09427

### What they did
- Used a **small** Reddit dataset: 1,895 posts from r/SuicideWatch (suicidal) and r/Depression (depressed)
- Argued that web-scraped datasets have noisy labels (posts labelled by subreddit, not by clinicians)
- Proposed SDCNL: combines BERT/SBERT/GUSE embeddings with unsupervised label correction (PCA/UMAP + GMM clustering, replace labels when confidence > 0.90)

### Their actual results (verified)

| Model | Accuracy | F1-Score | AUC |
|---|---|---|---|
| GUSE + Dense | 72.24% | 73.61% | 77.76% |
| BERT + CNN | 72.14% | 72.92% | 76.35% |
| BERT + Dense | 70.50% | 71.25% | 75.43% |

Their best F1 is **73.61%** — on a 1,895-post dataset. This is substantially lower than your Reddit results, because their dataset is much smaller and mixes r/SuicideWatch with r/Depression (a harder distinction than suicidal vs. control).

### Critical observation
Their paper raises an important methodological concern: labels based on subreddit membership are **proxies**, not ground truth. A post in r/SuicideWatch might not describe suicidal ideation; a post in r/Depression might contain suicidal content. Your Reddit dataset has the same issue.

### What you can learn
- Your thesis should acknowledge **label noise** in Reddit-based datasets as a limitation
- The 0.9653 BERT F1 on your Reddit dataset may partly reflect how lexically separable the subreddits are, rather than how well the model understands suicidality
- This connects to your SHAP analysis: if models learn subreddit-specific vocabulary rather than psychological content, generalisability suffers

---

## Paper 6 — Review of Reddit-Based Suicidal Ideation Detection

**Citation:**  
Yeskuatov, E., Chua, S.-L., & Foo, L. K. (2022). *Leveraging Reddit for Suicidal Ideation Detection: A Review of Machine Learning and NLP Techniques.* PMC/International Journal of Environmental Research and Public Health.  
https://pmc.ncbi.nlm.nih.gov/articles/PMC9407719/

### What they did
A systematic review of 21 supervised ML algorithms applied to Reddit-based suicide/depression datasets. Not an experiment — a survey paper.

### Key performance numbers they report (across reviewed papers)

| Model | Dataset | F1-Score | Accuracy |
|---|---|---|---|
| XGBoost | Ji et al. 7k binary Reddit | **0.957** | — |
| RoBERTa | Binary classification | — | **95.21%** |
| LSTM-CNN | User-level UMD | — | 93.8% |
| SVM | UMD multiclass | 0.460 | — |
| BERT-Softmax | Multiclass risk levels | 0.477 | — |

**Important:** These numbers come from different papers using different datasets (not your 232k dataset). Direct comparison requires caution.

### Key conclusion from the review
> "Standard ML methods outperform newer deep learning techniques" — found in 3 of the reviewed studies.

This directly supports your own finding that SVM is competitive with BERT on several datasets.

### How this relates to your project
- Your finding (SVM nearly matches BERT) is **consistent with the broader literature**
- You can cite this review to support that conclusion
- The SVM multiclass F1 of 0.460 shows multiclass tasks are much harder than binary — relevant context for C-SSRS

---

## Paper 7 — First Multilingual Model for Suicide Text Detection

**Citation:**  
Aguirre, et al. (2024). *The First Multilingual Model For The Detection of Suicide Texts.* Second Workshop on Scaling Up Multilingual & Multi-Cultural Evaluation, January 2025 (ACL Anthology).  
arXiv: https://arxiv.org/abs/2412.15498

### What they did
- Took a Spanish suicide ideation Twitter dataset (2,068 tweets: 498 positive, 1,570 negative)
- **Translated it** to English, German, Catalan, Portuguese, Italian using SeamlessM4T
- Fine-tuned mBERT, XLM-R, mT5 on **all 6 languages simultaneously**
- Evaluated on test sets in each language

### Their actual results (verified)

| Language | mBERT F1 | XLM-R F1 | mT5 F1 |
|---|---|---|---|
| Spanish | 82.1 | 84.3 | **87.7** |
| English | 83.3 | 85.5 | **88.1** |
| Italian | 78.5 | 80.6 | **83.2** |
| German | 80.9 | 82.9 | **86.1** |
| Catalan | 81.1 | 82.7 | **86.1** |
| Portuguese | 79.9 | 81.7 | **84.8** |

### Critical difference from your zero-shot experiment
**This is NOT zero-shot.** They train on data in all 6 languages simultaneously (via translation). Your experiment trains on English only and tests on Russian with no Russian data at all. Your setup is significantly more constrained and harder.

| | Aguirre et al. (2024) | Your experiment |
|---|---|---|
| Training languages | 6 (via translation) | 1 (English only) |
| Test language | Same 6 languages | Russian (unseen) |
| Setup | Multilingual fine-tuning | True zero-shot transfer |
| XLM-R F1 | 84.3–85.5 | 78.82 |

Your **0.79 F1** on true zero-shot is a strong result. Their XLM-R with actual training data in every language gets 84-85%. The 5-6 point gap is the cost of having zero target-language data — and that is exactly your research question.

### How this relates to your project
- You can cite this as the most recent related multilingual work and clearly explain why your setup is more challenging
- Their paper covers 6 European languages; Russian is absent — **your work extends cross-lingual suicide detection to Russian**
- mT5 consistently outperforms XLM-R in their setting; this suggests mT5 could be an additional model to try in your zero-shot experiment

---

## Paper 8 — Evaluating Transformer Models: IEEE BigData Competition

**Citation:**  
Pokrywka, J., Kaczmarek, J. I., & Gorzelańczyk, E. J. (2024). *Evaluating Transformer Models for Suicide Risk Detection on Social Media.* arXiv:2410.08375.  
https://arxiv.org/html/2410.08375v1

### What they did
Participated in the IEEE BigData 2024 Cup on suicide risk detection. Dataset: **500 annotated Reddit posts**, 4-class (Indicator, Ideation, Behavior, Attempt) from r/SuicideWatch + 14 related subreddits.

### Their actual results (verified)

| Model | Weighted F1 |
|---|---|
| DeBERTa-base (single) | 64.8 |
| DeBERTa-large (ensemble) | 73.0 |
| GPT-4o fine-tuned (single) | 73.6 |
| **GPT-4o fine-tuned (ensemble)** | **74.8** |
| Competition 1st place | 76.1 |

### Important caveat
This is a **different dataset** from yours — a 4-class competition dataset, not the 232k binary Reddit dataset. Results are **not directly comparable** to your numbers.

### What you can learn
- 4-class severity classification is significantly harder than binary detection (F1 ~73 vs. your 0.96)
- GPT-4o fine-tuned outperforms DeBERTa — LLMs are now competitive with encoder-only models when fine-tuned
- This is the current state of the art for **multiclass** suicide risk on Reddit
- If you wanted a stretch goal: convert C-SSRS back to 5-class and compare to this line of work

---

## Consolidated Gap Analysis

### What your project does well
| Strength | Evidence |
|---|---|
| Multi-dataset benchmark (4 datasets) | Most papers use 1–2 datasets |
| Includes Russian — rare in the field | Aguirre et al. (2024) covers 6 European languages, not Russian |
| True zero-shot cross-lingual transfer | No other verified paper does EN→RU zero-shot on social media |
| Baseline comparisons (majority + keyword) | Many papers lack these |
| SHAP explainability + artifact discovery | Found geographic/temporal bias — a genuine contribution |
| McNemar statistical tests | Most comparative papers skip this |

---

### What is missing or weak

#### 1. No comparison to domain-adapted language models (HIGH PRIORITY)
**Gap:** You compare BERT (general) vs. SVM but never ask whether a mental-health-specific BERT would help.  
**Fix:** Add MentalBERT (`mental/mental-bert-base-uncased`) to your experiments. Same pipeline, one config entry.  
**Reference:** Ji et al. (2022) — MentalBERT

#### 2. No discussion of label noise (MEDIUM PRIORITY)
**Gap:** Your Reddit labels come from subreddit membership, not clinical annotation. Haque et al. (2021) show this creates noisy labels that inflate performance metrics.  
**Fix:** Add a paragraph in your thesis limitations: acknowledge that Reddit-based labels are weak supervision and that the 0.9653 F1 may partly reflect lexical differences between subreddits rather than genuine suicide detection.  
**Reference:** Haque et al. (2021)

#### 3. Binary vs. multi-class on C-SSRS not explained (MEDIUM PRIORITY)
**Gap:** The original C-SSRS paper (Gaur et al. 2019) uses 5 classes. You use 2. You never explain this decision.  
**Fix:** Add one sentence: *"We collapse the C-SSRS labels to binary (suicidal/non-suicidal) to enable cross-dataset comparison. The original 5-class annotation scheme (Gaur et al. 2019) encodes severity; binarisation loses this information but simplifies evaluation."*  
**Reference:** Gaur et al. (2019)

#### 4. No temporal/user-level analysis for C-SSRS (LOW PRIORITY for now)
**Gap:** Gaur et al. (2021) show that temporal post sequences carry signal for C-SSRS. Your model treats posts independently.  
**Fix:** Acknowledge in limitations. Could be a future work item.  
**Reference:** Gaur et al. (2021)

#### 5. No comparison to mT5 in zero-shot experiment (LOW PRIORITY)
**Gap:** Aguirre et al. (2024) show mT5 consistently outperforms XLM-R for multilingual suicide detection.  
**Fix:** Add mT5 as an optional additional zero-shot model (requires HuggingFace `google/mt5-base`).  
**Reference:** Aguirre et al. (2024)

#### 6. No published benchmark to compare Russian VK results to (NEUTRAL)
**Observation:** The Narynov et al. (2020) dataset paper does not report classification benchmarks. No other verified paper publishes results on this dataset.  
**This is actually a positive:** Your work provides the first published benchmarks on this dataset. State this explicitly in your contribution section.

---

## Recommended Comparison Table for Thesis

*Include this in your results/discussion chapter. Use † to flag dataset differences.*

| Work | Dataset | Best Model | F1 |
|---|---|---|---|
| Gaur et al. (2019) | C-SSRS Reddit (5-class) | CNN + knowledge | +4.2% recall vs. prior SOTA† |
| Haque et al. (2021) | Reddit r/SW + r/Depression (1.9k) | GUSE + Dense | 0.7361 |
| Ji et al. (2022) MentalBERT | UMD Reddit Suicide | MentalBERT | 0.5826 |
| Yeskuatov et al. review (2022) | Ji et al. Reddit (7k binary) | XGBoost | 0.957 |
| Aguirre et al. (2024) | Spanish/EN tweets (multilingual) | mT5 | 0.881 |
| Pokrywka et al. (2024) | Reddit 4-class competition | GPT-4o fine-tuned | 0.748 |
| **This thesis** | Reddit 232k binary | BERT | **0.9653** |
| **This thesis** | Twitter 1.8k binary | BERT | **0.9468** |
| **This thesis** | C-SSRS Reddit (binary†) | SVM | **0.7270** |
| **This thesis** | Russian VK 64k | XLM-R | **0.9942** |
| **This thesis** | Zero-shot EN→RU | XLM-R | **0.7882** |

† Direct comparison not possible due to different label schemes or dataset sizes.

---

## Immediate Action List

| Priority | Action | Effort | Impact |
|---|---|---|---|
| 1 | Add citations for Narynov et al. and Gaur et al. in thesis | 30 min | Required |
| 2 | Add MentalBERT experiment (one config line + run) | 2h | Directly answers supervisor |
| 3 | Add label noise discussion (limitations section) | 1h | Addresses methodological gap |
| 4 | Add comparison table above to results chapter | 1h | Directly answers supervisor |
| 5 | Clarify binary vs. 5-class C-SSRS in thesis | 30 min | Fixes a real gap |
| 6 | Add Aguirre et al. as related work; explain your zero-shot is harder | 30 min | Strengthens contribution claim |

---

## Full Reference List

1. Narynov, S., Mukhtarkhanuly, D., & Omarov, B. (2020). Dataset of depressive posts in Russian language collected from social media. *Data in Brief*, 29, 105122. https://doi.org/10.1016/j.dib.2020.105122

2. Gaur, M., Alambo, A., Sain, J.P., et al. (2019). Knowledge-aware Assessment of Severity of Suicide Risk for Early Intervention. *Proceedings of WWW 2019*. https://scholarcommons.sc.edu/aii_fac_pub/4/

3. Gaur, M., Aribandi, V., Alambo, A., et al. (2021). Characterization of time-variant and time-invariant assessment of suicidality on Reddit using C-SSRS. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0250448

4. Ji, S., Zhang, T., Ansari, L., et al. (2022). MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare. *Proceedings of LREC 2022*. https://arxiv.org/abs/2110.15621

5. Haque, A., Reddi, V., & Giallanza, T. (2021). Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction. *arXiv:2102.09427*. https://arxiv.org/abs/2102.09427

6. Yeskuatov, E., Chua, S.-L., & Foo, L.K. (2022). Leveraging Reddit for Suicidal Ideation Detection: A Review of Machine Learning and NLP Techniques. *International Journal of Environmental Research and Public Health*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9407719/

7. Aguirre, et al. (2024). The First Multilingual Model For The Detection of Suicide Texts. *Proceedings of SumEval Workshop, COLING 2025*. https://arxiv.org/abs/2412.15498

8. Pokrywka, J., Kaczmarek, J.I., & Gorzelańczyk, E.J. (2024). Evaluating Transformer Models for Suicide Risk Detection on Social Media. *arXiv:2410.08375*. https://arxiv.org/html/2410.08375v1
