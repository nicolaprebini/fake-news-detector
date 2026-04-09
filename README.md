# Fake or Fact? — A Machine Learning Approach to Fake News Detection

> **Bachelor's Thesis Project** · University of Salerno · Academic Year 2024–2025
> **Author:** Nicola Prebini · **Supervisors:** Prof.ssa Marta Rinaldi · Prof. Massimiliano De Iuliis

---

## Overview

This repository contains the code and documentation for my Master's thesis in Management Engineering (*Ingegneria Gestionale*) at the University of Salerno. The project develops a **statistical classification system** capable of distinguishing fake news articles from reliable ones, combining **Natural Language Processing (NLP)** and **supervised Machine Learning** techniques.

The core idea: given any news article (title or full text), the model outputs a probability that the content is fake, leveraging patterns in writing style and word usage that systematically differ between misinformation and credible journalism.

---

## Motivation

The global information ecosystem is increasingly polluted by disinformation. Algorithmic amplification on social platforms, filter bubbles, and the plummeting cost of synthetic content production have made manual fact-checking insufficient at scale. This project explores how a lightweight, interpretable ML model can serve as a **first-level automated filter** — not to replace human judgment, but to triage and flag potentially misleading content for further review.

---

## Methodology

### Pipeline Overview

```
Raw Text (title or full article)
        │
        ▼
   Pre-processing
   ─ Tokenization
   ─ Lowercasing
   ─ Stop-word removal
   ─ Stemming
        │
        ▼
  Feature Extraction
   ─ Bag-of-Words (BoW)
   ─ TF-IDF weighting
        │
        ▼
  Classification
   ─ Logistic Regression (L1 / Lasso regularization)
        │
        ▼
  Output: P(Fake) ∈ [0, 1]
```

### Key Design Choices

| Component | Choice | Rationale |
|---|---|---|
| Text representation | TF-IDF (BoW) | Efficient, sparse, proven on text classification tasks |
| Classifier | Logistic Regression + L1 (Lasso) | Interpretable, handles high-dimensional sparse data, built-in feature selection |
| Regularization tuning | 10-Fold Cross-Validation | Robust hyperparameter search, prevents data leakage |
| Implementation | MATLAB 2023b (Text Analytics Toolbox) | Native support for efficient sparse matrix operations on large corpora |

### Why Lasso (L1) Regularization?

A vocabulary of tens of thousands of words creates extreme dimensionality. Lasso regularization forces irrelevant features to exactly zero — effectively performing **automatic feature selection** as part of training. From a vocabulary of ~50,000 terms, the final model retains only the few hundred words that are genuinely predictive, producing a model that is both more generalizable and fully interpretable.

---

## Dataset

**Primary dataset — ISOT Fake News Dataset** (Harvard Dataverse / Kaggle):

| Split | Records | Label |
|---|---|---|
| True.csv | 21,417 | Fact (y=0) |
| Fake.csv | 23,502 | Fake (y=1) |
| **Total** | **44,919** | Balanced (~52% Fake) |

Each record contains: `title`, `text`, `subject`, `date`.

**Important note on EDA:** An exploratory analysis of the `subject` metadata revealed a severe methodological bias — all "Fact" articles were tagged exclusively as `politicsNews` or `worldnews`, while "Fake" articles spanned many diverse categories. Including `subject` as a feature would have inflated results artificially. The `subject` and `date` columns were therefore excluded from all model inputs.

**External validation dataset:** `url-versions-2015-06-14` — a structurally different corpus with short claim-style texts, used to stress-test generalization beyond the training distribution.

---

## Results

### Internal Validation (10-Fold CV on ISOT)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| `Mdl_Title` (headline only) | **95.9%** | 0.964 | 0.957 | 0.961 | **0.9914** |
| `Mdl_Text` (full article) | **98.9%** | 0.988 | 0.992 | 0.990 | **0.9979** |

Both models show stable performance across folds (std < 0.5%), confirming the absence of overfitting.

### External Validation (Out-of-Distribution Test)

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| `Mdl_Title` | ~49% | ~0.55 | ~0.51 |
| `Mdl_Text` | ~52.8% | ~0.63 | ~0.51 |

The performance drop on the external dataset is expected and methodologically informative: it reflects the well-known **domain shift** problem in NLP. The model learns writing-style patterns specific to the ISOT corpus (English political journalism, 2016–2017). Short fact-checking claims with different linguistic distributions expose the limits of the BoW approach and motivate future work on semantic representations.

---

## Interpretability

One of the explicit design goals was **Explainable AI (XAI)**. Because Lasso assigns a weight to every word, the model's decision on any given article can be explained by listing the words that pushed the classification toward "Fake" or "Fact".

Example influential features identified by the model:

| Direction | Example words |
|---|---|
| → Fake | `video`, `breaking`, `boiler`, `racist` |
| → Fact | `factbox`, `says`, `exclusive`, `reuters` |

Stylistic finding from EDA: fake news titles average **14.8 words** vs. **10.0 words** for real news — a ~50% difference suggesting systematic use of longer, more sensationalist headlines in misinformation.

---

## Repository Structure

```
├── scripts/
│   ├── nuovo.m                         # Main entry point
│   ├── process_text.m                  # Pre-processing pipeline
│   ├── local_sort_vocabulary.m         # Vocabulary construction
│   ├── filter_vocabulary_by_frequency.m # Frequency-based filtering
│   ├── create_bow_matrix.m             # Sparse BoW/TF-IDF matrix builder
│   ├── train_model_simple.m            # Training + 10-Fold CV + Lasso tuning
│   └── test_emergent_urlversions.m     # External dataset evaluation
├── data/
│   ├── True.csv                        # ISOT real news (download separately)
│   └── Fake.csv                        # ISOT fake news (download separately)
└── README.md
```

> **Note:** The ISOT dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). Download and place the CSV files in the `data/` folder before running the scripts.

---

## Requirements

- **MATLAB R2023b** or later
- **Text Analytics Toolbox**
- **Statistics and Machine Learning Toolbox**

---

## How to Run

1. Clone the repository and place the ISOT CSV files in `data/`.
2. Open MATLAB and set the repository root as the working directory.
3. Run the main script:
   ```matlab
   run('scripts/nuovo.m')
   ```
4. To evaluate on the external dataset:
   ```matlab
   run('scripts/test_emergent_urlversions.m')
   ```

---

## Future Work

- **Semantic embeddings:** Replace BoW/TF-IDF with contextual representations (Word2Vec, GloVe, BERT) to capture meaning beyond word co-occurrence.
- **Deep learning:** Explore LSTM or Transformer-based classifiers for richer sequential modeling.
- **Multilingual extension:** Adapt the pipeline to Italian-language corpora (e.g., FIEG/Luiss DataLab initiatives).
- **Cross-domain generalization:** Fine-tune on diverse datasets to reduce domain shift.
- **Explainability layer:** Build a user-facing interface that highlights the words driving each prediction.

---

## References

- ISOT Fake News Dataset — University of Victoria
- Wardle, C. et al. — "Information Disorder" (First Draft News)
- Sunstein, C. — *#Republic: Divided Democracy in the Age of Social Media*
- Tukey, J.W. — *Exploratory Data Analysis* (1977)
- Luhn, H.P. — "A Statistical Approach to Mechanized Encoding" (1957)

---

## Author

**Nicola Prebini**
B.Sc. in Management Engineering (*Ingegneria Gestionale*)
University of Salerno · Department of Industrial Engineering
Email: n.prebini@studenti.unisa.it

---

*Thesis supervised by Prof.ssa Marta Rinaldi and Prof. Massimiliano De Iuliis — A.Y. 2024–2025*
