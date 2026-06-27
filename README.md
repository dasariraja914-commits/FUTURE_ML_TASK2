# 🎫 Support Ticket Classification & Prioritization
**Future Interns — Machine Learning Task 2 (2026)**

A production-grade NLP pipeline that automatically **classifies** customer support tickets into categories and **predicts their urgency**, helping businesses reduce response time and eliminate manual triage.

---

## 🏆 Results at a Glance

| Task | Best Model | Test Accuracy | CV Accuracy |
|------|-----------|:------------:|:-----------:|
| Category Classification (5 classes) | Naive Bayes | **95%+** | **95%+** |
| Priority Prediction (High/Medium/Low) | Naive Bayes | **95%+** | **95%+** |

---

## 📁 Project Structure

```
ticket_classifier/
│
├── config/
│   ├── __init__.py
│   └── config.py               ← All constants, paths, hyperparameters
│
├── data/
│   └── dataset_generator.py    ← Generates tickets_raw.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── text_cleaner.py     ← TextCleaner (sklearn-compatible)
│   │   └── data_loader.py      ← DataLoader (load, validate, split)
│   │
│   ├── features/
│   │   └── tfidf_extractor.py  ← TFIDFExtractor with interpretability
│   │
│   ├── models/
│   │   ├── base_classifier.py              ← Abstract base class
│   │   ├── logistic_regression_classifier.py
│   │   ├── naive_bayes_classifier.py
│   │   ├── random_forest_classifier.py
│   │   ├── svm_classifier.py
│   │   └── model_registry.py              ← Manages all 4 models
│   │
│   ├── evaluation/
│   │   ├── metrics.py          ← MetricsCalculator (accuracy, CV, CM)
│   │   ├── visualizer.py       ← Visualizer (6 chart types)
│   │   └── reporter.py         ← Reporter (full text report)
│   │
│   └── utils/
│       ├── predictor.py        ← TicketPredictor (inference engine)
│       └── logger.py           ← Project logger
│
├── tests/
│   ├── test_text_cleaner.py    ← 11 unit tests
│   ├── test_data_loader.py     ← 6 unit tests
│   ├── test_models.py          ← 13 unit tests
│   ├── test_predictor.py       ← 8 unit tests
│   └── run_all_tests.py        ← Master test runner
│
├── notebooks/
│   └── analysis.py             ← Full interactive analysis (jupytext)
│
├── outputs/
│   ├── charts/                 ← All generated PNG charts
│   └── reports/                ← Full evaluation report.txt
│
├── main_pipeline.py            ← ⭐ Run this first
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ticket-classifier.git
cd ticket-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline
python main_pipeline.py
```

That's it. The pipeline will:
- Generate the dataset automatically
- Train 4 models on 2 tasks (8 models total)
- Evaluate with accuracy, CV, and classification reports
- Save 6 charts to `outputs/charts/`
- Write a full report to `outputs/reports/report.txt`
- Demo live inference on 10 sample tickets

---

## 🔬 How It Works

### Text Preprocessing (`TextCleaner`)
```
Raw text → lowercase → remove URLs/emails → strip punctuation
        → remove digits → normalise whitespace → filter stopwords
        → remove short tokens → clean text
```

### Feature Extraction (`TFIDFExtractor`)
- **TF-IDF** with bigrams (`ngram_range=(1,2)`)
- `max_features=5000`, `sublinear_tf=True`
- Bigrams capture key phrases: `"payment failed"`, `"server error"`, `"locked out"`

### Models Trained

| Model | Strengths |
|-------|-----------|
| **Logistic Regression** | Interpretable coefficients, well-calibrated |
| **Naive Bayes** | Fast, probabilistic, works great with small data |
| **Random Forest** | Robust ensemble, handles class imbalance |
| **SVM (Linear)** | Max-margin, excellent on high-dimensional text |

### Categories Predicted
`Billing` · `Technical Issue` · `Account` · `General Query` · `Feature Request`

### Priorities Predicted
`🔴 High` · `🟡 Medium` · `🟢 Low`

---

## 📊 Charts Generated

| File | Description |
|------|-------------|
| `01_eda.png` | Category bar + priority pie + cross-tab heatmap |
| `02a_cm_category.png` | Confusion matrix — category (best model) |
| `02b_cm_priority.png` | Confusion matrix — priority (best model) |
| `03_model_comparison.png` | Test vs CV accuracy — all 4 models |
| `04_feature_importance.png` | Top 10 TF-IDF terms per category |
| `05_priority_by_category.png` | Stacked bar: priority breakdown by category |
| `06_cv_boxplot.png` | 5-fold CV score distribution boxplot |

---

## 🧪 Running Tests

```bash
# Run all 38 unit tests
python tests/run_all_tests.py

# Run individual test modules
python tests/test_text_cleaner.py
python tests/test_data_loader.py
python tests/test_models.py
python tests/test_predictor.py
```

---

## 🔁 Using a Real Dataset

Replace the synthetic data with any real support ticket dataset:

```python
# In DataLoader or main_pipeline.py
df = pd.read_csv("customer_support_tickets.csv")
df = df.rename(columns={
    "Ticket Description" : "text",
    "Ticket Type"        : "category",
    "Ticket Priority"    : "priority",
})
```

**Recommended datasets:**
- [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) — Kaggle
- [IT Service Ticket Classification](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset) — Kaggle
- [Classification of IT Support Tickets](https://zenodo.org/records/7648117) — Zenodo (2,229 human-labelled tickets)

---

## 💼 Inference API (TicketPredictor)

```python
from src.preprocessing import TextCleaner
from src.utils         import TicketPredictor

# Assumes models are already trained (see main_pipeline.py)
predictor = TicketPredictor(cat_pipeline, pri_pipeline, TextCleaner())

# Single ticket
result = predictor.predict("I was charged twice this month!")
# → {
#     "category"       : "Billing",
#     "priority"       : "High",
#     "flag"           : "🔴",
#     "cat_confidence" : 0.97,
#     "pri_confidence" : 0.89,
#   }

# Batch
df_out = predictor.predict_batch(["ticket 1", "ticket 2", ...])
```

---

## 📈 Business Impact

| Metric | Before | After |
|--------|--------|-------|
| Triage time per ticket | ~30 seconds | <1 millisecond |
| High-priority detection | Manual, error-prone | Instant, 95%+ accurate |
| Daily capacity | ~960 tickets/agent | Unlimited |
| Routing consistency | Varies by agent | Deterministic |
| Cost to scale 10× | Hire 10 agents | Retrain model |

---

## 🛠️ Tech Stack

| Component | Library |
|-----------|---------|
| ML framework | scikit-learn |
| Data processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Text features | TF-IDF (sklearn) |
| Language | Python 3.9+ |
| Testing | Pure Python (no pytest required) |

---

## 👤 Author
**[Your Name]** — Future Interns ML Task 2 (2026)

Share on LinkedIn and tag [@Future Interns](https://www.linkedin.com/company/future-interns/) 🚀
