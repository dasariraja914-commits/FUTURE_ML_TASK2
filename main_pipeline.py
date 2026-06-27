"""
main_pipeline.py
=================
Full end-to-end ML pipeline orchestrator.

Run this file to:
  1. Generate (or load) the dataset
  2. Preprocess and split the data
  3. Train all four models for both tasks
  4. Evaluate with metrics + cross-validation
  5. Generate all charts
  6. Write a full text report
  7. Demo the inference engine on sample tickets

Usage
-----
    python main_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.config    import RAW_DATA_PATH, CHARTS_DIR, REPORTS_DIR
from src.preprocessing import DataLoader, TextCleaner
from src.features      import TFIDFExtractor
from src.models        import ModelRegistry
from src.evaluation    import MetricsCalculator, Visualizer, Reporter
from src.utils         import TicketPredictor, get_logger

log = get_logger("pipeline")


# ══════════════════════════════════════════════════════════
# STEP 0 — GENERATE DATASET IF NEEDED
# ══════════════════════════════════════════════════════════
def ensure_dataset():
    if not RAW_DATA_PATH.exists():
        log.info("Dataset not found. Generating...")
        import subprocess, sys as _sys
        subprocess.run(
            [_sys.executable, str(ROOT / "data" / "dataset_generator.py")],
            check=True,
        )
    else:
        log.info(f"Dataset found: {RAW_DATA_PATH.name}")


# ══════════════════════════════════════════════════════════
# STEP 1 — LOAD & SPLIT DATA
# ══════════════════════════════════════════════════════════
def load_data():
    log.info("Loading data...")
    loader = DataLoader()
    df     = loader.load(use_cache=False)
    DataLoader.summary(df)
    X_train, X_test, yc_train, yc_test, yp_train, yp_test = loader.split(df)
    return df, X_train, X_test, yc_train, yc_test, yp_train, yp_test


# ══════════════════════════════════════════════════════════
# STEP 2 — TRAIN ALL MODELS
# ══════════════════════════════════════════════════════════
def train_models(X_train, yc_train, yp_train):
    log.info("Training category classifiers...")
    cat_registry = ModelRegistry(task="category")
    cat_registry.train_all(X_train, yc_train)

    log.info("Training priority classifiers...")
    pri_registry = ModelRegistry(task="priority")
    pri_registry.train_all(X_train, yp_train)

    return cat_registry, pri_registry


# ══════════════════════════════════════════════════════════
# STEP 3 — EVALUATE ALL MODELS
# ══════════════════════════════════════════════════════════
def evaluate_models(
    cat_registry, pri_registry,
    X_train, X_test,
    yc_train, yc_test,
    yp_train, yp_test,
    X_all, y_cat_all, y_pri_all,
):
    log.info("Evaluating models...")
    cat_evals = {}
    pri_evals = {}

    print("\n── Category Classification ──────────────────────────")
    for name, model in cat_registry.models.items():
        calc = MetricsCalculator(name)
        r    = calc.compute(
            model.pipeline_, X_train, yc_train, X_test, yc_test,
            X_all, y_cat_all,
        )
        cat_evals[name] = r
        print(calc.summary_line())

    print("\n── Priority Prediction ──────────────────────────────")
    for name, model in pri_registry.models.items():
        calc = MetricsCalculator(name)
        r    = calc.compute(
            model.pipeline_, X_train, yp_train, X_test, yp_test,
            X_all, y_pri_all,
        )
        pri_evals[name] = r
        print(calc.summary_line())

    return cat_evals, pri_evals


# ══════════════════════════════════════════════════════════
# STEP 4 — GENERATE CHARTS
# ══════════════════════════════════════════════════════════
def generate_charts(df, cat_evals, pri_evals, cat_registry):
    log.info("Generating charts...")
    viz = Visualizer()

    viz.plot_eda(df)
    viz.plot_priority_by_category(df)

    # Confusion matrices — best model each task
    best_cat_name = max(cat_evals, key=lambda k: cat_evals[k]["accuracy"])
    best_pri_name = max(pri_evals, key=lambda k: pri_evals[k]["accuracy"])

    viz.plot_confusion_matrix(
        cat_evals[best_cat_name]["conf_matrix"],
        cat_evals[best_cat_name]["labels"],
        best_cat_name, "Category Classification",
        cat_evals[best_cat_name]["accuracy"],
        "02a_cm_category.png",
    )
    viz.plot_confusion_matrix(
        pri_evals[best_pri_name]["conf_matrix"],
        pri_evals[best_pri_name]["labels"],
        best_pri_name, "Priority Prediction",
        pri_evals[best_pri_name]["accuracy"],
        "02b_cm_priority.png",
    )

    viz.plot_model_comparison(cat_evals, pri_evals)
    viz.plot_cv_boxplot(cat_evals, pri_evals)

    # Feature importance from best interpretable model
    extractor = TFIDFExtractor()
    # use the LR pipeline's fitted vectorizer
    lr_pipeline = cat_registry["Logistic Regression"].pipeline_
    lr_clf      = lr_pipeline["clf"]
    lr_vec      = lr_pipeline["tfidf"]

    class _FakeExtractor:
        @property
        def feature_names(self):
            return lr_vec.get_feature_names_out()

    import numpy as np
    fake_ext = _FakeExtractor()
    terms_out = lr_vec.get_feature_names_out()
    top_terms = {}
    for cls, coef in zip(lr_clf.classes_, lr_clf.coef_):
        top_idx = np.argsort(coef)[-10:][::-1]
        top_terms[cls] = list(terms_out[top_idx])

    viz.plot_feature_importance(top_terms, "Logistic Regression")
    log.info("All charts saved.")


# ══════════════════════════════════════════════════════════
# STEP 5 — WRITE REPORT
# ══════════════════════════════════════════════════════════
def write_report(df, cat_evals, pri_evals, train_size, test_size):
    log.info("Writing report...")
    reporter = Reporter()
    dataset_info = {
        "total"      : len(df),
        "train"      : train_size,
        "test"       : test_size,
        "categories" : df["category"].value_counts().to_dict(),
        "priorities" : df["priority"].value_counts().to_dict(),
    }
    reporter.build(cat_evals, pri_evals, dataset_info)
    reporter.save()
    reporter.print_summary(cat_evals, pri_evals)


# ══════════════════════════════════════════════════════════
# STEP 6 — DEMO INFERENCE
# ══════════════════════════════════════════════════════════
DEMO_TICKETS = [
    "I was charged twice for my subscription and need a refund immediately.",
    "The application crashes every time I try to log in, please fix this urgently.",
    "Can you please add a dark mode option to the web application?",
    "How do I reset my password when I have forgotten it?",
    "Our entire team cannot access the platform since this morning.",
    "I need an invoice copy for my company expense report.",
    "API is returning 500 errors for all POST requests since last night.",
    "What is the difference between the Pro and Enterprise pricing plan?",
    "My account was suspended without any prior notification or warning.",
    "Please integrate push notifications with Microsoft Teams.",
]

def run_inference_demo(cat_registry, pri_registry):
    log.info("Running inference demo...")
    best_cat_name = max(cat_registry._scores, key=cat_registry._scores.__getitem__) if cat_registry._scores else "Naive Bayes"
    best_pri_name = max(pri_registry._scores, key=pri_registry._scores.__getitem__) if pri_registry._scores else "Naive Bayes"

    cat_pipe = cat_registry[best_cat_name].pipeline_
    pri_pipe = pri_registry[best_pri_name].pipeline_
    cleaner  = TextCleaner()

    predictor = TicketPredictor(cat_pipe, pri_pipe, cleaner)
    predictor.print_predictions(DEMO_TICKETS)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("\n" + "═" * 62)
    print("  SUPPORT TICKET CLASSIFIER — Full Pipeline")
    print("  Future Interns ML Task 2 (2026)")
    print("═" * 62 + "\n")

    ensure_dataset()

    (df, X_train, X_test,
     yc_train, yc_test,
     yp_train, yp_test) = load_data()

    X_all      = pd.concat([X_train, X_test])
    y_cat_all  = pd.concat([yc_train, yc_test])
    y_pri_all  = pd.concat([yp_train, yp_test])

    cat_registry, pri_registry = train_models(X_train, yc_train, yp_train)

    cat_evals, pri_evals = evaluate_models(
        cat_registry, pri_registry,
        X_train, X_test,
        yc_train, yc_test,
        yp_train, yp_test,
        X_all, y_cat_all, y_pri_all,
    )

    generate_charts(df, cat_evals, pri_evals, cat_registry)
    write_report(df, cat_evals, pri_evals, len(X_train), len(X_test))
    run_inference_demo(cat_registry, pri_registry)

    log.info("✅ Pipeline complete. Check outputs/ for charts and report.")


if __name__ == "__main__":
    main()
