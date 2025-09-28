# Bias-Aware, Explainable AI for Fair Recruitment

This project contains a self-contained pipeline that demonstrates building a bias-aware recruitment model,
measuring fairness, applying simple mitigation strategies, and producing a small static dashboard of results.

**What is included:**
- `data/synthetic_candidates.csv`: synthetic dataset with features, target (hire), and sensitive attributes.
- `src/train_and_evaluate.py`: main script that trains models, computes fairness metrics, saves plots and results.
- `src/utils.py`: helper functions for fairness metrics and mitigation strategies.
- `outputs/`: contains generated plots, CSVs, and model artifacts after running the pipeline.
- `requirements.txt`: minimal packages required (numpy, pandas, scikit-learn, matplotlib, joblib)
- `run.sh`: convenience script to run the pipeline.

**Notes & limitations:**
- This project avoids optional libraries that may not be available offline (e.g., `shap`, `lime`, `aif360`). Instead,
  it provides simple, explainable outputs: logistic coefficients, feature importances from RandomForest, and local
  class-prototype explanations.
- To extend with `SHAP`/`LIME` or a dashboard (Streamlit/Flask), install the extra packages and follow instructions in the
  README's Extensions section.

Run the pipeline:
```bash
bash run.sh
```

Outputs will be saved in `outputs/`. The pipeline is designed to run offline with common Python scientific packages.