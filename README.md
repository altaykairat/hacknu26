# 🚀 HackNU 2026: Psychological & Mechanical Churn Prediction

An advanced, end-to-end machine learning pipeline built to predict user churn by decoupling **Voluntary** (behavioral) and **Involuntary** (technical) attrition. This repository represents the evolution from initial data exploration to a state-of-the-art **two-stage ensemble specialist architecture**.

---

## 🧠 Core Philosophy: Churn is not Binary
Most models fail because they treat churn as a simple `1/0`. We realized churn is fundamentally split into two orthogonal psychological and mechanical problems:

1.  **Voluntary Churn (`vol_churn`)**: Psychological burnout, price sensitivity, or unmet quality expectations. The user actively decides to leave.
2.  **Involuntary Churn (`invol_churn`)**: Mechanical friction, payment gateway failures, or geographical blocks. The user wants to stay, but the system fails to bill them.

---

## 🏗️ Repository Structure

```tree
.
├── 📂 v1/                  # Initial Engineering Phase
│   ├── HackNU_Project_Report.md  # Detailed technical deep-dive
│   ├── generate_advanced_features_v3.py
│   └── train_catboost.py   # Baseline model training
├── 📂 v3/                  # Final Solution & Breakthrough
│   ├── train_v11_breakthrough.py # 👑 Final Ensembling Script
│   └── generate_v10_features.py  # Optimized O(N) Feature Ingestion
├── 📂 dataset/             # (Ignored) Raw & Merged CSVs
├── HackNU_Features_Math_Breakdown.md # Technical feature derivations
├── HackNU_Project_Handbook.md        # Presenter's Guide & FAQ
└── requirements.txt        # Dependencies
```

---

## 🎖️ The V3 Breakthrough: Two-Stage Specialist Pipeline
To break through the **51.03% Weighted F1 ceiling**, we moved away from a single multi-class model to a sophisticated **Two-Stage Architecture**:

*   **Stage A (Binary Specialist)**: Focused exclusively on identifying **disengagement** (`churned` vs `not_churned`). Uses high-recall settings to ensure no latent churner is missed.
*   **Stage B (Multi-class Specialist)**: Only triggered for users flagged as `churned`. It focuses on the structural signals to distinguish between `vol_churn` and `invol_churn`.
*   **Probability Blending**: We combined outputs via probability multiplication and blended them with an **Upweighted Single Model** (using 3x penalty for `vol_churn` misses) to maximize performance on minority classes.

---

## 🛠️ Engineering Excellence

### 1. 🛡️ Defeating Target Leakage ("Time-Travel" Prevention)
We identified that standard models achieve 96% F1 scores by "hallucinating" via leakage. We brutally amputated:
*   **The Index Artifact**: `Unnamed: 0` columns that correlated with churn position in the DB.
*   **Time-Travel Variables**: Removed lifecycle markers like `days_since_last_activity` which mechanically inflate *after* a user has already churned.

### 2. ⚡ RAM-Efficient Architecture (`O(N)` Streaming)
Standard `pandas.merge()` exploded RAM when joining millions of rows. Our pipeline uses `csv.DictReader` generators to process data with **linear time complexity**, restricting RAM overhead to ~50MB regardless of dataset size.

### 3. 🧪 Advanced Cross-Validation
We strictly bypassed `train_test_split()` to avoid temporal leakage. 
*   **Out-Of-Time (OOT) Holdout**: Trained on the "Past" (oldest 90% of cohorts) and validated on the "Future" (newest 10%).
*   **5-Fold Expanding Window**: Proved stability by training on growing consecutive cohorts.

---

## 📊 Feature Engineering: "Mathematical Psychology"
We engineered **83 active features** that go beyond raw telemetry:

| Feature | Concept | Business Rationale |
| :--- | :--- | :--- |
| **`activity_drop_ratio`** | Engagement Derivative | The "Death Rattle" — detecting slow burnout before the final "Cancel" click. |
| **`wasted_life_index`** | Frustration Metric | Seconds spent staring at failed spinners. Measures absolute UX torture. |
| **`real_cost_per_gen`** | Expectation Vector | Did they pay $50 for 2 images or 10,000? High cost implies high scrutiny. |
| **`total_payment_friction`** | Mechanical Index | Fuses 3DS hurdles, CVC fails, and billing timeouts into a single blocker signal. |

---

## 🚀 How to Run

1.  **Environment Setup**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Generate Final Features**:
    ```bash
    python v3/generate_v10_features.py
    ```

3.  **Train Final Solution (V3 Breakthrough)**:
    ```bash
    python v3/train_v11_breakthrough.py
    ```

---

## 📈 Results
*   **Baseline (v1)**: ~48% Weighted F1 (Post-leakage removal).
*   **V10 Optimized**: 51.03% Weighted F1.
*   **Final V11 Breakthrough**: **Targeting 52.6%+ Weighted F1** via Stage-decoupling and Ensemble Probability Blending.

> [!TIP]
> Use `v1/HackNU_Project_Report.md` for a deeper dive into the SHAP analysis and business insights derived from the model.
