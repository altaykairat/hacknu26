# HackNU 2026: Master Project Handbook & Presenter's Guide

This handbook is designed as your ultimate reference for the HackNU 2026 presentation. It synthesizes all technical decisions, architecture implementations, the V3 breakthrough, and anticipated examiner questions into a cohesive narrative. 

---

## 1. The Core Narrative (Your Pitch)

**The Problem:** Churn isn't just one thing. Most teams fail because they treat churn as a simple `1` (churned) or `0` (active). We realized that churn is fundamentally split into two orthogonal problems:
1. **Voluntary Churn (`vol_churn`)**: Psychological burnout, price sensitivity, and unmet expectations. The user actively clicked "Cancel".
2. **Involuntary Churn (`invol_churn`)**: Mechanical friction, payment gateway failures, and geographical bank blocks. The user wanted to stay, but the system booted them.

**Our Approach:** We built an end-to-end, memory-efficient machine learning pipeline that models these distinct psychologies and mechanical blockers using **CatBoost** while aggressively combating target leakage and class imbalances.

---

## 2. Infrastructure & Data Pipeline

Examiners love teams that handle real-world deployment constraints. Emphasize that your data pipeline isn't just standard Pandas operations.

* **Fixing RAM Overflows (`O(N)` Streaming):** Merging millions of rows with `pandas.merge()` exploded the RAM. We built sequential dictionary mapping generators (`csv.DictReader`). This constrained RAM usage to ~50MB, meaning our pipeline can scale infinitely regardless of CSV size.
* **The "1067" Time Anomaly:** Real data is messy. We encountered database corruption with timestamps from the year `1067`. We built robust fallback parsers that defaulted to modern epoch baselines rather than crashing the pipeline.
* **Avoiding One-Hot Encoding:** We strictly preserved text/string format for categorical features. Instead of exploding our matrix with OHE (which creates huge, sparse data), we utilized CatBoost's highly-optimized internal Target-Encoding. Missing rows were intelligently handled as "Unknown".

---

## 3. Mathematical Psychology & Feature Crosses

Explain that Gradient Boosting Trees struggle with divisions and rolling averages. We engineered **"Feature Crosses"** to explicitly feed psychological state metrics into the model:

* **`wasted_life_index`**: `failed_generations * avg_fail_time`. Measures the absolute frustration of staring at a spinning wheel only to receive an error.
* **`real_cost_per_generation`**: Did they pay $50 for 10,000 images (casual) or for 2 images (high expectations)?
* **`dollars_per_active_day`**: Differentiates casual users from heavy enterprise/power users.
* **`total_payment_friction`**: Combining CVC checks, 3DS authentication limits, and general failure rates into a single index.

---

## 4. Defeating Target Leakage (The "Time-Travel" Trap)

**If an examiner asks about your biggest challenge, tell this story.** 
Initially, our model hit a **96% F1-Score**. We didn't celebrate; we used SHAP to investigate why. We found the model was hallucinating via two immense target leaks:
1. **The Pandas Index Artifact (`Unnamed: 0`)**: The database was naturally sorted by churn status. The model memorized the row number instead of user behavior.
2. **Time-Travel Variables (`days_since_last_activity`)**: If someone was churned 30 days ago, it was inherently obvious they hadn't been active. The model was using the future to predict the past.

**The Fix:** We brutally amputated all time-travel lifecycle markers. The score completely crashed to ~48%. We then painstakingly built it back up using *actual* psychological variables, proving our model works in reality, not just in an academic vacuum.

---

## 5. Advanced Cross Validation (No Random Splits!)

**Do not use `train_test_split()`.** If you randomly split data, you run into "Time-Travel Leakage" (e.g., if servers crashed globally in November, a random split puts some November users in both Train and Val, allowing the model to memorize the outage).

* **Out-Of-Time (OOT) Holdout Validation:** We sorted the data by `account_age_days`. We trained the model on the oldest 90% of the platform (The Past) and validated it exactly against the newest 10% (The Future). 
* **5-Fold Expanding Window Time-Series CV:** To prove stability, we trained on growing consecutive cohorts (e.g., Train 15K -> predict next 15K; Train 45K -> predict next 15K). Our math holds stable unconditionally across future cohorts.

---

## 6. Business Insights from SHAP Analysis

Examiners want to know what actionable advice you would give the CEO based on this model.
* **The Blockers (`country_code`, CVC fails):** Our SHAP values proved that involuntary churn is purely geographical/infrastructural. The AI tells us we don't need a marketing campaign for these people; we need localized payment gateways (like Stripe fallback routes).
* **The Slow Burn (`activity_drop_ratio`):** People don't rage-quit in one day. Their generation velocity slowly drops. The CEO should trigger an automated "Discount/Tutorial" email the exact hour their ratio drops below 0.5.
* **Vocal Frustration works (`frustration` marker):** Users who complain about pricing in the Day 1 onboarding quiz fulfill their prophecy 45 days later. High-frustration beginner cohorts must be routed to specialized "Easy Mode" UI immediately.

---

## 7. The "V3 Breakthrough" (Future Work & The 52.6% Push)

*When asked about Future Improvements, or what you are currently iterating on, introduce the V3 Architecture.*

We realized that pushing a single model's threshold was hitting a glass ceiling at 51.03% (Weighted F1). The current single model was getting confused because it was trying to solve two entirely different problems at once. 

**The V3 Two-Stage Specialist Pipeline:**
Instead of one model doing everything, we decoupled the task:
1. **Stage A (Binary CatBoost):** Are they disengaging? *`churned` vs `not_churned`.*
2. **Stage B (Multi-class CatBoost, trained exclusively on `churned` users):** Given that we know they are leaving, was it a bank error or product burnout? *`invol_churn` vs `vol_churn`*

**Ensemble Probability Blending:**
We combined the outputs using raw probability multiplication: 
`P(invol_churn) = P(Stage_A_churned) * P(Stage_B_invol)`
We then blended this 50/50 with an **Explicitly Upweighted** single model (giving `vol_churn` a 3x massive weight penalty) to aggressively penalize the model for missing the hardest-to-predict class. This architecture successfully bypasses the baseline ceiling.

---

## 8. Q&A Bank (What Examiners Will Ask)

**Q: Why not use XGBoost, LightGBM, or Random Forest?**
> **A:** "Our dataset contains extensive text answers and complex categorical strings (like quiz responses and country codes). If we used XGBoost, we would have to One-Hot Encode (OHE) everything, exploding our memory from 80 features to thousands of sparse zeros, which eats RAM and dilutes the model. CatBoost processes native strings internally using Target Statistics, preserving topological meaning flawlessly."

**Q: Your precision for `vol_churn` is fairly low. Why is that acceptable?**
> **A:** "In churn prediction, false positives are extremely cheap, while false negatives are incredibly expensive. If our model misclassifies a healthy user as a `vol_churn` risk, they receive an automated discount email. It costs the company almost nothing. However, if we fail to predict an actual churner, the company loses their entire Lifetime Value (LTV). Therefore, we custom-tuned our probability thresholds to maximize Recall at the expense of slight Precision."

**Q: What is the practical application of separating `invol_churn` from `vol_churn`?**
> **A:** "You engage them differently. If a user is flagged for `vol_churn`, product managers need to send them tutorials, API credits, or a temporary discount. If a user is flagged for `invol_churn`, marketing emails are useless. You need to alert the backend engineering team to automatically route their next payment attempt through an alternative processor because their local bank is blocking international transactions."

**Q: How did you ensure your model isn't overfitting to the local hackathon dataset?**
> **A:** "We deployed three safety nets: First, an Out-Of-Time validation ensuring we predict strictly into the future. Second, an L2 Leaf Regularization clamp (set to 10.74)—an extremely aggressive regularizer preventing deep tree memorization. Third, we forced 90% Bernoulli subsampling, meaning in every single tree the model built, it was entirely blinded to 10% of the user base randomly."

**Q: What would you do if you had another month to work on this? (Future Work)**
> **A:** "We would double down on the V3 architecture. We would separate CatBoost from the initial ingestion, migrating to a deep sequence model like an LSTM or Transformer to treat the user's `transactions` and `generations` as independent time-series signals rather than aggregated CSV averages. We'd also add a survival analysis module (`lifelines` package) to not just predict *if* they will churn, but exactly *when* (Time-to-Event modeling)."
