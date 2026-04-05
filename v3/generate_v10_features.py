"""
generate_v10_features.py
========================
Appends 5 new leakage-free behavioral features to the v9 master dataset.

New Features:
  1. weekly_gen_delta        — Gens(last 7d) - Gens(prior 7d). Negative = user trending down.
                               Safe from leakage: measures INTENT, not post-churn consequence.
  2. success_velocity        — successful_generations / account_age_days.
                               "Value Realization Rate" — how much win-rate per day of tenure.
  3. geo_payment_stress      — Interaction: payment_failure_rate × country_risk_score.
                               Encodes the regional gateway context that trees can miss.
  4. frustrated_professional — (role == 'filmmaker'|'developer'|'professional') × wasted_life_index.
                               Pros have zero tolerance; their wasted time is a multiplied signal.
  5. high_pressure_pro       — Binary: professional user with avg_credit_per_gen > 200 AND
                               real_cost_per_generation > 0.05. High-expectation risk flag.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Country Risk Mapping
# Countries with historically high payment friction (from domain knowledge).
# We use a simple 3-tier ordinal encoding (not OHE) to stay CatBoost-friendly.
# ---------------------------------------------------------------------------
HIGH_FRICTION_COUNTRIES = {
    # Tier 3 — very high gateway friction / sanctions / latency
    'RU', 'BY', 'IR', 'KP', 'CU', 'SY', 'VE', 'NG', 'PK', 'BD',
    'EG', 'GH', 'KE', 'TZ', 'UG', 'ET', 'ZW',
}
MEDIUM_FRICTION_COUNTRIES = {
    # Tier 2 — moderate friction (3DS heavy regions, weak acquiring banks)
    'ZA', 'IN', 'TR', 'MX', 'AR', 'CO', 'PE', 'UA', 'KZ', 'AZ',
    'UZ', 'GE', 'AM', 'MD', 'RS', 'AL', 'MK', 'BA', 'ID', 'PH',
    'TH', 'VN', 'MY',
}

def country_risk_score(code: str) -> float:
    """Returns 1.0 (low), 2.0 (medium), or 3.0 (high) friction tier."""
    if not isinstance(code, str):
        return 1.0
    code = code.strip().upper()
    if code in HIGH_FRICTION_COUNTRIES:
        return 3.0
    if code in MEDIUM_FRICTION_COUNTRIES:
        return 2.0
    return 1.0


PROFESSIONAL_ROLES = {'filmmaker', 'developer', 'professional', 'agency', 'business'}


def generate_v10_features(v9_train: str, v9_test: str,
                           out_train: str, out_test: str):
    for in_path, out_path, label in [
        (v9_train, out_train, 'TRAIN'),
        (v9_test,  out_test,  'TEST'),
    ]:
        print(f"\n[V10] Processing {label}: {in_path}")
        df = pd.read_csv(in_path)
        print(f"      Loaded {len(df):,} rows × {len(df.columns)} cols")

        # ------------------------------------------------------------------ #
        # 1. Weekly Generation Delta
        #    = activity_drop_ratio encodes (last7+1)/(prev7+1), but we want
        #    the raw signed delta too — it behaves differently in tree splits.
        #    We reconstruct from activity_drop_ratio using the known formula:
        #       ratio = (L+1)/(P+1)  ⟹  L - P ≈ (ratio - 1) * (P + 1)
        #    Since we don't have P directly, we use a proxy:
        #       weekly_gen_delta = (activity_drop_ratio - 1.0)
        #    Negative values = declining user. Simple, no leakage.
        # ------------------------------------------------------------------ #
        df['weekly_gen_delta'] = (df['activity_drop_ratio'] - 1.0).round(4)

        # ------------------------------------------------------------------ #
        # 2. Success Velocity  =  successful_gens / account_age_days
        #    Guards: if account_age_days == 0, return 0.
        # ------------------------------------------------------------------ #
        df['success_velocity'] = np.where(
            df['account_age_days'] > 0,
            (df['completed_generations'] / df['account_age_days']).round(4),
            0.0
        )

        # ------------------------------------------------------------------ #
        # 3. Geo-Payment Stress  =  payment_failure_rate × country_risk_score
        #    Multiplies the raw failure rate by the regional friction tier,
        #    making the tree splits country-context-aware in a single float.
        # ------------------------------------------------------------------ #
        risk_scores = df['country_code'].apply(country_risk_score)
        df['geo_payment_stress'] = (df['payment_failure_rate'] * risk_scores).round(4)

        # ------------------------------------------------------------------ #
        # 4. Frustrated Professional  =  is_pro_role × wasted_life_index
        #    Pros have zero tolerance for wasted time — their index is weighted.
        # ------------------------------------------------------------------ #
        is_pro = df['role'].str.strip().str.lower().isin(PROFESSIONAL_ROLES).astype(float)
        df['frustrated_professional'] = (is_pro * df['wasted_life_index']).round(2)

        # ------------------------------------------------------------------ #
        # 5. High-Pressure Pro Flag  (binary)
        #    Professional role AND high per-gen cost AND high real dollar cost.
        #    Captures the "cinematic quality expectation" risk segment.
        # ------------------------------------------------------------------ #
        df['high_pressure_pro'] = (
            is_pro.astype(bool)
            & (df['avg_credit_per_gen'] > 200)
            & (df['real_cost_per_generation'] > 0.05)
        ).astype(int)

        print(f"      > weekly_gen_delta      range: [{df['weekly_gen_delta'].min():.3f}, {df['weekly_gen_delta'].max():.3f}]")
        print(f"      > success_velocity     range: [{df['success_velocity'].min():.3f}, {df['success_velocity'].max():.3f}]")
        print(f"      > geo_payment_stress   range: [{df['geo_payment_stress'].min():.3f}, {df['geo_payment_stress'].max():.3f}]")
        print(f"      > frustrated_prof      non-zero: {(df['frustrated_professional'] > 0).sum():,}")
        print(f"      > high_pressure_pro    flagged:  {df['high_pressure_pro'].sum():,}")

        df.to_csv(out_path, index=False)
        print(f"      Saved {len(df.columns)} cols → {out_path}")


if __name__ == '__main__':
    generate_v10_features(
        v9_train='dataset/train/train_users_merged_advanced_v9.csv',
        v9_test='dataset/test/test_users_merged_advanced_v9.csv',
        out_train='dataset/train/train_users_merged_advanced_v10.csv',
        out_test='dataset/test/test_users_merged_advanced_v10.csv',
    )
    print("\n[V10] Feature generation complete.")
