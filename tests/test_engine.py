"""Decision-engine behaviour: validation, eligibility, and budget constraint."""
import sys, tempfile, unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from growth_engine import recommend_customers, score_customers  # noqa: E402
from growth_model import fit_two_models, load_experiment, split_arms  # noqa: E402


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        parts = split_arms(load_experiment())
        pre, ctrl, trt = fit_two_models(parts["Control"][0], parts["Treatment"][0])
        for name, obj in [("preprocessor", pre), ("control_model", ctrl), ("treatment_model", trt)]:
            joblib.dump(obj, Path(cls.tmp.name) / f"{name}.joblib")
        cls.batch = pd.read_csv(ROOT / "data" / "sample_customer_input.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def scored(self):
        return score_customers(self.batch, model_dir=self.tmp.name)

    def test_scores_are_probabilities(self):
        s = self.scored()
        for c in ["predicted_control_probability", "predicted_treatment_probability"]:
            self.assertTrue(s[c].between(0, 1).all())
        self.assertEqual(len(s), len(self.batch))

    def test_rejects_duplicate_ids(self):
        bad = pd.concat([self.batch, self.batch.head(1)])
        with self.assertRaises(ValueError):
            score_customers(bad, model_dir=self.tmp.name)

    def test_rejects_missing_feature(self):
        with self.assertRaises(ValueError):
            score_customers(self.batch.drop(columns=["orders_90d"]), model_dir=self.tmp.name)

    def test_budget_is_never_exceeded(self):
        for budget in (0, 5, 25, 100):
            r = recommend_customers(self.scored(), budget=budget)
            self.assertLessEqual(r.budget_used.sum(), budget + 1e-9)

    def test_only_eligible_positive_value_customers_selected(self):
        r = recommend_customers(self.scored(), budget=1000)
        sel = r[r.recommended_action == "$5 off $30"]
        self.assertTrue(sel.offer_eligible.all())
        self.assertTrue((sel.estimated_incremental_response > 0).all())
        self.assertTrue((sel.illustrative_net_value_after_offer_cost > 0).all())

    def test_zero_budget_selects_nobody(self):
        r = recommend_customers(self.scored(), budget=0)
        self.assertEqual((r.recommended_action == "$5 off $30").sum(), 0)

    def test_custom_offer_amount(self):
        r = recommend_customers(self.scored(), budget=50, offer_amount=3, minimum_spend=20)
        sel = r[r.recommended_action == "$3 off $20"]
        self.assertGreater(len(sel), 0)
        self.assertLessEqual(r.budget_used.sum(), 50 + 1e-9)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            recommend_customers(self.scored(), budget=-1)


if __name__ == "__main__":
    unittest.main()
