"""The headline numbers must match across Python, SQL and the documented README values."""
import sqlite3, sys, unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import experiment_results  # noqa: E402
from growth_model import load_experiment  # noqa: E402


def run_sql(conn, name):
    return pd.read_sql_query((ROOT / "sql" / name).read_text(), conn)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_experiment()
        cls.res, cls.seg = experiment_results(cls.df)
        cls.conn = sqlite3.connect(":memory:")
        cls.df[["customer_id", "segment", "treatment_group", "converted", "post_intervention_value"]].to_sql("experiment", cls.conn, index=False)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_randomization_and_headline(self):
        self.assertEqual((self.res["n_control"], self.res["n_treatment"]), (7409, 7591))
        self.assertAlmostEqual(self.res["control_rate"], 0.0553, places=4)
        self.assertAlmostEqual(self.res["treatment_rate"], 0.1255, places=4)
        self.assertAlmostEqual(self.res["lift_pp"], 7.02, places=2)

    def test_confidence_interval_excludes_zero(self):
        lo, hi = self.res["ci95_pp"]
        self.assertGreater(lo, 0)
        self.assertLess(lo, self.res["lift_pp"])
        self.assertGreater(hi, self.res["lift_pp"])

    def test_sql_overall_matches_python(self):
        q = run_sql(self.conn, "02_experiment_results.sql").set_index("treatment_group")
        self.assertAlmostEqual(q.loc["Control", "conversion_rate_pct"], self.res["control_rate"] * 100, places=2)
        self.assertAlmostEqual(q.loc["Treatment", "conversion_rate_pct"], self.res["treatment_rate"] * 100, places=2)

    def test_sql_segments_match_python(self):
        q = run_sql(self.conn, "03_segment_lift.sql").set_index("segment")
        py = self.seg.set_index("segment")
        for s in py.index:
            self.assertAlmostEqual(q.loc[s, "lift_pp"], py.loc[s, "lift_pp"], places=2, msg=s)
            self.assertAlmostEqual(q.loc[s, "net_value"], py.loc[s, "net_value"], places=1, msg=s)
            self.assertAlmostEqual(q.loc[s, "wasted_share_pct"], py.loc[s, "wasted_share"] * 100, places=1, msg=s)

    def test_at_risk_has_highest_lift(self):
        self.assertEqual(self.seg.sort_values("lift_pp").iloc[-1]["segment"], "At Risk")


if __name__ == "__main__":
    unittest.main()
