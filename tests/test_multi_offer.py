"""Coupon-amount chooser: decision rule, budget, simulator and policy comparison."""
import sys, unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import multi_offer as mo  # noqa: E402
from simulate_multi_offer import simulate  # noqa: E402


class DecisionRuleTests(unittest.TestCase):
    def setUp(self):
        # three customers: one responds strongly, one barely, one would order anyway
        self.p = pd.DataFrame({"none": [0.02, 0.10, 0.50], "$3 off": [0.20, 0.11, 0.52],
                               "$5 off": [0.24, 0.12, 0.53], "$8 off": [0.26, 0.12, 0.54]})

    def test_picks_highest_net_value(self):
        nv = mo.net_values(self.p, 12)
        ch = mo.choose_offers(self.p, 12)
        self.assertEqual(ch.recommended_offer[0], nv.loc[0].idxmax())

    def test_no_coupon_when_it_loses_money(self):
        ch = mo.choose_offers(self.p, 12)
        self.assertEqual(ch.recommended_offer[1], "none")
        self.assertEqual(ch.recommended_offer[2], "none")

    def test_budget_is_respected(self):
        ch = mo.choose_offers(self.p, 12, budget=0.5)
        self.assertLessEqual(ch.expected_coupon_cost.sum(), 0.5 + 1e-9)
        self.assertTrue((mo.choose_offers(self.p, 12, budget=0).recommended_offer == "none").all())


class SimulationAndPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res, cls.mix, cls.summary, _ = mo.run(12.0)
        cls.v = cls.res.set_index("policy").net_value

    def test_simulator_is_reproducible_and_balanced(self):
        a, b = simulate(2000, seed=3), simulate(2000, seed=3)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(set(a.offer), set(mo.ARMS))

    def test_ceiling_beats_every_policy(self):
        for p, val in self.v.items():
            self.assertLessEqual(val, self.v["Perfect knowledge (ceiling)"] + 1e-6, p)

    def test_personalized_beats_every_blanket_offer(self):
        for p in ["Everyone $3", "Everyone $5", "Everyone $8"]:
            self.assertGreater(self.v["Personalized (model)"], self.v[p], p)

    def test_active_customers_mostly_get_no_coupon(self):
        self.assertGreater(self.mix.loc["Active", "none"], 0.6)


if __name__ == "__main__":
    unittest.main()
