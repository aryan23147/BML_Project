import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from phase1_toy import run_test_a, run_test_b, run_test_c


class Phase1ToyAuditTest(unittest.TestCase):
    def test_a_central_difference_matches_quadratic_truth(self):
        res_a = run_test_a(seed=42)
        self.assertLess(res_a["max_err_central_at_opt"], 1e-11)
        self.assertLess(res_a["max_err_central_off_opt"], 1e-11)
        # Off-optimum one-sided difference should strongly disagree due to first-order gradient term
        self.assertGreater(res_a["max_err_onesided_off_opt"], 1.0)
        self.assertLess(res_a["max_err_onesided_linear_term_check"], 1e-11)

    def test_b_coordinate_vs_eigendirections(self):
        res_b = run_test_b()
        ud = res_b["unit_damages"]
        self.assertAlmostEqual(ud["eig_plus_(1,1)/sqrt2"]["measured_damage"], 9.95, places=10)
        self.assertAlmostEqual(ud["coord_w1_(1,0)"]["measured_damage"], 5.00, places=10)
        self.assertAlmostEqual(ud["coord_w2_(0,1)"]["measured_damage"], 5.00, places=10)
        self.assertAlmostEqual(ud["eig_minus_(1,-1)/sqrt2"]["measured_damage"], 0.05, places=10)
        self.assertAlmostEqual(res_b["anisotropy_ratio_eig_plus_over_minus"], 199.0, places=8)

    def test_c_scores_and_rho_sweep(self):
        res_c = run_test_c(seed=42)
        p2 = res_c["part2_ranking_20d"]
        self.assertAlmostEqual(p2["spearman_single_mf_inv_var"], 1.0, places=10)
        self.assertAlmostEqual(p2["spearman_single_mf_inv_std"], 1.0, places=10)
        self.assertAlmostEqual(p2["spearman_single_fisher_diag"], 1.0, places=10)
        self.assertLess(p2["spearman_single_exact_marginal_inv_var"], 0.9)

        p3 = res_c["part3_rho_sweep"]
        self.assertAlmostEqual(p3["ratio_at_r_0"], 1.0, places=12)
        self.assertAlmostEqual(p3["ratio_at_r_099"], 1.0 / (1.0 - 0.99 ** 2), places=10)


if __name__ == "__main__":
    unittest.main()

