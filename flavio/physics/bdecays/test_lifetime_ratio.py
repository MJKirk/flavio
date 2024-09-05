import unittest
import flavio
import flavio.physics


par = flavio.default_parameters.get_central_all()

wc_sm = flavio.WilsonCoefficients()

class TestTauBpoBd(unittest.TestCase):
    def test_sm(self):
        self.assertAlmostEqual(flavio.sm_prediction('tau_B+/tau_Bd'), 1.08, delta=0.01)

    def test_WE_cu(self):
        self.assertEqual(flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc_sm, par, "B0"), 0)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15*flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc, par, "B0"), 3.03734, places=5)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e15*flavio.physics.bdecays.lifetime_ratio.weak_exchange(wc, par, "B0"), 9.79347, places=5)

    def test_PI_cd(self):
        self.assertEqual(flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc_sm, par, "B+"), 0)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLL_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e14*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 2.82494, places=5)

        wc = flavio.WilsonCoefficients()
        wc.set_initial({"CVLLt_bcud": 1}, scale=4.5)
        self.assertAlmostEqual(1e13*flavio.physics.bdecays.lifetime_ratio.pauli_interference(wc, par, "B+"), 1.38771, places=5)