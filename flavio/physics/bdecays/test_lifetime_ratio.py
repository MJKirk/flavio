import unittest
import flavio


par = flavio.default_parameters.get_central_all()


class TestTauBpoBd(unittest.TestCase):
    def test_sm(self):
        self.assertAlmostEqual(flavio.sm_prediction('tau_B+/tau_Bd'), 1.08, delta=0.05)
