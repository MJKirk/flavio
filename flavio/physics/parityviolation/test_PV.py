import unittest
import flavio

class TestParityViolation(unittest.TestCase):
    def test_QW(self):
        # SM predictions and errors taken from 2107.13569
        self.assertAlmostEqual(flavio.sm_prediction("Q_W", Z=1, N=0), 0.0710, delta=0.0004)
        self.assertAlmostEqual(flavio.sm_prediction("Q_W", Z=55, N=78), -73.24, delta=0.05)
