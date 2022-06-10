import unittest
import pickle
import numpy as np
from src.calibrator import Calibrator
from pathlib import Path


cwd = Path.cwd()
TEST_DIR = str(cwd/'test')
RTOL = 1e-05
ATOL = 1e-05

class TestCalibrator(unittest.TestCase):

    #### test linear method ####
    def test_linear_01(self):
        X, d, t, g = self.unpack_data('test_linear_01')
        calib = Calibrator(X, d, t, method='linear', lower_bound=None, upper_bound=None)
        calib.calibrate()
        g_exp = calib.g
        self.assertTrue(np.allclose(g, g_exp, rtol=RTOL, atol=ATOL))

    def test_linear_02(self):
        X, d, t, g = self.unpack_data('test_linear_02')
        calib = Calibrator(X, d, t, method='linear', lower_bound=None, upper_bound=None)
        calib.calibrate()
        g_exp = calib.g
        self.assertTrue(np.allclose(g, g_exp, rtol=RTOL, atol=ATOL))

    #### test truncated method ####
    def test_truncated_01(self):
        X, d, t, g = self.unpack_data('test_truncated_01')
        calib = Calibrator(X, d, t, method='truncated', lower_bound=0.75, upper_bound=1.2)
        calib.calibrate()
        g_exp = calib.g
        self.assertTrue(np.allclose(g, g_exp, rtol=RTOL, atol=ATOL))

    def test_truncated_02(self):
        X, d, t, g = self.unpack_data('test_truncated_02')
        calib = Calibrator(X, d, t, method='truncated', lower_bound=0.5, upper_bound=1.5)
        calib.calibrate()
        g_exp = calib.g
        self.assertTrue(np.allclose(g, g_exp, rtol=RTOL, atol=ATOL))

    #### helpers ####
    @staticmethod
    def unpack_data(case_name):
        with open(f"{TEST_DIR}/{case_name}.pkl", 'rb') as handle:
            data = pickle.load(handle)
        X, d, t, g = data.pop('X'), data.pop('d'), data.pop('t'), data.pop('g')
        return X, d, t, g


if __name__ == '__main__':
    unittest.main()
