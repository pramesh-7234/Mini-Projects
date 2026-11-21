"""Run from project root: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
from main import CableParameters, build_operator, simulate, infinite_cable_reference, calculate_validation

class TestCable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = CableParameters(n_points=101, time_samples=201)
        cls.sol = simulate(cls.p)

    def test_grid_and_operator(self):
        p = self.p
        self.assertAlmostEqual(self.sol.x_um[-1], p.length_um)
        self.assertAlmostEqual(self.sol.x_um[1] - self.sol.x_um[0], p.dx_um)
        op = build_operator(p)
        self.assertEqual(op.shape, (p.n_points, p.n_points))
        self.assertTrue(np.allclose(op.getrow(0).toarray(), 0))
        self.assertTrue(np.allclose(op.getrow(-1).toarray(), 0))

    def test_finite_nonnegative_clamped(self):
        v = self.sol.voltage_mV
        self.assertTrue(np.isfinite(v).all())
        self.assertGreater(v.max(), 0)
        self.assertGreaterEqual(v.min(), -1e-9)
        self.assertTrue(np.allclose(v[[0, -1], :], 0, atol=1e-11))
        self.assertTrue(np.allclose(v[:, self.sol.t_ms < self.p.pulse_start_ms], 0, atol=1e-11))

    def test_zero_source(self):
        p = CableParameters(n_points=51, time_samples=51, input_mV=0)
        v = simulate(p).voltage_mV
        self.assertTrue(np.array_equal(v, np.zeros_like(v)))

    def test_green_reference_before_pulse(self):
        p = self.p
        x = np.array([p.length_um / 2, p.length_um / 2 + 10])
        t = np.array([0.0, p.pulse_start_ms - .1, p.pulse_start_ms])
        self.assertTrue(np.array_equal(infinite_cable_reference(x, t, p), np.zeros((2,3))))

    def test_green_agreement_near_centre(self):
        metrics, _, _ = calculate_validation(self.sol)
        self.assertLess(metrics['centre_temporal_rmse_mV'], 0.06)
        self.assertLess(metrics['central_1lambda_profile_rmse_mV'], 0.08)

    def test_validation(self):
        with self.assertRaises(ValueError): CableParameters(n_points=200)
        with self.assertRaises(ValueError): CableParameters(pulse_end_ms=100)

if __name__ == '__main__': unittest.main()
