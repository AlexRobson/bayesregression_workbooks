import unittest
from scipy.stats._multivariate import multivariate_normal_frozen
import numpy as np
from bayesregression.bayesregression import prior, _posterior, posterior, phi, loglikelihood


class TestBayesRegression(unittest.TestCase):

    def setUp(self):
        self.alpha = 10.0
        self.beta = 5.0
        self.t = np.array([1.0, 2, 3, 4])
        self.x = np.array([1.0, 2, 3, 4])
        self.phi = np.array([[1.0, 2], [1.0, 3], [1.0, 4], [1.0, 5]])

    def test_prior(self):
        result = prior(self.alpha)
        self.assertIsInstance(result, multivariate_normal_frozen)
        self.assertTrue(np.allclose(result.mean, np.zeros(2)))
        self.assertTrue(np.allclose(result.cov, (1/self.alpha) * np.eye(2)))

        # Test that it works with a different alpha
        result_alpha = prior(5.0)
        self.assertTrue(np.allclose(result_alpha.mean, np.zeros(2)))
        self.assertTrue(np.allclose(result_alpha.cov, (1/5.0) * np.eye(2)))

    def test__posterior(self):
        
        result = _posterior(self.phi, self.t, self.alpha, self.beta)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (2,))
        self.assertEqual(result[1].shape, (2, 2))
        
        S_n = np.linalg.inv(self.alpha * np.eye(2) + self.beta * self.phi.T @ self.phi)
        m_n = self.beta * S_n @ self.phi.T @ self.t

        # Test that they equal
        self.assertTrue(np.allclose(result[0], m_n))
        self.assertTrue(np.allclose(result[1], S_n))

        # Test that it works with a different alpha and beta
        result_alpha_beta = _posterior(self.phi, self.t, alpha = 3.3, beta = 7.8)
        S_n_alpha_beta = np.linalg.inv(3.3 * np.eye(2) + 7.8 * self.phi.T @ self.phi)
        m_n_alpha_beta = 7.8 * S_n_alpha_beta @ self.phi.T @ self.t
        self.assertTrue(np.allclose(m_n_alpha_beta, result_alpha_beta[0]))
        self.assertTrue(np.allclose(S_n_alpha_beta, result_alpha_beta[1]))

        
    def test_posterior(self):
        m_n, S_n = _posterior(self.phi, self.t, self.alpha, self.beta)
        result = posterior(self.phi, self.t, self.alpha, self.beta)

        self.assertIsInstance(result, multivariate_normal_frozen)
        self.assertTrue(np.allclose(result.mean, m_n))
        self.assertTrue(np.allclose(result.cov, S_n))

    def test_phi(self):
        x = [1.0,2,3,4]
        expected = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2, 3, 4]]).T
        result = phi(x)
        self.assertTrue(np.allclose(result, expected))

    def test_loglikelihood(self):
        w = np.array([1.0, 2])
        result = loglikelihood(w, self.phi, self.t, self.beta)

        E_d = 0.5 * np.sum((self.t - np.dot(self.phi, w))**2)
        expected = (4 / 2) * (np.log(self.beta) - np.log(2 * np.pi)) - self.beta * E_d
        self.assertAlmostEqual(result, expected)

        # Test that it works with a different beta. Use differences in loglikelihoods
        result_beta = loglikelihood(w, self.phi, self.t, 10.0)
        expected_differences = (4 / 2) * (np.log(5) - np.log(10.0)) - (5 - 10.0) * E_d
        self.assertAlmostEqual(result - result_beta, expected_differences)