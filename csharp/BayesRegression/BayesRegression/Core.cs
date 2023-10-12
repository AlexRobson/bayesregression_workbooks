using MathNet.Numerics.LinearAlgebra;
using Accord.Statistics.Distributions.Multivariate;
using System;

namespace BayesRegression {

    public class BayesRegressor
    {
        // P(w | alpha)
        public static MultivariateNormalDistribution Prior(double alpha)
        {
            double[] m_0 = Vector<double>.Build.Dense(2, 0).ToArray();
            double[,] S_0 = (Matrix<double>.Build.DenseIdentity(2) / alpha).ToArray();
            return new MultivariateNormalDistribution(m_0, S_0);
        }

        // Helper method for posterior calculation
        public static (Vector<double> m_n, Matrix<double> S_n) _Posterior(Matrix<double> phi, Vector<double> t, double alpha = 1.0, double beta = 1.0)
        {
            int D = 2;
            var m_0 = Vector<double>.Build.Dense(D, 0);
            var S_0 = Matrix<double>.Build.DenseIdentity(D) / alpha;
            var S_n_inv = S_0 * alpha + (phi.TransposeThisAndMultiply(phi)) * beta;
            var S_n = S_n_inv.Inverse();
            var m_n = S_n * (phi.Transpose() * t) * beta;
            return (m_n, S_n);
        }

        // P(w | t, alpha, beta)
        public static MultivariateNormalDistribution Posterior(Matrix<double> phi, Vector<double> t, double alpha = 1.0, double beta = 1.0)
        {
            var (m_n, S_n) = _Posterior(phi, t, alpha, beta);
            return new MultivariateNormalDistribution(m_n.ToArray(), S_n.ToArray());
        }

    }

}