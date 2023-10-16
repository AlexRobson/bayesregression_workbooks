using MathNet.Numerics.LinearAlgebra;
using Accord.Statistics.Distributions.Multivariate;
using System;

namespace BayesRegression {

    public class BayesRegressor
    {
        // 
        /// <summary>
        /// Returns a MultivariateNormalDistribution object representing the prior distribution P(w | alpha) over the weights.
        /// <param> alpha: The precision parameter for the prior.  </param>
        /// </summary>
        /// <returns> A MultivariateNormalDistribution object </returns>
        public static MultivariateNormalDistribution Prior(double alpha)
        {
            double[] m_0 = Vector<double>.Build.Dense(2, 0).ToArray();
            double[,] S_0 = (Matrix<double>.Build.DenseIdentity(2) / alpha).ToArray();

            // The MvNormal type doesn't seem to support the specialise Matrix types, so we convert to arrays
            return new MultivariateNormalDistribution(m_0, S_0);
        }

        // P(w | t, alpha, beta)
        /// <summary>
        /// Returns a MultivariateNormalDistribution object representing the posterior distribution P(w | t, alpha, beta) over the weights.
        /// <param> phi: The design matrix </param>
        /// <param> t: The target values </param>
        /// <param> alpha: The precision parameter for the prior. Default 1.0 </param>
        /// <param> beta: The precision parameter for the likelihood. Default 1.0 </param>
        /// </summary>
        public static MultivariateNormalDistribution Posterior(Matrix<double> phi, Vector<double> t, double alpha = 1.0, double beta = 1.0)
        {
            int D = 2;
            var m_0 = Vector<double>.Build.Dense(D, 0);
            var S_0 = Matrix<double>.Build.DenseIdentity(D) / alpha;
            // S_0 inverse is just alphaI
            var S_n_inv = S_0.Inverse() + (phi.TransposeThisAndMultiply(phi)) * beta;
            var S_n = S_n_inv.Inverse();
            var m_n = S_n * (phi.Transpose() * t) * beta;
            return new MultivariateNormalDistribution(m_n.ToArray(), S_n.ToArray());
        }

        /// <summary>
        /// Returns the design matrix for a given vector of x values. Simply concatenates a column of ones to the x values.
        /// <param> x: The vector of x values </param>
        /// </summary>
        public static Matrix<double> Phi(Vector<double> x)
        {
            int N = x.Count;
            var ones = Vector<double>.Build.Dense(N, 1.0);
            var stackedMatrix = Matrix<double>.Build.DenseOfColumnVectors(ones, x);
            return stackedMatrix;
        }

        /// <summary>
        /// Returns the log likelihood of the data given the weights and the precision parameter beta.
        /// <param> w: The weights </param>
        /// <param> phi: The design matrix </param>
        /// <param> t: The target values </param>
        /// <param> beta: The precision parameter for the likelihood. Default 1.0 </param>
        /// </summary>
        public static double LogLikelihood(Vector<double> w, Matrix<double> phi, Vector<double> t, double beta = 1.0)
        {
            int N = phi.RowCount;
            var difference = t - phi * w;
            double E_d = 0.5 * difference.DotProduct(difference);
            double ll = (N / 2.0) * (Math.Log(beta) - Math.Log(2 * Math.PI)) - beta * E_d;
            return ll;
        }
    }
}