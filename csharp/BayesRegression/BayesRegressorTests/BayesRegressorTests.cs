using Xunit;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using Accord.Statistics.Distributions.Multivariate;
using BayesRegression;

namespace BayesRegressorTests
{

    public static class TestUtilities
    {

    public static void AssertArraysAlmostEqual(double[] array1, double[] array2, double tolerance)
    {
        var length = array1.GetLength(0);
        Assert.Equal(array1.GetLength(0), array2.GetLength(0));

        for (int i = 0; i < length; i++)
        {
            Assert.InRange(array1[i], array2[i] - tolerance, array2[i] + tolerance);
        }
    }

        public static void AssertMatriciesAlmostEqual(double[,] array1, double[,] array2, double tolerance)
    {
        int rows = array1.GetLength(0);
        int cols = array1.GetLength(1);
        Assert.Equal(rows, array2.GetLength(0));
        Assert.Equal(cols, array2.GetLength(1));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                Assert.InRange(array1[i, j], array2[i, j] - tolerance, array2[i, j] + tolerance);
            }
        }
    }

    }

    public class BayesRegressorTests
    {
        private double alpha;
        private double beta;
        private Vector<double> t;
        private Vector<double> x;
        private Matrix<double> phi;

        public BayesRegressorTests()
        {
            alpha = 10.0;
            beta = 5.0;
            t = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 2, 3, 4 });
            x = Vector<double>.Build.DenseOfArray(new double[] { 1.0, 2, 3, 4 });
            phi = Matrix<double>.Build.DenseOfArray(new double[,] { { 1.0, 2 }, { 1.0, 3 }, { 1.0, 4 }, { 1.0, 5 } });
        }

        [Fact]
        public void TestPrior()
        {
            var result = BayesRegressor.Prior(alpha);
            Assert.IsType<MultivariateNormalDistribution>(result);
            Assert.Equal(Vector<double>.Build.Dense(2, 0), result.Mean);
            Assert.Equal(Matrix<double>.Build.DenseIdentity(2).Multiply(1/alpha).ToArray(), result.Covariance);

            // Test with a different alpha
            var resultAlpha = BayesRegressor.Prior(5.0);
            Assert.Equal(Vector<double>.Build.Dense(2, 0), resultAlpha.Mean);
            Assert.Equal(Matrix<double>.Build.DenseIdentity(2).Multiply(1/5.0).ToArray(), resultAlpha.Covariance);
        }

        [Fact]
        public void TestPosterior()
        {
            var result = BayesRegressor.Posterior(phi, t, alpha, beta);
            Assert.IsType<MultivariateNormalDistribution>(result);
            
            var S_n = (
                Matrix<double>.Build.DenseIdentity(2).Multiply(alpha) + beta * phi.Transpose() * phi)
                .Inverse();
            var m_n = S_n * (phi.Transpose() * t) * beta;

            // Test that they equal
            Assert.Equal(result.Mean, m_n.ToArray());
            Assert.Equal(result.Covariance, S_n.ToArray());

            // Test that it works with a different alpha and beta
            var result_alpha_beta = BayesRegressor.Posterior(phi, t, alpha = 3.3, beta = 7.8);
            var S_n_alpha_beta = (3.3 * Matrix<double>.Build.DenseIdentity(2) + 7.8 * phi.Transpose() * phi).Inverse();
            var m_n_alpha_beta = 7.8 * S_n_alpha_beta * phi.Transpose() * t;
            // Needs precision
            TestUtilities.AssertArraysAlmostEqual(result_alpha_beta.Mean, m_n_alpha_beta.ToArray(), 1e-10);
            TestUtilities.AssertMatriciesAlmostEqual(result_alpha_beta.Covariance, S_n_alpha_beta.ToArray(), 1e-10);
        }

        // Test the test utilities
        [Fact]
        public void TestMethod()
        {
            double[] array1 = {1.0, 2.0, 3.0, 4.0 };
            double[] array2 = {1.01, 2.01, 3.01, 4.01};

            double[,] matrix1 = { { 1.0, 2.0 }, { 3.0, 4.0 } };
            double[,] matrix2 = { { 1.01, 2.01 }, { 3.01, 4.01 } };

            TestUtilities.AssertArraysAlmostEqual(array1, array2, 1e-2);
            TestUtilities.AssertMatriciesAlmostEqual(matrix1, matrix2, 1e-2);
        }
    }
}