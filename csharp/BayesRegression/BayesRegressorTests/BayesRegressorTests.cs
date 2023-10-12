using Xunit;
using MathNet.Numerics.LinearAlgebra;
using Accord.Statistics.Distributions.Multivariate;
using BayesRegression;

namespace BayesRegressorTests
{
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
        public void Test_Posterior()
        {
            // ... Your translated test code here ...
        }

        [Fact]
        public void TestPosterior()
        {
            // ... Your translated test code here ...
        }

        // ... other test methods ...
    }
}