using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace BayesRegression {

public class Data
{
    public double[] x { get; }
    public double[] y { get; }
    public double[] w { get; }

    public Data(double[] x, double[] y, double[] w)
    {
        this.x = x;
        this.y = y;
        this.w = w;
    }
}

public static class DataGenerator
{
    /// <GenerateData>
    /// Generates data from a linear model with Gaussian noise: 
    /// y = w_0 + w_1 * x + epsilon where epsilon ~ N(0, 0.2)
    /// <param> N: The number of data points to generate </param>
    /// <param> w: The weights to use in the linear model. Defaults = {-0.3, 0.5} </param>
    /// </GenerateData>
    public static Data GenerateData(int N, double[] w = null)
    {
        w ??= new double[] { -0.3, 0.5 };
        var random = new Random();
        double[] x = Enumerable.Range(0, N).Select(_ => 2 * random.NextDouble() - 1).ToArray();
        double[] y = x.Select(val => w[0] + w[1] * val + 0.2 * random.NextGaussian()).ToArray();

        return new Data(x, y, w);
    }

    // This does something like ITerators.product in Python. It should be replaced by a similar inbuilt function if one exists.
    public static IEnumerable<double[]> GenerateGrid(double[] W)
    {
        return W.SelectMany(w0 => W, (w0, w1) => new double[] { w0, w1 });
    }
}

public static class RandomExtensions
{
    private static Random random = new Random();

    /// <NextGaussian>
    /// Generates a random number from a Gaussian distribution with mean 0 and variance 1. Should be replaced by an inbuilt function if one exists.
    /// </NextGaussian>
    public static double NextGaussian(this Random random)
    {
        double u1 = 1.0 - random.NextDouble();
        double u2 = 1.0 - random.NextDouble();
        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                               Math.Sin(2.0 * Math.PI * u2);
        return randStdNormal;
    }
}

// Reshape method to convert a 1D array to a 2D array
static class Utilities
{
    /// <Reshape>
    /// Reshapes a 1D array to a 2D array with the specified number of rows and columns.
    /// </Reshape>
    public static double[,] Reshape(double[] array, int rows, int cols)
    {
        // Reshape may exist for vectors, but this was needd for arrays.
        double[,] reshapedArray = new double[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                reshapedArray[i, j] = array[i * cols + j];
        return reshapedArray;
    }

}

}