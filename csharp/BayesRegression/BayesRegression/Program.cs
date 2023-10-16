using System;
using BayesRegression;


class Program
{
    static void Main(string[] args)
    {

        // Initial values
        double alpha = 2.0;
        double beta = Math.Pow(5, 2);
        double[] w_true = { -0.3, 0.5 };

        // Generate data
        var data = DataGenerator.GenerateData(10, w_true);

        Plotting.BayesPlot(data, alpha, beta);

    }
}