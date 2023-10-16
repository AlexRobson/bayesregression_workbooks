using System;
using BayesRegression;


class Program
{
    /// <Main>
    /// Main method for Bayesregression. 
    /// Generates the figure 3.7 plot from Bishop's Pattern Recognition and Machine Learning. 
    /// Creates a MultiPlot.bmp file in the current directory.
    /// Takes no arguments and returns nothing.
    /// Doesn't ingest data, instead generates data randomly from a known distribution.
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