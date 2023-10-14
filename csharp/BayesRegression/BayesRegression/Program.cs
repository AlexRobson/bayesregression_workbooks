using System;
using ScottPlot;
using BayesRegression;
using MathNet.Numerics.LinearAlgebra;

class Program
{
    static void Main(string[] args)
    {

        // Initial values
        double alpha = 2.0;
        double beta = Math.Pow(5, 2);
        double[] w_true = { -0.3, 0.5 };

        var plot_grid = new ScottPlot.Plot[5,3];
        var bitmap_grid = new System.Drawing.Bitmap[5,3];

        // Generate data
        var data = DataGenerator.GenerateData(10, w_true);

        double deltaW = 0.2;
        double[] W = Enumerable.Range(0, (int)((1 - -1) / deltaW) + 1).Select(i => -1 + i * deltaW).ToArray();
        double[][] WGrid = DataGenerator.GenerateGrid(W).ToArray();

        // Generate prior
        double[,] pPrior = Utilities.Reshape(
            WGrid.Select(w => BayesRegressor.Prior(alpha).ProbabilityDensityFunction(w)).ToArray(),
            W.Length, W.Length
        );

        // Plot the prior
        plot_grid[0, 1] = new ScottPlot.Plot(300, 250);
        var hm_prior = plot_grid[0, 1].AddHeatmap(pPrior, lockScales: false);
        hm_prior.FlipHorizontally = true; hm_prior.OffsetX = -1; hm_prior.OffsetY = -1; hm_prior.CellWidth = deltaW; hm_prior.CellHeight = deltaW;
        bitmap_grid[0, 1] = plot_grid[0, 1].Render();

        
        var rowData = new int[] { 1, 2, 5, data.x.Length };
        for (int i = 0; i < rowData.Length; i++)
        {
            int ndata = rowData[i];
            double [] _x = data.x.Take(ndata).ToArray();
            double [] _y = data.y.Take(ndata).ToArray();

            var phi = BayesRegressor.Phi(Vector<Double>.Build.Dense(_x));

            double [,] pLikelihood = Utilities.Reshape(
                WGrid.Select(w => Math.Exp(BayesRegressor.LogLikelihood(Vector<Double>.Build.Dense(w), phi, Vector<Double>.Build.Dense(_y), beta))).ToArray(), 
                W.Length, W.Length
            );

            double [,] pPosterior = Utilities.Reshape(
                WGrid.Select(w => BayesRegressor.Posterior(phi, Vector<Double>.Build.Dense(_y), alpha, beta).ProbabilityDensityFunction(w)).ToArray(),
                W.Length, W.Length
            );

            plot_grid[i+1, 0] = new ScottPlot.Plot(300, 250);
            plot_grid[i+1, 1] = new ScottPlot.Plot(300, 250);
            plot_grid[i+1, 2] = new ScottPlot.Plot(300, 250);

            // This requires manually aligning. 

            // Likelihood
            var hm = plot_grid[i+1, 0].AddHeatmap(pLikelihood, lockScales: false);
            plot_grid[i+1, 0].AddScatter(new double[] {w_true[0]}, new double[] {w_true[1]}, color: System.Drawing.Color.Red);
            hm.FlipHorizontally = true; hm.OffsetX = -1; hm.OffsetY = -1; hm.CellWidth = deltaW; hm.CellHeight = deltaW;

            // Posterior
            hm = plot_grid[i+1, 1].AddHeatmap(pPosterior, lockScales: false);
            hm.FlipHorizontally = true; hm.OffsetX = -1; hm.OffsetY = -1; hm.CellWidth = deltaW; hm.CellHeight = deltaW;
            plot_grid[i+1, 1].AddScatter(new double[] {w_true[0]}, new double[] {w_true[1]}, color: System.Drawing.Color.Red);

            // Move to Utils
            double[] x_grid = Enumerable.Range(0, (int)((1 - -1) / 0.01)).Select(i => -1 + i * 0.01).ToArray();
            for (int _ = 1; _ <= 5; _++)
            {
                double [] sample = BayesRegressor.Posterior(phi, Vector<Double>.Build.Dense(_y), alpha, beta).Generate();
                double a0 = sample[0], a1 = sample[1];
                double[] y_values = x_grid.Select(x => a0 + x * a1).ToArray();
                plot_grid[i+1, 2].AddScatter(x_grid, y_values, color: System.Drawing.Color.Red, lineWidth: 0.01f);
            }

            plot_grid[i+1, 2].AddScatter(_x, _y, lineWidth: 0);

            bitmap_grid[i+1, 0] = plot_grid[i+1, 0].Render();
            bitmap_grid[i+1, 1] = plot_grid[i+1, 1].Render();
            bitmap_grid[i+1, 2] = plot_grid[i+1, 2].Render();
        }

        using (var bmp = new System.Drawing.Bitmap(300 * 3, 250 * 5)) // Adjusted size to fit all plots
        using (var gfx = System.Drawing.Graphics.FromImage(bmp))
        {
            for (int row = 0; row < 5; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    if (bitmap_grid[row, col] != null) // Null check
                    {
                        int x = col * 300;
                        int y = row * 250;
                        gfx.DrawImage(bitmap_grid[row, col], x, y);
                    }
                }
            }
        bmp.Save("MultiPlot.bmp");
        }
    }
}