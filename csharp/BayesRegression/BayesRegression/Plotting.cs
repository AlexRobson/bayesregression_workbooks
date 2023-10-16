using MathNet.Numerics.LinearAlgebra;
using ScottPlot;

namespace BayesRegression
{

public static class Plotting {

    /// <BayesPlot>
    /// Generates the Figure 3.7 plot from Bishop's Pattern Recognition and Machine Learning. Creates a MultiPlot.bmp file in the current directory.
    /// Note parameter ranges outside of -1 to 1 may cause issues with the plotting.
    /// </BayesPlot>
    /// <param name="data">The Data Tuple as specified in Utilities.Data containing fields x, t and w </param>
    /// <param name="alpha">alpha: The precision parameter for the prior. See Bishop for details. </param>
    /// <param name="beta">beta: The precision parameter for the likelihood. See Bishop for details. </param>
    public static void BayesPlot(Data data, double alpha, double beta) {

        double [] x = data.x;
        double [] y = data.y;
        double [] w_star = data.w;

        var rowData = new int[] { 1, 2, 5, data.x.Length };
        int rows = rowData.Length + 1, cols = 3; // +1 for the prior

        var plot_grid = new ScottPlot.Plot[rows,cols];
        var bitmap_grid = new System.Drawing.Bitmap[rows,cols];
        var width = 300;
        var height = 250;

        // If using weights outside of the range -1 to 1, this will need to be adjusted
        double deltaW = 0.2;
        double w_min = -1, w_max = 1;
        double[] W = Enumerable.Range(0, (int)((w_max - w_min) / deltaW) + 1).Select(i => -1 + i * deltaW).ToArray();
        double[][] WGrid = DataGenerator.GenerateGrid(W).ToArray();

        // Generate prior
        double[,] pPrior = Utilities.Reshape(
            WGrid.Select(w => BayesRegressor.Prior(alpha).ProbabilityDensityFunction(w)).ToArray(),
            W.Length, W.Length
        );

        // Plot the prior
        plot_grid[0, 1] = new ScottPlot.Plot(width, height);
        var hm_prior = plot_grid[0, 1].AddHeatmap(pPrior, lockScales: false);
        hm_prior.FlipHorizontally = true; hm_prior.OffsetX = -1; hm_prior.OffsetY = -1; hm_prior.CellWidth = deltaW; hm_prior.CellHeight = deltaW;
        bitmap_grid[0, 1] = plot_grid[0, 1].Render();

        // Create each row in the 3.7 plot
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

            plot_grid[i+1, 0] = new ScottPlot.Plot(width, height);
            plot_grid[i+1, 1] = new ScottPlot.Plot(width, height);
            plot_grid[i+1, 2] = new ScottPlot.Plot(width, height);

            // Likelihood
            var hm = plot_grid[i+1, 0].AddHeatmap(pLikelihood, lockScales: false);
            plot_grid[i+1, 0].AddScatter(new double[] {w_star[0]}, new double[] {w_star[1]}, color: System.Drawing.Color.Red);
            hm.FlipHorizontally = true; hm.OffsetX = -1; hm.OffsetY = -1; hm.CellWidth = deltaW; hm.CellHeight = deltaW;

            // Posterior
            hm = plot_grid[i+1, 1].AddHeatmap(pPosterior, lockScales: false);
            hm.FlipHorizontally = true; hm.OffsetX = -1; hm.OffsetY = -1; hm.CellWidth = deltaW; hm.CellHeight = deltaW;
            plot_grid[i+1, 1].AddScatter(new double[] {w_star[0]}, new double[] {w_star[1]}, color: System.Drawing.Color.Red);

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

        // Take all the rendered images and combine them into a grid. Save the grid as a bitmap
        SaveBitMapGrid(bitmap_grid, 5, 3, (width, height));

    }

    private static void SaveBitMapGrid (System.Drawing.Bitmap[,] bitmap_grid, int rows, int cols, (int width, int height) size) {

        using (var bmp = new System.Drawing.Bitmap(size.width * cols, size.height * rows)) // Adjusted size to fit all plots
        using (var gfx = System.Drawing.Graphics.FromImage(bmp))
        {
            for (int row = 0; row < 5; row++)
            {
                for (int col = 0; col < 3; col++)
                {
                    if (bitmap_grid[row, col] != null) // Null check
                    {
                        int x = col * size.width;
                        int y = row * size.height;
                        gfx.DrawImage(bitmap_grid[row, col], x, y);
                    }
                }
            }
        bmp.Save("MultiPlot.bmp");
        }
    }
}

}