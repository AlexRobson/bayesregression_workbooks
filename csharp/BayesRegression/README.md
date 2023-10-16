Bayesian Linear Regression in C#



# Building and Compiling

## Other architectures

Untested

## Linux-x64

This package uses ScottPlot. You may need to install additional dependencies [here](https://scottplot.net/faq/dependencies/), in particular `libgdiplus`

To build the package: 

```
dotnet publish -c Release -r linux-x64 --self-contained
```

Alternatively, you can run with 

```
dotnet run
```

# Running

The code follows a similar pattern to the Python approach. To run in the dotnet-script REPL (assuming it has been built):

```
#r "bin/Release/net7.0/linux-x64/BayesRegression.dll"
using BayesRegression;
// Initial values
double alpha = 2.0;
double beta = Math.Pow(5, 2);
double[] w_true = { -0.3, 0.5 };

// Generate data
var data = DataGenerator.GenerateData(10, w_true);

Plotting.BayesPlot(data, alpha, beta); 
```

Similarly to the Python implemention, `data` can be replaced with custom data, using the `Data` class.
This is essentially a struct with fields  x, y, and w, each of type double []. See the Data class in Utils.cs
Similarly to the Python implementation, the ranges used in plotting are [-1, 1], and probability mass outside of this is not visualised. 

# Testing

Tests are included in the BayesRegressionTests. Run with `dotnet test .` at the project root. 








