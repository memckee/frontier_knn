# Mean-Variance Optimization and Machine Learning

This project combines multi-period mean-variance optimization through the Markowitz efficient frontier and time series forecasting through the k-nearest neighbors (KNN) algorithm. The process of building an efficient frontier relies on historical covariances and expected returns. Previously, return predictions were made using historical average investment returns. Backtesting the optimization process resulted in higher risk-adjusted portfolio returns, but the strategy underperformed a portfolio weighted 60% to U.S. equities and 40% to U.S. credit.

To address the weakness in return forecasting, this project employs the KNN algorithm with time series complexity adjustments and normalization. Normalization allows for comparisons to previous sub-series. Complexity adjustments scale the Euclidean distance metric so that the values of the sub-series are not the only determinant of distance.

More information about implementing KNN for time series [is available in this repository.](https://github.com/mem692/knn_tspi)

More information about this project's implementation of mean-variance optimization [is available in this repository.](https://github.com/mem692/eff_frontier)

To test KNN and multi-period optimizations, this project uses portfolio allocations aggregated by [Portfolio Charts.](https://portfoliocharts.com/portfolios/) Portfolio Charts has tested allocations recommended across many different publications. This project answers the question - given the same investments, can machine learning and optimization create better performing portfolios?

Python code, output, analysis, and discussion are available in the Jupyter Notebook included in this repository.

