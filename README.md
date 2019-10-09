<!-- badges: start -->
[![Travis build status](https://travis-ci.org/Blunde1/gbtorch.svg?branch=master)](https://travis-ci.org/Blunde1/gbtorch)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---------

# GBTorch

GBTorch is a lightning fast gradient boosting library designed to **avoid manual tuning** and **cross-validation** by utilizing an information theoretic approach.
This makes the algorithm **adaptive** to the dataset at hand; it is **completely automatic**, and with **minimal worries of overfitting**.
Consequently, the speed-ups relative to state-of-the-art implementations are in the thousands while mathematical and technical knowledge required on the user are minimized.


## Installation

**R**: Install the development version from GitHub
```r
devtools::install_github("Blunde1/gbtorch/R-package")
```

## Example code and documentation

Below is an example using the R-package `gbtorch` to model the `Smarket` data (S&P500 daily stock-market data) contained in the `ISLR` package.
`gbtorch` essentially has two functions, a train function `gbt.train` and a predict function `predict`.
From the code below it should be clear how to train a GBTorch model using a design matrix `x` and a response vector `y`, write `?gbt.train` in the console for detailed documentation. 
```r
library(gbtorch)
library(ISLR) # Contains the "Smarket" data

# -- Data management --
data("Smarket")

# Out of time train-test: year == 2005 is test data
ind_train <- which(Smarket$Year <= 2004)
Smarket <- subset(Smarket, select=-c(Today, Year))

# One-hot encoding
data <- model.matrix(Direction~., data=Smarket)[,-1]

# Split into train and test datasets
x.train <- as.matrix(data[ind_train, ])
y.train <- as.matrix(ifelse(Smarket[ind_train, "Direction"]=="Up",1,0))
x.test <- as.matrix(data[-ind_train, ])
y.test <- as.matrix(ifelse(Smarket[-ind_train, "Direction"]=="Up", 1, 0))

# -- Model building --
gbt.mod <- gbt.train(y.train, x.train, learning_rate = 0.01, loss_function = "logloss")

# -- Predictions --
test.pred <- predict(gbt.mod, x.test) # Score before logistic transformation
test.prob <- 1/(1+exp(-test.pred)) # Probabilities
```

## Dependencies

- [My research](https://berentlunde.netlify.com/) 
- [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) Linear algebra
- [Rcpp](https://github.com/RcppCore/Rcpp) for the R-package

## Scheduled updates

- Adaptive and automatic deterministic frequentist gradient tree boosting. (October/November 2019, planned)
- Optimal deterministic frequentist gradient tree boosting. (January/February 2020, planned)
- Optimal deterministic gradient tree boosting. (May/June 2020, planned)

## Hopeful updates

- Optimal stochastic gradient tree boosting.

## References
are coming.

## Contribute

Any help on the following subjects are especially welcome:

- Utilizing sparsity (possibly Eigen sparsity).
- Paralellizatin (CPU and/or GPU).
- Distribution (Python, Java, Scala, ...),
- good ideas and coding best-practices in general.

Please note that the priority is to work on and push the above mentioned scheduled updates. Patience is a virtue. :)
