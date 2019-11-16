<!-- badges: start -->
[![Travis build status](https://travis-ci.org/Blunde1/gbtorch.svg?branch=master)](https://travis-ci.org/Blunde1/gbtorch)
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://www.tidyverse.org/lifecycle/#experimental)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---------

# GBTorch

<dl>
  <dt>Torch</dt>
  <dd>something considered as a source of illumination, enlightenment, guidance, etc.
    </dd>
</dl>

GBTorch is a lightning fast gradient boosting library designed to **avoid manual tuning** and **cross-validation** by utilizing an information theoretic approach.
This makes the algorithm **adaptive** to the dataset at hand; it is **completely automatic**, and with **minimal worries of overfitting**.
Consequently, the speed-ups relative to state-of-the-art implementations are in the thousands while mathematical and technical knowledge required on the user are minimized.


## Installation

**R**: Install the development version from GitHub
```r
devtools::install_github("Blunde1/gbtorch/R-package")
```
Users experiencing errors after warnings during installlation, may be helped by the following command prior to installation:

```r
Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS="true")
```

## Example code and documentation

`gbtorch` essentially has two functions, a train function `gbt.train` and a predict function `predict`.
From the code below it should be clear how to train a GBTorch model using a design matrix `x` and a response vector `y`, write `?gbt.train` in the console for detailed documentation. 
```r
library(gbtorch)

# -- Load data --
data(caravan.train, package = "gbtorch")
data(caravan.test, package = "gbtorch")
train <- caravan.train
test <- caravan.test

# -- Model building --
mod <- gbt.train(train$y, train$x, loss_function = "logloss", verbose=10)

# -- Predictions --
pred <- predict(mod, test$x) # Score before logistic transformation
prob <- 1/(1+exp(-pred)) # Probabilities
```

Furthermore, a GBTorch model is (see example code)

- highly robust to dimensions: [Comparisons to (penalized) linear regression in (very) high dimensions](R-package/demo/gbt_high_dim.R)
- has minimal worries of overfitting: [Stock market classificatin](R-package/demo/stock_market_classification.R)
- and can train further given previous models: [Boosting from a regularized linear model](R-package/demo/boost_from_predictions.R)



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
