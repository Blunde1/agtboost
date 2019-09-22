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
