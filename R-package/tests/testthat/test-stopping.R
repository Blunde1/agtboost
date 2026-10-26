context("Stopping")
library(agtboost)

test_that("boosting stops when no split is possible", {
    set.seed(112358)
    n <- 400
    y <- rnorm(n, 5)
    x <- matrix(1, n, 1)
    mod <- gbt.train(y, x, nrounds = 5000)
    expect_lt(mod$get_num_trees(), 5)
    expect_equal(predict(mod, x), rep(mean(y), n), tolerance = 1e-8)
    # The same holds when training continues from previous predictions
    mod_prev <- gbt.train(y, x, nrounds = 5000, previous_pred = rep(5, n))
    expect_lt(mod_prev$get_num_trees(), 5)
})
