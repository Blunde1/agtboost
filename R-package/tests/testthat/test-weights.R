context("Observation weights")
# agtboost simulates its complexity criterion, so fits compared for equality
# are run from the same seed.
library(agtboost)

# Each check has a closed-form answer: with weights, the optimal constant for
# every supported loss is the weighted mean of y (on the response scale), and a
# well-separated two-group design converges to the weighted group means.

wmean <- function(y, w) sum(w * y) / sum(w)

test_that("intercept-only fit equals the weighted mean for every loss", {
    set.seed(112358)
    n <- 400
    x <- matrix(1, n, 1) # no split possible: the model is its initial prediction
    w <- runif(n, 0.1, 3)
    cases <- list(
        list(loss = "mse", y = rnorm(n, 5, 2) + 3 * (w > 1.5)),
        list(loss = "logloss", y = rbinom(n, 1, ifelse(w > 1.5, 0.8, 0.2))),
        list(loss = "poisson", y = rpois(n, ifelse(w > 1.5, 6, 2))),
        list(loss = "gamma::log", y = rexp(n, ifelse(w > 1.5, 1 / 6, 1 / 2))),
        list(loss = "gamma::neginv", y = rexp(n, ifelse(w > 1.5, 1 / 6, 1 / 2))),
        list(loss = "negbinom", y = rnbinom(n, size = 2, mu = ifelse(w > 1.5, 6, 2)))
    )
    for (case in cases) {
        args <- list(case$y, x, loss_function = case$loss, weights = w,
                     learning_rate = 0.1, nrounds = 50)
        if (case$loss == "negbinom") args$dispersion <- 2
        mod <- suppressWarnings(do.call(gbt.train, args))
        pred <- predict(mod, x)
        # The design makes weighted and unweighted means differ materially
        expect_gt(abs(wmean(case$y, w) - mean(case$y)), 0.1 * mean(case$y))
        expect_equal(pred, rep(wmean(case$y, w), n), tolerance = 1e-6,
                     info = case$loss)
    }
})

test_that("weights shift group predictions by the weighted-minus-unweighted mean", {
    # The information criterion stops boosting slightly short of the group
    # means whether or not weights are used, so compare the shift caused by
    # weighting with the shift in the target rather than the levels.
    set.seed(112358)
    n <- 2000
    g <- rep(0:1, each = n / 2)
    x <- matrix(g + runif(n, 0, 0.1), n, 1)
    heavy <- rep(c(TRUE, FALSE), n / 2)
    w <- ifelse(heavy, 3, 0.5)
    # Within each group, heavily weighted rows have a higher mean
    y_mse <- 10 * g + ifelse(heavy, 2, -2) + rnorm(n, 0, 0.5)
    y_pois <- rpois(n, exp(1 + g + ifelse(heavy, 0.5, -0.5)))
    for (case in list(list(loss = "mse", y = y_mse), list(loss = "poisson", y = y_pois))) {
        set.seed(1)
        fit_w <- suppressWarnings(gbt.train(case$y, x, loss_function = case$loss, weights = w,
                                            learning_rate = 0.1, nrounds = 2000))
        set.seed(1)
        fit_u <- suppressWarnings(gbt.train(case$y, x, loss_function = case$loss,
                                            learning_rate = 0.1, nrounds = 2000))
        pred_w <- predict(fit_w, x)
        pred_u <- predict(fit_u, x)
        for (k in 0:1) {
            target_shift <- wmean(case$y[g == k], w[g == k]) - mean(case$y[g == k])
            pred_shift <- wmean(pred_w[g == k], w[g == k]) - wmean(pred_u[g == k], w[g == k])
            expect_equal(pred_shift, target_shift, tolerance = 0.05, info = case$loss)
        }
    }
})

test_that("rescaling all weights by a constant leaves the fit unchanged", {
    set.seed(112358)
    n <- 500
    x <- matrix(runif(n * 2), n, 2)
    y <- 3 * x[, 1] + rnorm(n)
    w <- runif(n, 0.1, 3)
    set.seed(1)
    fit_1 <- gbt.train(y, x, weights = w, learning_rate = 0.1, nrounds = 500)
    set.seed(1)
    fit_5 <- gbt.train(y, x, weights = 5 * w, learning_rate = 0.1, nrounds = 500)
    expect_equal(predict(fit_5, x), predict(fit_1, x), tolerance = 1e-8)
})

test_that("rescaling all weights leaves the non-linearity check unchanged", {
    # The training loss grows with the scale of the weights, while the check
    # uses a fixed tolerance; weights are therefore rescaled to mean one.
    set.seed(112358)
    n <- 500
    x <- matrix(runif(n * 2), n, 2)
    y <- rexp(n, 1 / exp(0.5 + x[, 1]))
    w <- runif(n, 0.1, 3)
    count_warnings <- function(weights) {
        n_warn <- 0
        set.seed(1)
        withCallingHandlers(
            gbt.train(y, x, loss_function = "gamma::log", weights = weights),
            warning = function(e) {
                n_warn <<- n_warn + 1
                invokeRestart("muffleWarning")
            })
        n_warn
    }
    expect_equal(count_warnings(10 * w), count_warnings(w))
})

test_that("weighted fits do not depend on row order", {
    set.seed(112358)
    n <- 500
    x <- matrix(runif(n * 2), n, 2)
    y <- 3 * x[, 1] + rnorm(n)
    w <- runif(n, 0.1, 3)
    p <- sample(n)
    set.seed(1)
    fit <- gbt.train(y, x, weights = w, learning_rate = 0.1, nrounds = 500)
    set.seed(1)
    fit_p <- gbt.train(y[p], x[p, ], weights = w[p], learning_rate = 0.1, nrounds = 500)
    expect_equal(predict(fit_p, x), predict(fit, x), tolerance = 1e-8)
})

test_that("unit weights reproduce the unweighted fit", {
    set.seed(112358)
    n <- 500
    x <- matrix(runif(n * 2), n, 2)
    y <- 3 * x[, 1] + rnorm(n)
    set.seed(1)
    fit_null <- gbt.train(y, x, learning_rate = 0.1, nrounds = 500)
    set.seed(1)
    fit_one <- gbt.train(y, x, weights = rep(1, n), learning_rate = 0.1, nrounds = 500)
    expect_equal(predict(fit_one, x), predict(fit_null, x))
    expect_equal(fit_one$get_num_trees(), fit_null$get_num_trees())
})

test_that("weighted mse loss matches its gradient approximation", {
    # The training loss and its second-order approximation agree exactly for
    # mse, so a weighted fit should raise no non-linearity warning.
    set.seed(112358)
    n <- 500
    x <- matrix(runif(n), n, 1)
    y <- 3 * x[, 1] + rnorm(n)
    w <- runif(n, 0.1, 3)
    expect_silent(gbt.train(y, x, weights = w, learning_rate = 0.1, nrounds = 500))
})

test_that("invalid weights are rejected", {
    x <- matrix(runif(100), 100, 1)
    y <- rnorm(100)
    expect_error(gbt.train(y, x, weights = rep(1, 99)))
    expect_error(gbt.train(y, x, weights = c(-1, rep(1, 99))))
    expect_error(gbt.train(y, x, weights = c(NA, rep(1, 99))))
    expect_error(gbt.train(y, x, weights = rep(0, 100)))
    expect_error(suppressWarnings(gbt.train(rpois(100, 2), x, loss_function = "count::auto",
                                            weights = runif(100))))
})
