context("Saving and loading models")
library(agtboost)

test_that("a reloaded model predicts exactly as the fitted one", {
    set.seed(112358)
    n <- 200
    # Features spread by about 0.001, so split values need more than six
    # decimal places to survive the round trip
    x <- matrix(1234.5 + rnorm(n * 2, sd = 1e-3), ncol = 2)
    y <- 3 * x[, 1] + rnorm(n, sd = 1e-3)
    for (loss in c("mse", "poisson")) {
        y_loss <- if (loss == "mse") y else rpois(n, exp(1 + 1000 * (x[, 1] - 1234.5)))
        mod <- suppressWarnings(gbt.train(y_loss, x, loss_function = loss, learning_rate = 0.1))
        f <- tempfile()
        gbt.save(mod, f)
        mod2 <- gbt.load(f)
        expect_identical(predict(mod2, x), predict(mod, x), info = loss)
        unlink(f)
    }
})
