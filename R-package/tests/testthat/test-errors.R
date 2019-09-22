context("Error handling")
library(gbtorch)

test_that("gbt.train throws appropriate errors", {
    
    # Empty example, iteratively build until non-missing
    
    # Empty
    expect_error( gbt.train( param, x, y ) )
    
    # Example
    x <- as.matrix(runif(500, 0, 4))
    y <- rnorm(500, x, 1)
    param <- list("learning_rate" = 0.03, "loss_function" = "mse", "nrounds"=2000)
    
    # x not a matrix
    expect_error( gbt.train( param, y, "hello" ) )
    
    # y not a vector or 1-d matrix
    expect_error( gbt.train( param, "hello", x ) )
    
    # x y different dim
    expect_error( gbt.train( param, rep(0, nrow(x)+1), x ) )
    
    # param wrong values
    expect_error( gbt.train( list("learning_rate" = 2, "loss_function" = "mse", "nrounds"=2000), y, x ) )
    expect_error( gbt.train( list("learning_rate" = 0.5, "loss_function" = "not_a_function", "nrounds"=2000), y, x ) )
    expect_error( gbt.train( list("learning_rate" = 0.5, "loss_function" = "mse", "nrounds"=c(1:5)), y, x ) )
    expect_error( gbt.train( list("learning_rate" = 0.5, "loss_function" = "mse", "nrounds"=-1), y, x ) )
    
})