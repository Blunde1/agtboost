context("Error handling")
library(agtboost)

test_that("gbt.train throws appropriate errors", {
    
    # Empty example, iteratively build until non-missing
    
    # Empty
    expect_error( gbt.train( x, y ) )
    
    # Example
    x <- as.matrix(runif(500, 0, 4))
    y <- rnorm(500, x, 1)

    # x not a matrix
    expect_error( gbt.train( y, "not a matrix" ) )
    
    # y not a vector or 1-d matrix
    expect_error( gbt.train( "not a vector", x ) )
    
    # x y different dim
    expect_error( gbt.train( rep(0, nrow(x)+1), x ) )
    
    # param wrong values
    expect_error( gbt.train( y, x, learning_rate = 2) )
    expect_error( gbt.train( y, x, loss_function = "not_a_function" ) )
    expect_error( gbt.train( y, x, nrounds = c(1:5) ) )
    expect_error( gbt.train( y, x, nrounds = -1) )
    
})