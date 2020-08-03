# greedy trees vs greedy ensemble
# Berent Lunde
# 26.09.2019

# Description: Illustrates the difference of standard gradient tree boosting 
# which is implemented in V1 aGTBoost (greedy_complexities = FALSE) and the 
# V2 aGTBoost greedy complexities algorithm (greedy_complexities = TRUE).
# References to future paper is coming...!


# Load aGTBoost library
library(agtboost)


# Simulate data
## A simple gtb.train example with linear regression:
set.seed(123)
x <- as.matrix(runif(500, 0, 4))
y <- rnorm(500, 5 * x, 1)
x.test <- as.matrix(runif(500, 0, 4))
y.test <- rnorm(500, 5* x.test, 1)


# -- Train models --
mod_vanilla <- gbt.train(y, x, verbose=1, gsub_compare=F)
mod_gsubc <- gbt.train(y, x, verbose=1, gsub_compare=T)


# -- Predict on test --
# greedy tree
y.pred.vanilla <- predict( mod_vanilla, as.matrix( x.test ) )
plot(x.test, y.test)
points(x.test, y.pred.vanilla, col="red")
mean((y.test - y.pred.vanilla)^2)
# greedy complexities
y.pred.modified <- predict( mod_gsubc, as.matrix( x.test ) )
#plot(x.test, y.test)
points(x.test, y.pred.modified, col="blue")
mean((y.test - y.pred.modified)^2)


# -- ILLUSTRATION --
# Greedy tree algorithm
plot(x,y)
k1 <- mod_vanilla$get_num_trees()
for(k in ceiling(seq(1,k1, length.out = 10))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- mod_vanilla$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}

# Greedy complexities algorithm
plot(x,y)
k2 <- mod_gsubc$get_num_trees()
for(k in ceiling(seq(1,k2, length.out = 10))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- mod_gsubc$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}
