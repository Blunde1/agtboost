# greedy trees vs greedy ensemble
# Berent Lunde
# 26.09.2019

# Description: Illustrates the difference of standard gradient tree boosting 
# which is implemented in V1 GBTorch (greedy_complexities = FALSE) and the 
# V2 GBTorch greedy complexities algorithm (greedy_complexities = TRUE).
# References to future paper is coming...!


# Load GBTorch library
library(gbtorch)


# Simulate data
## A simple gtb.train example with linear regression:
set.seed(123)
x <- as.matrix(runif(500, 0, 4))
y <- rnorm(500, 5 * x, 1)
x.test <- as.matrix(runif(500, 0, 4))
y.test <- rnorm(500, 5* x.test, 1)


# -- Train models --
greedy_tree_mod <- gbt.train(y, x, verbose=1, greedy_complexities=F)
greedy_complexities_mod <- gbt.train(y, x, verbose=1, greedy_complexities=T)


# -- Predict on test --
# greedy tree
y.pred.vanilla <- predict( greedy_tree_mod, as.matrix( x.test ) )
plot(x.test, y.test)
points(x.test, y.pred.vanilla, col="red")
mean((y.test - y.pred.vanilla)^2)
# greedy complexities
y.pred.modified <- predict( greedy_complexities_mod, as.matrix( x.test ) )
#plot(x.test, y.test)
points(x.test, y.pred.modified, col="blue")
mean((y.test - y.pred.modified)^2)


# -- ILLUSTRATION --
# Greedy tree algorithm
plot(x,y)
k1 <- greedy_tree_mod$get_num_trees()
for(k in ceiling(seq(1,k1, length.out = 10))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- greedy_tree_mod$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}

# Greedy complexities algorithm
plot(x,y)
k2 <- greedy_complexities_mod$get_num_trees()
for(k in ceiling(seq(1,k2, length.out = 10))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- greedy_complexities_mod$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}
