# boosted training given previous predictions
# Berent Lunde
# 29.09.2019

# Description: Illustrates boosted training given previous predictions from some model.
# aGTBoost will immediately calculate derivatives to train from, and try to find what complexity
# still not added to the previous model (e.g. non-linearity, interactions, ...)
# Useful if previous model is relatively simple, e.g. a sparse (regularized) linear model

# Load aGTBoost library
library(agtboost)

# Simulate y~N(x, 4), x~U(0,4)
x <- runif(500, 0, 4)
y <- rnorm(500, x, 1)
x.test <- runif(500, 0, 4)
y.test <- rnorm(500, x.test, 1)

# First do standard boosting
mod <- gbt.train(y, as.matrix(x), verbose=0)
mod$get_num_trees()
y.pred <- predict( mod, as.matrix( x.test ) )

plot(x.test, y.test)
points(x.test, y.pred, col=2)

# Train a linear model
lm.mod <- lm(y~., data.frame(y=y, x=x))

# Assume L2 regularized with some strength --> preds scaled with 0.7
lm.mod$coefficients["x"] <- lm.mod$coefficients["x"] * 0.7
preds <- predict(lm.mod, data.frame(x))
preds.test <- predict(lm.mod, newdata=data.frame(x=x.test))

# How does this look like?
points(x.test, preds.test, col=3)

# Train from regularized linear model
mod2 <- gbt.train(y, as.matrix(x), verbose=0, previous_pred = preds)
mod2$get_num_trees() # Smaller than boosting iterations for mod -- less added complexity needed

y.pred2 <- predict( mod2, as.matrix(x.test))
points(x.test, preds.test + y.pred2, col=4)

# MSE comparisons
mean((y.test - y.pred)^2)
mean((y.test - preds.test)^2)
mean((y.test - (preds.test + y.pred2))^2)
