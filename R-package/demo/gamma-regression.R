# agtboost gamma regression
# log and negative inverse link functions
# Berent Lunde
# 10.11.2019

library(agtboost)

# NEGATIVE INVERSE LINK
# Simulate gamma glm data with negative inverse link
set.seed(314)
n <- 1000
x <- runif(n, 0.5, 4)
b <- -100
shape=3
neginvmu_scaled <- x*b
mu <- -1.0/neginvmu_scaled * shape
scale = mu / shape
y <- rgamma(n, shape=shape, scale = scale)
plot(x,y, main="Gamma glm observations, negative inverse link")
points(x, mu, col=2)
#hist(y)

# prepare for agtboost
y.train <- y
x.train <- as.matrix(x)

# model
mod <- gbt.train(y.train, x.train, loss_function = "gamma::neginv", verbose=20)

# predict
pred.mu <- predict(mod, x.train)

# add predictions to plot
points(x.train, pred.mu, col=3)
legend("topright", c("data", "true mean", "gbt estimated mean"), col=1:3, pch=rep(1,3))


# LOG-LINK
set.seed(314)
n <- 1000
shape <- 10
x.train <- as.matrix(rnorm(n, 0, 1))
mu <- exp(x.train) # = shape * scale
scale <- mu / shape

#y.train <- sapply(1:n, function(i){rgamma(1, shape=shape, scale=scale[i])})
y.train <- rgamma(n, shape=shape, scale=scale)
plot(x.train, y.train, main="Gamma glm observations, log-link")
points(x.train, mu, col=2)

mod <- gbt.train(y.train, x.train, loss_function = "gamma::log", verbose=50)
pred.mu <- predict(mod, x.train) 
points(x.train, pred.mu, col=3)
legend("topleft", c("data", "true mean", "gbt estimated mean"), col=1:3, pch=rep(1,3))
