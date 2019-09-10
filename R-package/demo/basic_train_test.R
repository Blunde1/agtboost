# load gbtorch
library(gbtorch)

# generate data fun
generateX <- function(n, groups){runif(n,0,groups)}
generateY <- function(n, age, groups, sd){rnorm(n,1*age, sd)}


# One dim train and test
set.seed(14325)
n <- 500
groups <- 2^2
sd = 1
age <- generateX(n, groups)
x.train <- as.matrix(age,ncol=1) # matrix with one column feature
y.train <- generateY(n, age, groups, sd)
age.test <- generateX(n, groups)
x.test <- as.matrix(age.test, ncol=1)
y.test <- generateY(n, age.test, groups, sd)

# train model
param <- list("learning_rate" = 0.03, "loss_function" = "mse", "nrounds"=2000)
mod <- gbt.train(param, y.train, x.train )
mod$get_param()
pred.test <- predict( mod, x.test)
plot(age.test,y.test, main="Predictions on test-data")
points(age.test,pred.test,col=2)

# compare with a linear model that potentially holds the truth model
mod.lm <- lm(y~., data=data.frame(y=y.train, x=x.train))
pred.test.lm <- predict(mod.lm, newdata = data.frame(y=y.test, x=x.test))
points(age.test, pred.test.lm, col="blue")

# multidim and noisy
ndim = 99
df <- data.frame(age=age)
df.test <- data.frame(age = age.test)
for(i in 1:ndim){
    df[,i+1] <- rnorm(n, 0,1)
    df.test[,i+1] <- rnorm(n, 0,1)
}
x.train2 <- as.matrix(df)
x.test2 <- as.matrix(df.test)
dim(x.train2); dim(x.test2)

# build model on multidim designmatrix
mod2 <- gbt.train(param, y.train, x.train2)
pred.test2 <- predict( mod2, x.test2 )
points(age.test, pred.test2, col=3)

# compare with a second linear model that potentially holds the truth model
library(scales)
mod.lm2 <- lm(y~., data=data.frame(y=y.train, x.train2))
pred.test.lm2 <- predict(mod.lm2, newdata = data.frame(y=y.test, x.test2))
points(age.test, pred.test.lm2, col=alpha("blue",0.5))
