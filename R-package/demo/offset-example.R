## Example illustrating offset
library(agtboost)
n <- 1000
xtr <- as.matrix(runif(n, 0, 4))
ytr <- rnorm(n, xtr, 1)
xte <- as.matrix(runif(n, 0, 4))
yte <- rnorm(n, xte, 1)

model <- gbt.train(ytr, xtr, learning_rate = 0.1, verbose=1)
model <- gbt.train(ytr, xtr, learning_rate = 0.1, verbose=1, offset=0.5*xtr)
model <- gbt.train(ytr, xtr, learning_rate = 0.1, verbose=1, offset=1.0*xtr)
gbt.complexity(model, type="xgboost")
agtb_pred <- predict(model, xte)

# Use-case: Poisson
xtr <- as.matrix(runif(n, 0, 4))
ztr <- runif(n) 
ytr <- rpois(n, ztr*xtr)
xte <- as.matrix(runif(n, 0, 4))
yte <- rpois(n, xte) # test for z=1

par(mfrow=c(1,2))
plot(xte, yte)
plot(xtr*ztr, ytr)
par(mfrow=c(1,1))

agtb <- gbt.train(ytr, xtr, loss_function = "poisson", offset=log(ztr), verbose=100)

library(xgboost)
dtrain <- xgb.DMatrix(xtr, label=ytr)
attr(dtrain, 'exposure') <- ztr

poisson_obj_fun <- function(preds, dtrain) {
  exposure <- attr(dtrain, "exposure")
  y <- getinfo(dtrain, "label")
  preds_adj <- preds + log(exposure)
  grad <- exp(preds_adj) - y
  hess <- exp(preds_adj)
#  grad <- exposure*exp(preds) - y*log(exposure)
#  hess <- exposure*exp(preds)
  return(list(grad = grad, hess = hess))
}

poisson_eval <- function(preds, dtrain){
  exposure <- attr(dtrain, "exposure")
  y <- getinfo(dtrain, "label")
  preds_adj <- preds + log(exposure)
  nll <- -mean(dpois(y, lambda=exp(preds_adj), log=TRUE))
  return(list(metric="poisson-nll", value=err))
}

param <- gbt.complexity(agtb, type="xgboost")
param$objective <- poisson_obj_fun
param$eval_metric <- poisson_eval
param$verbosity <- 1
xgb_model <- xgb.train(param, dtrain, nrounds=param$nrounds, print_every_n=1)
xgb_pred <- predict(xgb_model, xte)

# Check results
plot(xte, yte)
points(xte, predict(agtb, xte), col=2)
points(xte, exp(xgb_pred), col=3)
