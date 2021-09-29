## Example illustrating gbt.complexity
library(agtboost)
n <- 10000
xtr <- as.matrix(runif(n, 0, 4))
ytr <- rnorm(n, xtr, 1)
xte <- as.matrix(runif(n, 0, 4))
yte <- rnorm(n, xte, 1)

model <- gbt.train(ytr, xtr, learning_rate = 0.1)
gbt.complexity(model, type="xgboost")
agtb_pred <- predict(model, xte)

plot(xte, yte)
points(xte, agtb_pred, col=2)

# xgboost
library(xgboost)
xgb_param <- gbt.complexity(model, type="xgboost")
xgb_param$verbose=1
dtrain_xgb <- xgb.DMatrix(xtr, label=ytr)
dtest_xgb <- xgb.DMatrix(xte)
xgb_model <- xgb.train(xgb_param, dtrain_xgb, nrounds=xgb_param$nrounds)
xgb_pred <- predict(xgb_model, dtest_xgb)
points(xte, xgb_pred, col=3)

# lightgbm
library(lightgbm)
lgb_param <- gbt.complexity(model, type="lightgbm")
dtrain_lgb <- lgb.Dataset(data = xtr, label = xte)
dtest_lgb <- lgb.Dataset(data=xte)
lgb_model <- lgb.train(data=dtrain_lgb, params=lgb_param, nrounds=lgb_param$nrounds)
lgb_pred <- predict(lgb_model, xte)
points(xte, lgb_pred, col=4)

