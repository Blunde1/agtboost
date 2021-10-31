## Example illustrating gbt.complexity
set.seed(123)
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
colnames(xtr) <- colnames(xte) <- c("x1")
lgb_param <- gbt.complexity(model, type="lightgbm")
dtrain_lgb <- lgb.Dataset(data = xtr, label=ytr)
# Optional, train from predictions
#lightgbm::set_field(dtrain_lgb, "init_score", rep(lgb_param$init_score, dtrain_lgb$dim()[1]))
lgb_model <- lgb.train(data=dtrain_lgb, 
                       params=lgb_param, 
                       nrounds=lgb_param$nrounds,
                       obj=lgb_param$objective,
                       verbose=1)
lgb_pred <- predict(lgb_model, xte)
points(xte, lgb_pred, col=4)
# Optional if trained from predictions
# points(xte, lgb_param$init_score+lgb_pred, col=4)
