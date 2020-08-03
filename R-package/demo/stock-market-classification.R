# Libraries
library(agtboost)
library(xgboost)
library(randomForest)
library(ISLR) # Contains the "Smarket" data


# -- Data management --
data("Smarket")
# Description
# Daily percentage returns for the S&P 500 stock index between 2001 and 2005.
# Goal: predict "Direction" Up or Down for the market return
# Variables: Percentage return on 1-5 last days, volume

# Remove return today and year
dim(Smarket)

# Out of time validation
ind_train <- which(Smarket$Year <= 2004)

# Remove all-knowing features
Smarket <- subset(Smarket, select=-c(Today, Year))

# One-hot encoding
data <- model.matrix(Direction~., data=Smarket)[,-1]
dim(data)

# Split into train and test datasets
x.train <<- as.matrix(data[ind_train, ])
y.train <<- as.matrix(ifelse(Smarket[ind_train, "Direction"]=="Up",1,0))
x.test <<- as.matrix(data[-ind_train, ])
y.test <<- as.matrix(ifelse(Smarket[-ind_train, "Direction"]=="Up", 1, 0))


# -- Model building --
# agtboost
gbt.mod <- gbt.train(y.train, x.train, learning_rate = 0.01, loss_function = "logloss")
gbt.mod$get_num_trees()

# glm
glm.mod <- glm(y~., data=data.frame(y=y.train, x.train), family=binomial)

#rf
rf.mod <- randomForest(y~., data=data.frame(y=as.factor(y.train), x.train))

#xgb
p.xgb <- list(eta = 0.01, lambda=0, objective="binary:logistic", eval_metric = "logloss") # lambda=0, vanilla 2'nd order gtb
x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
xgb.n <- xgb.cv(p.xgb, x.train.xgb, nround=2000, nfold=5, early_stopping_rounds =50, verbose = F)
xgb.nrounds <- which.min(xgb.n$evaluation_log$test_logloss_mean)
xgb.mod <- xgb.train(p.xgb, x.train.xgb, xgb.nrounds)


# -- Predictions and evaluations on test --
iegtb.pred <- predict(gbt.mod, x.test)
glm.pred <- predict(glm.mod, data.frame(y=y.test, x.test), type="response")
rf.pred <- predict(rf.mod, newdata = data.frame(y=y.test, x.test), type = "prob")[,2]
xgb.pred <- xgboost:::predict.xgb.Booster(xgb.mod, x.test)

LogLossBinary = function(y, x) {
    x = pmin(pmax(x, 1e-14), 1-1e-14)
    -sum(y*log(x) + (1-y)*log(1-x)) / length(y)
}

# Does the models beat the average constant prediction from train?
# Or do they overfit?
LogLossBinary(y.test, mean(y.train))
LogLossBinary(y.test, iegtb.pred) # score is before logistic transformation
LogLossBinary(y.test, glm.pred)
LogLossBinary(y.test, rf.pred)
LogLossBinary(y.test, xgb.pred)
