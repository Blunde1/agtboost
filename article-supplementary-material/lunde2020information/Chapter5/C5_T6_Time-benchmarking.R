# BENCHMARKING VS XGBOOST

library(MASS)
library(ISLR)
library(ElemStatLearn)
library(tree)
library(randomForest)
library(xgboost)
library(gbtorch)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# loss function
loss_mse <- function(y,y.hat){
    sum((y-y.hat)^2)
}
loss_binary_logistic <- function(y, x){
    sum(y*log(1+exp(-x)) + (1-y)*log(1+exp(x))) / length(y)
}
LogLossBinary = function(y, x) {
    x = pmin(pmax(x, 1e-14), 1-1e-14)
    -sum(y*log(x) + (1-y)*log(1-x)) / length(y)
}

# OJ: Purchase - Classification
data("OJ")
dim(OJ)
n_full <- nrow(OJ)
ind_train <- sample(n_full, 0.7*n_full)
data <- model.matrix(Purchase~., data=OJ)[,-1]
dim(data)
x.train <- as.matrix(data[ind_train, ])
y.train <- as.matrix(ifelse(OJ[ind_train, "Purchase"]=="MM",1,0))
x.test <- as.matrix(data[-ind_train, ])
y.test <- as.matrix(ifelse(OJ[-ind_train, "Purchase"]=="MM", 1, 0))

# Naive loss
LogLossBinary(y.test, mean(y.train))

# IEGTB
# LOGLOSS
param <- list("learning_rate" = 0.01, "loss_function" = "logloss", "nrounds"=5000)

train_iegtb <- function(){
    
    mod <<- gbt.train(y.train, x.train, learning_rate = 0.01, loss_function = "logloss", 
                     greedy_complexities = F)
    
    #mod <- new(ENSEMBLE)
    #mod$set_param(param)
    #mod$train(y.train, x.train)
    return(mod)
}

gbt_time <- system.time(train_iegtb())
iegtb.pred <- predict(mod, x.test)
gbt_loss <- loss_binary_logistic(y.test, iegtb.pred)


# XGB
p.xgb <- list(objective="binary:logistic", eval_metric = "logloss", 
              eta = 0.01, lambda=0, nthread = 1)
x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)


# XGB VAL
train.ind2 <- sample(nrow(x.train) , ceiling(0.7*nrow(x.train)))
dtrain.xgb.val <- xgb.DMatrix(data = as.matrix(x.train[train.ind2,]), label = y.train[train.ind2])
dval.xgb <- xgb.DMatrix(data = as.matrix(x.train[-train.ind2, ]), label = y.train[-train.ind2])
train_xgb_val <- function(){
    mod.xgb.val <<-  xgb.train(p.xgb, dtrain.xgb.val, nrounds = 2000, list(val = dval.xgb),
                              early_stopping_rounds = 50, verbose=10)    
}

xgb.val_time <- system.time(train_xgb_val())
pred.xgb.val <- xgboost:::predict.xgb.Booster(mod.xgb.val, x.test)
xgb.val_loss <- LogLossBinary(y.test, pred.xgb.val)

# XGB NROUNDS
train_xgb_nrounds <- function(){
    xgb.n <- xgb.cv(p.xgb, x.train.xgb, nround=2000, nfold=10, early_stopping_rounds =50, verbose = F)
    xgb.nrounds <- which.min(xgb.n$evaluation_log$test_logloss_mean)
    xgb.mod <<- xgb.train(p.xgb, x.train.xgb, xgb.nrounds)
    
}

xgb.nrounds_time <- system.time( train_xgb_nrounds() )
xgb.pred <- xgboost:::predict.xgb.Booster(xgb.mod, x.test)
xgb.nrounds_loss <- LogLossBinary(y.test, xgb.pred)


# REGULARISED


# XGB REG GAMMA
train_xgb_gamma <- function(p.xgb, x.train.xgb){
    p.xgb.reg <- p.xgb
    p.xgb.reg$max_depth <- 10
    gamma.grid <- seq(0,9,1)
    cv.score <- numeric(length(gamma.grid))
    cv.niter <- cv.score
    for(j in 1:length(gamma.grid)){
        cat("j iter: ", j, "\n")
        p.xgb.reg$gamma <- gamma.grid[j]
        cv.tmp <- xgb.cv(p.xgb.reg,
                         data = x.train.xgb, nround = 10000,
                         nfold = 10, prediction = TRUE, showsd = TRUE,
                         early_stopping_round = 50, maximize = FALSE, verbose=F)
        cv.score[j] <- cv.tmp$evaluation_log[,min(test_logloss_mean)]
        cv.niter[j] <- cv.tmp$best_iteration
    }
    rind <-  which(cv.score==min(cv.score), arr.ind = T)[1]
    p.xgb.reg$gamma <- gamma.grid[rind]
    p.xgb.reg$nrounds <- cv.niter[rind]
    mod.xgb.reg <<- xgb.train(p.xgb.reg, x.train.xgb, p.xgb.reg$nrounds)    
}

x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
xgb.gamma_time <- system.time( train_xgb_gamma(p.xgb, x.train.xgb) )
pred.xgb.reg.gamma <- xgboost:::predict.xgb.Booster(mod.xgb.reg, x.test)
xgb.gamma_loss <- LogLossBinary(y.test, pred.xgb.reg.gamma)


# XGB REG DEPTH
train_xgb_depth <- function(p.xgb, x.train.xgb){
    p.xgb.reg <- p.xgb
    depth.grid <- 1:10
    cv.score <- numeric(length(depth.grid))
    cv.niter <- cv.score
    for(j in 1:length(depth.grid)){
        cat("j iter: ", j, "\n")
        p.xgb.reg$max_depth <- depth.grid[j]
        cv.tmp <- xgb.cv(p.xgb.reg,
                         data = x.train.xgb, nround = 10000,
                         nfold = 10, prediction = TRUE, showsd = TRUE,
                         early_stopping_round = 50, maximize = FALSE, verbose=F)
        cv.score[j] <- cv.tmp$evaluation_log[,min(test_logloss_mean)]
        cv.niter[j] <- cv.tmp$best_iteration
    }
    rind <-  which(cv.score==min(cv.score), arr.ind = T)[1]
    p.xgb.reg$max_depth <- depth.grid[rind]
    p.xgb.reg$nrounds <- cv.niter[rind]
    mod.xgb.reg <<- xgb.train(p.xgb.reg, x.train.xgb, p.xgb.reg$nrounds)    
}

xgb.depth_time <- system.time( train_xgb_depth( p.xgb, x.train.xgb ) )
pred.xgb.depth <- xgboost:::predict.xgb.Booster(mod.xgb.reg, x.test)
xgb.depth_loss <- LogLossBinary(y.test, pred.xgb.depth)



# XGB REG GAMMA DEPTH
train_xgb_gamma_depth <- function( p.xgb, x.train.xgb ){
    p.xgb.reg <- p.xgb
    depth.grid <- 1:10
    gamma.grid <- seq(0,9,1)
    cv.score <- matrix(nrow=length(gamma.grid), ncol = length(depth.grid))
    cv.niter <- cv.score
    for(j in 1:length(depth.grid)){
        p.xgb.reg$max_depth <- depth.grid[j]
        for(k in 1:length(gamma.grid)){
            cat("j iter: ", j, "k iter: ", k, "\n")
            p.xgb.reg$gamma <- gamma.grid[j]
            cv.tmp <- xgb.cv(p.xgb.reg,
                             data = x.train.xgb, nround = 10000,
                             nfold = 10, prediction = TRUE, showsd = TRUE,
                             early_stopping_round = 50, maximize = FALSE, verbose=F)
            cv.score[k,j] <- cv.tmp$evaluation_log[,min(test_logloss_mean)]
            cv.niter[k,j] <- cv.tmp$best_iteration
        }
    }
    rind <- which(cv.score == min(cv.score), arr.ind = T)[1]
    cind <- which(cv.score == min(cv.score), arr.ind = T)[2]
    p.xgb.reg$gamma <- gamma.grid[rind]
    p.xgb.reg$max_depth <- depth.grid[cind]
    p.xgb.reg$nrounds <- cv.niter[rind, cind]
    mod.xgb.reg <<- xgb.train(p.xgb.reg, x.train.xgb, p.xgb.reg$nrounds)    
}

xgb.gamma.depth_time <- system.time( train_xgb_gamma_depth( p.xgb, x.train.xgb ) )
pred.xgb.depth.gamma <- xgboost:::predict.xgb.Booster(mod.xgb.reg, x.test)
xgb.gamma.depth_loss <- LogLossBinary(y.test, pred.xgb.depth.gamma)

gbt_loss
xgb.val_time
xgb.nrounds_time
xgb.gamma_time
xgb.depth_time
xgb.gamma.depth_time
benchmark_loss <- list(
    "benchmark_loss" = LogLossBinary(y.test,mean(y.train)),
    "gbt_loss" = gbt_loss,
    "xgb.val_loss" = xgb.val_loss,
    "xgb.nrounds_loss" = xgb.nrounds_loss,
    "xgb.gamma_loss" = xgb.gamma_loss,
    "xgb.depth_loss" = xgb.depth_loss,
    "xgb.gamma.depth_loss" = xgb.gamma.depth_loss
)
benchmark_times <- list(
    "gbt_time" = gbt_time,
    "xgb.val_time" = xgb.val_time,
    "xgb.nrounds_time" = xgb.nrounds_time,
    "xgb.gamma_time" = xgb.gamma_time,
    "xgb.depth_time" = xgb.depth_time,
    "xgb.gamma.depth_time" = xgb.gamma.depth_time
)
getwd()
if(F){
    save(benchmark_times, benchmark_loss, file="results/benchmark_times.RData")
    load("results/benchmark_times.RData")
}

for(i in 1:length(benchmark_times)){
    cat(
        paste0(
            format(benchmark_times[[i]][3], digits=3),
            " & "
        )
    )
}
for(i in 1:length(benchmark_times)){
    cat(
        paste0(
            format(benchmark_loss[[i+1]], digits=4),
            " & "
        )
    )
}
