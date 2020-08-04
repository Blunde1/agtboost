# Comparisons on ISLR data
# Simulated splits: distributions
# Normalized to mean XGB = 1


library(MASS)
library(ISLR)
library(ElemStatLearn)
library(tree)
library(randomForest)
library(xgboost)
library(gbm)
library(ggplot2)
library(gbtorch)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# datasets
dataset <- function(i, seed)
{
    
    # Returns training and test datasets for x and y
    # i = 1:7 returns mse datasets
    # i = 8:12 returns logloss datasets
    # i = 1: Boston
    # i = 2: ozone
    # i = 3: Auto
    # i = 4: Carseats
    # i = 5: College
    # i = 6: Hitters
    # i = 7: Wage
    # i = 8: Caravan
    # i = 9: Default
    # i = 10: OJ
    # i = 11: Smarket
    # i = 12: Weekly
    
    # reproducibility
    set.seed(seed) 
    
    if(i==1){
        # boston
        data(Boston)
        medv_col <- which(colnames(Boston)=="medv")
        n_full <- nrow(Boston)
        ind_train <- sample(n_full, 0.5*n_full)
        x.train <<- as.matrix(Boston[ind_train,-medv_col])
        y.train <<- as.matrix(log(Boston[ind_train,medv_col]))
        x.test <<- as.matrix(Boston[-ind_train,-medv_col])
        y.test <<- as.matrix(log(Boston[-ind_train,medv_col]))
        
    }else 
        if(i==2){
        # ozone
        data(ozone)
        n_full <- nrow(ozone)
        ind_train <- sample(n_full, 0.5*n_full)
        x.train <<- as.matrix(log(ozone[ind_train,-1]))
        y.train <<- as.matrix(log(ozone[ind_train,1]))
        x.test <<- as.matrix(log(ozone[-ind_train,-1]))
        y.test <<- as.matrix(log(ozone[-ind_train,1]))
        
    }else 
        if(i==3){
        
        # auto
        data("Auto")
        dim(Auto)
        mpg_col <- which(colnames(Auto)=="mpg")
        n_full <- nrow(Auto)
        data <- model.matrix(mpg~.,data=Auto)[,-1]
        dim(data)
        ind_train <- sample(n_full, 0.5*n_full)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(Auto[ind_train, mpg_col])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(Auto[-ind_train, mpg_col])
    }else 
        if(i==4){
        # Carseats - sales - mse
        data("Carseats")
        dim(Carseats)
        Carseats =na.omit(Carseats)
        dim(Carseats)
        sales_col <- which(colnames(Carseats)=="Sales")
        n_full <- nrow(Carseats)
        data <- model.matrix(Sales~., data=Carseats)[,-1]
        dim(data)
        ind_train <- sample(n_full, 0.7*n_full)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(Carseats[ind_train, sales_col])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(Carseats[-ind_train, sales_col])
        
    }else 
        if(i==5){
        
        # College - apps: applications received - mse
        data("College")
        dim(College)
        n_full <- nrow(College)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Apps~., data=College)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(College[ind_train, "Apps"])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(College[-ind_train, "Apps"])
        
    }else 
        if(i==6){
        # Hitters: Salary - mse
        data("Hitters")
        dim(Hitters)
        Hitters =na.omit(Hitters)
        dim(Hitters)
        n_full <- nrow(Hitters)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Salary~., data=Hitters)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(Hitters[ind_train, "Salary"])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(Hitters[-ind_train, "Salary"])
        
    }else 
        if(i==7){
        # Wage - Wage - mse -- note: extremely deep trees!
        data(Wage)
        dim(Wage)
        n_full <- nrow(Wage)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(wage~., data=Wage)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(Wage[ind_train, "wage"])
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(Wage[-ind_train, "wage"])
        
    }else 
        if(i==8){
        # Caravan - classification
        data(Caravan)
        dim(Caravan)
        Caravan = na.omit(Caravan)
        dim(Caravan)
        n_full <- nrow(Caravan)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Purchase~., data=Caravan)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(ifelse(Caravan[ind_train, "Purchase"]=="Yes",1,0))
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(ifelse(Caravan[-ind_train, "Purchase"]=="Yes", 1, 0))
        
    }else 
        if(i==9){
        # Default - Default: if default on credit - classification
        data("Default")
        dim(Default)
        n_full <- nrow(Default)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(default~., data=Default)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(ifelse(Default[ind_train, "default"]=="Yes",1,0))
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(ifelse(Default[-ind_train, "default"]=="Yes", 1, 0))
        
    }else 
        if(i==10){
        # OJ: Purchase - Classification
        data("OJ")
        dim(OJ)
        n_full <- nrow(OJ)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Purchase~., data=OJ)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(ifelse(OJ[ind_train, "Purchase"]=="MM",1,0))
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(ifelse(OJ[-ind_train, "Purchase"]=="MM", 1, 0))
        
    }else 
        if(i==11){
        # Smarket : classification
        data("Smarket")
        dim(Smarket)
        Smarket <- subset(Smarket, select=-c(Today, Year))
        n_full <- nrow(Smarket)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Direction~., data=Smarket)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(ifelse(Smarket[ind_train, "Direction"]=="Up",1,0))
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(ifelse(Smarket[-ind_train, "Direction"]=="Up", 1, 0))
        
    }else 
        if(i==12){
        # Weekly - classification
        data("Weekly")
        dim(Weekly)
        Weekly <- subset(Weekly, select=-c(Today, Year))
        n_full <- nrow(Weekly)
        ind_train <- sample(n_full, 0.7*n_full)
        data <- model.matrix(Direction~., data=Weekly)[,-1]
        dim(data)
        x.train <<- as.matrix(data[ind_train, ])
        y.train <<- as.matrix(ifelse(Weekly[ind_train, "Direction"]=="Up",1,0))
        x.test <<- as.matrix(data[-ind_train, ])
        y.test <<- as.matrix(ifelse(Weekly[-ind_train, "Direction"]=="Up", 1, 0))
        
    }
}


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



# Seeds
set.seed(14912)
B <- 100
seeds <- sample(1e5, B) 
methods <- 6
param <- list("learning_rate" = 0.1, "loss_function" = "mse", "nrounds"=5000)
param.tree <- list("learning_rate" = 1, "loss_function" = "mse", "nrounds"=1)

# Store results
res <- list() # store matrices with results
for(i in 1:7){
    cat("iter: ", i,"\n")
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    res_mat <- matrix(nrow=B, ncol=methods)
    for(b in 1:B){
        
        # generate data
        dataset(i, seeds[b])
        
        # iegtb
        #gtb
        mod <- gbt.train(y.train, x.train, param$learning_rate, nrounds = param$nrounds,
                         greedy_complexities = F, verbose=0)
        iegtb.pred <- predict(mod, x.test)
        #mod <- new(ENSEMBLE)
        #mod$set_param(param)
        #mod$train(y.train, x.train)
        #iegtb.pred <- mod$predict(x.test)

        # iegt - one tree
        mod.tree <- gbt.train(y.train, x.train, param.tree$learning_rate, nrounds = param.tree$nrounds,
                              verbose = 0, greedy_complexities = F)
        iegt.pred <- predict(mod.tree, x.test)
        
        #mod.tree <- new(ENSEMBLE)
        #mod.tree$set_param(param.tree)
        #mod.tree$train(y.train, x.train)
        #iegt.pred <- mod.tree$predict(x.test)

        # regression tree
        tree.mod <- tree(y~., data=data.frame(y=y.train, x.train))
        tree.mod.cv=cv.tree(tree.mod)
        tree.pruned=prune.tree(tree.mod,best=which.min(rev(tree.mod.cv$dev)))
        tree.pred <- predict(tree.pruned,newdata=data.frame(y=y.test, x.test))

        # lm
        lm.mod <- lm(y~., data=data.frame(y=y.train, x.train))
        lm.pred <- predict(lm.mod, data.frame(y=y.test, x.test))
        
        #rf
        rf.mod <- randomForest(y~., data=data.frame(y=y.train, x.train))
        rf.pred <- predict(rf.mod, newdata = data.frame(y=y.test, x.test))
        
        #gbm
        if(F){
            gbm.mod <- gbm(y~., data=data.frame(y=y.train, x.train), distribution="gaussian", 
                           shrinkage = 0.01, cv.folds = 5, n.trees = 2*mod$get_num_trees())
            best.iter <- gbm.perf(gbm.mod, method = "cv") # 5-fold
            print(best.iter)
            gbm.pred <- predict(gbm.mod, newdata = data.frame(y=y.test, x.test), n.trees = best.iter)
            
        }

        #xgb
        p.xgb <- list(eta = 0.1, lambda=0, nthread = 1)
        x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
        xgb.n <- xgb.cv(p.xgb, x.train.xgb, nround=2000, nfold=10, early_stopping_rounds =10, verbose = F)
        xgb.nrounds <- which.min(xgb.n$evaluation_log$test_rmse_mean)
        xgb.mod <- xgb.train(p.xgb, x.train.xgb, xgb.nrounds)
        xgb.pred <- xgboost:::predict.xgb.Booster(xgb.mod, x.test)
        
        # check
        if(F)
        {
            loss_mse(y.test, iegtb.pred)
            loss_mse(y.test, iegt.pred)
            loss_mse(y.test, tree.pred)
            loss_mse(y.test, lm.pred)
            loss_mse(y.test, rf.pred)
            #loss_mse(y.test, gbm.pred)
            loss_mse(y.test, xgb.pred)
            
        }
        
        # update res matrice
        res_mat[b, 1] <- loss_mse(y.test, iegtb.pred)
        res_mat[b, 2] <- loss_mse(y.test, iegt.pred)
        res_mat[b, 3] <- loss_mse(y.test, tree.pred)
        res_mat[b, 4] <- loss_mse(y.test, lm.pred)
        res_mat[b, 5] <- loss_mse(y.test, rf.pred)
        res_mat[b, 6] <- loss_mse(y.test, xgb.pred)
        
        # udpate progress bar
        setTxtProgressBar(pb, b)
    }
    res[[i]] <- res_mat
    save(res, file = "results/res_list2.RData")
    close(pb)
}

for(i in 1:7){
    for(j in 1:6){
        cat( mean(res[[i]][ , j]), " (", sd(res[[i]][ , j]), ")  & " )
    }
    cat("\\ ","\n")
}

mxgb <- sapply(1:7, function(i) mean(res[[i]][,6]))
datasets <- c("Boston", "Ozone", "Auto", "Carseats", "College", "Hitters", "Wage")
order <- c(6,1,5,4,  3,2)
    
for(i in 1:7){
    cat(datasets[i], " & ")
    for(j in 1:6){
        cat( paste0(
          format(  mean(res[[i]][ , order[j]] / mxgb[i]) , digits=3 ), " (", 
          format(  sd(res[[i]][ , order[j]]/ mxgb[i]) , digits=3 ), ")  & "
        ) )
    }
    cat("\\\\ ","\n")
}
cat("\\hline \n")


# LOGLOSS
param <- list("learning_rate" = 0.1, "loss_function" = "logloss", "nrounds"=5000)

# Store results
load("results/res_list2.RData") # store matrices with results
for(i in 8:12){
    cat("iter: ", i,"\n")
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    res_mat <- matrix(nrow=B, ncol=methods)
    for(b in 1:B){
        
        # generate data
        dataset(i, seeds[b])
        
        # iegtb
        #gtb
        mod <- gbt.train(y.train, x.train, loss_function = "logloss", greedy_complexities=F, learning_rate = 0.01, 
                         verbose = 0)
        iegtb.pred <- predict(mod, x.test)
        #mod <- new(ENSEMBLE)
        #mod$set_param(param)
        #mod$train(y.train, x.train)
        #iegtb.pred <- mod$predict(x.test)
        
        # glm
        glm.mod <- glm(y~., data=data.frame(y=y.train, x.train), family=binomial)
        glm.pred <- predict(glm.mod, data.frame(y=y.test, x.test), type="response")
        
        #rf
        rf.mod <- randomForest(y~., data=data.frame(y=as.factor(y.train), x.train))
        rf.pred <- predict(rf.mod, newdata = data.frame(y=y.test, x.test), type = "prob")[,2]
        
        #xgb
        p.xgb <- list(eta = 0.1, lambda=0, objective="binary:logistic", eval_metric = "logloss")
        x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
        xgb.n <- xgb.cv(p.xgb, x.train.xgb, nround=2000, nfold=5, early_stopping_rounds =10, verbose = F)
        xgb.nrounds <- which.min(xgb.n$evaluation_log$test_logloss_mean)
        xgb.mod <- xgb.train(p.xgb, x.train.xgb, xgb.nrounds)
        xgb.pred <- xgboost:::predict.xgb.Booster(xgb.mod, x.test)
        
        # update res matrice
        res_mat[b, 1] <- LogLossBinary(y.test, 1/(1+exp(-iegtb.pred)))
        #res_mat[b, 2] <- loss_mse(y.test, iegt.pred)
        #res_mat[b, 3] <- loss_mse(y.test, tree.pred)
        res_mat[b, 2] <- LogLossBinary(y.test, glm.pred)
        res_mat[b, 3] <- LogLossBinary(y.test, rf.pred)
        res_mat[b, 4] <- LogLossBinary(y.test, xgb.pred)
        
        # udpate progress bar
        setTxtProgressBar(pb, b)
    }
    res[[i]] <- res_mat
    save(res, file = "results/res_list2.RData")
    close(pb)
}


mxgb <- sapply(1:12, function(i) mean(res[[i]][,4]))
datasets <- c("Boston", "Ozone", "Auto", "Carseats", "College", "Hitters", "Wage",
              "Caravan", "Default", "OJ", "Smarket", "Weekly")
order2 <- c(4,1,3,2)
for(i in 8:12){
    cat(datasets[i], " & ")
    for(j in 1:4){
        cat( paste0(
            format(  mean(res[[i]][ , order2[j]] / mxgb[i]) , digits=3 ), " (", 
            format(  sd(res[[i]][ , order2[j]]/ mxgb[i]) , digits=3 ), ")  & "
        ) )
    }
    cat("\\\\ ","\n")
}

