# P >> N Comparison to lm(), Ridge and Lasso regression (glmnet) and xgboost
# 15.10.2019
# Berent Lunde

# load gbtorch and other libraries needed
library(gbtorch)
library(glmnet)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(gridExtra)
library(xgboost)
library(tidyr)
library(latex2exp)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

theme_set(theme_bw(base_size = 12))

#MSE Loss and derivatives
loss <- function(y,y.hat){
    sum((y-y.hat)^2) / length(y)
}

train_model <- function(y.train, x.train, y.test, x.test, model){
    
    n <- length(y.train)
    res <- list()
    start <- end <- 0.0
    test_pred <- numeric(n)
    ntrees <- NULL
    
    if(model=="lm"){
        
        # lm model
        start <- Sys.time()
        mod <- lm(y~., data=data.frame(y=y.train, x=x.train))
        end <- Sys.time()
        test_pred <- predict(mod, newdata = data.frame(y=y.test, x=x.test))

    }else if(model=="ridge"){
        
        # ridge
        start <- Sys.time()
        mod.ridge <- cv.glmnet(y=y.train, x=x.train, alpha=0, nfolds=10, lambda=exp(seq(-2,5,length.out=100)))
        end <- Sys.time()
        test_pred <- predict(mod.ridge, x.test, s=mod.ridge$lambda.min)
        
    }else if(model=="lasso"){
        
        # lasso
        start <- Sys.time()
        mod.lasso <- cv.glmnet(y=y.train, x=x.train, alpha=1, nfolds=10, lambda=exp(seq(-5,3,length.out=100)))
        end <- Sys.time()
        test_pred <- predict(mod.lasso, x.test, s=mod.lasso$lambda.min)
        
    }else if(model=="gbtorch"){
        
        #gbtorch
        start <- Sys.time()
        mod <- gbt.train(y.train, x.train, greedy_complexities=F, learning_rate = 0.01, verbose=1)
        end <- Sys.time()
        test_pred <- predict( mod, x.test)
        ntrees <- mod$get_num_trees()
        
    }else if(model=="xgboost"){
        
        #xgboost
        grid.hpar <- expand.grid(
            #gamma = c(0.5, 1, 3),
            #max_depth = c(1,3,6),
            #min_child_weight = c(1,5,10),
            lambda = c(0),
            eta = c(0.01),
            nthread = c(1)
        )
        x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
        start <- Sys.time()
        hpar.cv <- apply(grid.hpar, 1, function(parameterList){
            
            #Extract Parameters to test
            p.xgb.cv <- list(
                objective="reg:squarederror", 
                eval_metric = "rmse",
                #gamma = parameterList[["gamma"]],
                #max_depth = parameterList[["max_depth"]],
                #min_child_weight = parameterList[["min_child_weight"]],
                lambda = parameterList[["lambda"]],
                eta = parameterList[["eta"]],
                nthread = parameterList[["nthread"]]
            )
            cv.tmp <- xgb.cv(p.xgb.cv, data=x.train.xgb, nround=2000, nfold=10, early_stopping_rounds =10, verbose = F)
            cv.score <- cv.tmp$evaluation_log[,min(test_rmse_mean)]
            cv.niter <- cv.tmp$best_iteration
            return(c("cv.score"=cv.score, "nrounds"=cv.niter, unlist(p.xgb.cv)[-c(1,2)]))
            
        })
        
        cvnames <- rownames(hpar.cv)[-1]
        par <- as.numeric( hpar.cv[,which.min(hpar.cv[1,])][-1] )
        names(par) <- cvnames
        p.xgb <- as.list(par)
        p.xgb$objective="reg:squarederror"
        p.xgb$eval_metric = "rmse"
        xgb.mod <- xgb.train( p.xgb, x.train.xgb, as.numeric(p.xgb$nrounds) )
        end <- Sys.time()
        test_pred <- predict(xgb.mod, x.test)
        ntrees <- as.numeric(p.xgb$nrounds)

    }else if(model=="xgboost::val"){
        
        #xgboost validation 30%
        p.xgb <- list(objective="reg:squarederror", eval_metric = "rmse", eta = 0.01, lambda=0, nthread = 1)
        idx.train <- sample(nrow(x.train) , ceiling(0.7*nrow(x.train)))
        dtrain.xgb.val <- xgb.DMatrix(data = as.matrix(x.train[idx.train,]), label = y.train[idx.train])
        dval.xgb <- xgb.DMatrix(data = as.matrix(x.train[-idx.train, ]), label = y.train[-idx.train])
        start <- Sys.time()
        xgb.mod <- xgb.train(p.xgb, dtrain.xgb.val, nrounds=2000, list(val=dval.xgb), 
                             early_stopping_rounds=10, verbose=F)
        end <- Sys.time()
        test_pred <- predict(xgb.mod, x.test)
        ntrees <- xgb.mod$best_iteration
        
    }
    
    res$time <- as.numeric(end - start, units="secs")
    res$test_pred <- test_pred
    res$loss <- loss(y.test, test_pred)
    res$ntrees <- ntrees
    # res num trees
    
    return(res)
    
}

# One dimensional case ####

# GENERATE DATA ####
set.seed(314)
cat("Generate data")
n <- 1000
x.train <- as.matrix( runif(n, 0, 4))
y.train <- rnorm(n, x.train[,1], 1)
x.test <- as.matrix( runif(n, 0, 4))
y.test <- rnorm(n, x.test[,1], 1)

# dependence
mdim = 9999
x.train2 <- x.train
x.test2 <- x.test
pb <- txtProgressBar(min = 0, max = mdim, style = 3)
for(i in 1:mdim){
    x.train2 <- cbind(x.train2, (mdim-i)/mdim *  x.train2[,i] + i/mdim * rnorm(n, 0, 1) )
    x.test2 <- cbind(x.test2, (mdim-i)/mdim *  x.test2[,i] + i/mdim * rnorm(n, 0, 1) )
    setTxtProgressBar(pb, i)
}
close(pb)

# independence
x.train3 <- x.train
x.test3 <- x.test
x.train3 <- cbind(x.train, matrix(rnorm(n*mdim), nrow=n, ncol=mdim))
x.test3 <- cbind(x.test, matrix(rnorm(n*mdim), nrow=n, ncol=mdim))

# dependence plots
if(F){
    cat("The dimensions of training and test:")
    dim(x.train2); dim(x.test2) 
    
    cat("The data is not independent, but noisy to a different degree")
    df_sub <- data.frame(x1=x.train2[,1], x50=x.train2[,50], x100=x.train2[,100],
                         x200=x.train2[,200], x500=x.train2[,500], 
                         x1000=x.train2[,1000], x10000=x.train2[,10000])
    pairs(~., data=df_sub)
    
    gather(df_sub, group, value, -x1) %>%
        ggplot() + 
        geom_point(aes(x=x1, y=value, colour=group, shape=group), size=1) + 
        xlab("Relevant feature") +
        ylab("Other features") + 
        scale_shape_manual(name = NULL,
                           values = c(14:20)) + 
        #scale_color_manual(name=NULL, 
        #                   values=cbp2) +
        theme_minimal(base_size = 12) 
    
}


# TRAINING ####

loss_base <- loss(y.test, rep(mean(y.train),length(y.test)))

# m=1
lm_res1 <- train_model(y.train, x.train, y.test, x.test, "lm")
gbt_res1 <- train_model(y.train, x.train, y.test, x.test, "gbtorch")
xgb_res1 <- train_model(y.train, x.train, y.test, x.test, "xgboost")
xgbv_res1 <- train_model(y.train, x.train, y.test, x.test, "xgboost::val")


# m=10000 -- dependence
lm_res2 <- train_model(y.train, x.train2, y.test, x.test2, "ridge")
gbt_res2 <- train_model(y.train, x.train2, y.test, x.test2, "gbtorch")
xgb_res2 <- train_model(y.train, x.train2, y.test, x.test2, "xgboost")
xgbv_res2 <- train_model(y.train, x.train2, y.test, x.test2, "xgboost::val")

# m=10000 -- independence
lm_res3 <- train_model(y.train, x.train3, y.test, x.test3, "lasso")
gbt_res3 <- train_model(y.train, x.train3, y.test, x.test3, "gbtorch")
xgb_res3 <- train_model(y.train, x.train3, y.test, x.test3, "xgboost")
xgbv_res3 <- train_model(y.train, x.train3, y.test, x.test3, "xgboost::val")

results1 <- list(lm_res1, gbt_res1, xgb_res1, xgbv_res1)
results2 <- list(lm_res2, gbt_res2, xgb_res2, xgbv_res2)
results3 <- list(lm_res3, gbt_res3, xgb_res3, xgbv_res3)

if(F){
    save(results1, results2, results3, file="results/linear_model_sim_results.RData")
    load("results/linear_model_sim_results.RData")
}
lm_res1 <- results1[[1]]
gbt_res1 <- results1[[2]]
xgb_res1 <- results1[[3]]
xgbv_res1 <- results1[[4]]

lm_res2 <- results2[[1]]
gbt_res2 <- results2[[2]]
xgb_res2 <- results2[[3]]
xgbv_res2 <- results2[[4]]

lm_res3 <- results3[[1]]
gbt_res3 <- results3[[2]]
xgb_res3 <- results3[[3]]
xgbv_res3 <- results3[[4]]

# TABLE 3
for(i in 1:4){
    if(i==1){
        cat("linear model & ")
    }else if(i==2){
        cat("gbtorch & ")
    }else if(i==3){
        cat("xgboost: cv & ")
    }else if(i==4){
        cat("xgboost: val & ")
    }
    cat(
        paste0(
            
            format(results1[[i]]$loss, digits=3),
            " & ",
            results1[[i]]$ntrees, 
            " & ",
            format(results1[[i]]$time, digits=3),
            " & ",
            format(results3[[i]]$loss, digits=3),
            " & ",
            results3[[i]]$ntrees, 
            " & ",
            format(results3[[i]]$time, digits=3),
            " & ",
            format(results2[[i]]$loss, digits=3),
            " & ",
            results2[[i]]$ntrees, 
            " & ",
            format(results2[[i]]$time, digits=3),
            " \n"
            
        )
    )
}


# FIT PLOTS
# PLOT1: 1D - TRUTH + LM + GBTORCH + XGB
cols <- c("test response"="#000000", 
          "linear model"="#56B4E9",  # light blue
          "lasso" = "#56B4E9",
          "ridge" = "#56B4E9",
          "agtboost" = "#E69F00", # orange
          "xgboost: cv" = "#009E73", # green?
          "xgboost: val" = "#F0E442"
        
)

df_1d <- data.frame(
    predictive_feature = x.test,
    #test_obs = y.test,
    pred.lm = lm_res1$test_pred,
    agtboost = gbt_res1$test_pred,
    pred.xgb = xgb_res1$test_pred,
    pred.xgb.val = xgbv_res1$test_pred
)

df_md_dep <- data.frame(
    predictive_feature = x.test,
    #test_obs = y.test,
    pred.ridge = as.vector(lm_res2$test_pred),
    agtboost = gbt_res2$test_pred,
    pred.xgb = xgb_res2$test_pred, # create
    pred.xgb.val = xgbv_res2$test_pred
)

df_md_indep <- data.frame(
    predictive_feature = x.test,
    #test_obs = y.test,
    pred.lasso = as.vector(lm_res3$test_pred),
    agtboost = gbt_res3$test_pred,
    pred.xgb = xgb_res3$test_pred, # create
    pred.xgb.val = xgbv_res3$test_pred
)

p1 <- df_1d %>%
    ggplot() +
    #geom_point(aes(predictive_feature, test_obs, colour="test response"), size=1, alpha=0.2) +
    geom_point(aes(predictive_feature, pred.xgb.val, colour="xgboost: val"), size=0.5) +
    geom_point(aes(predictive_feature, pred.xgb, colour="xgboost: cv"), size=0.5) +
    geom_point(aes(predictive_feature, pred.lm, colour="linear model"), size=0.5) +
    geom_point(aes(predictive_feature, agtboost, colour="agtboost"), size=0.5) +
    scale_color_manual(name=NULL, values=cols) +
    #xlab("$x_1$") +
    xlab(TeX("$$x_1$$")) + 
    ylab("Prediction") + 
    #ggtitle("Case 1 ($m=1$)") + 
    ggtitle(TeX("Case 1 $$(m=1)$$")) + #  "Case 2 (m=10000)") + 
    theme_bw(base_size = 12)
p1

p2 <- df_md_indep %>%
    ggplot() +
    #geom_point(aes(predictive_feature, test_obs, colour="test response"), size=1, alpha=0.2) +
    geom_point(aes(predictive_feature, pred.xgb.val, colour="xgboost: val"), size=0.5) +
    geom_point(aes(predictive_feature, pred.xgb, colour="xgboost: cv"), size=0.5) +
    geom_point(aes(predictive_feature, pred.lasso, colour="linear model"), size=0.5) +
    geom_point(aes(predictive_feature, agtboost, colour="agtboost"), size=0.5) +
    scale_color_manual(name=NULL, values=cols) +
    #xlab("$x_1$") +
    xlab(TeX("$$x_1$$")) + 
    ylab("Prediction") + 
    ggtitle(TeX("Case 2 $$(m=10000)$$")) + #  "Case 2 (m=10000)") + 
    theme_bw(base_size = 12)
p2

p3 <- df_md_dep %>%
    ggplot() +
    #geom_point(aes(predictive_feature, test_obs, colour="test response"), size=1, alpha=0.2) +
    geom_point(aes(predictive_feature, pred.xgb.val, colour="xgboost: val"), size=0.5) +
    geom_point(aes(predictive_feature, pred.xgb, colour="xgboost: cv"), size=0.5) +
    geom_point(aes(predictive_feature, pred.ridge, colour="linear model"), size=0.5) +
    geom_point(aes(predictive_feature, agtboost, colour="agtboost"), size=0.5) +
    scale_color_manual(name=NULL, values=cols) +
    #xlab("$x_1$ ") +
    xlab(TeX("$$x_1$$")) + 
    ylab("Prediction") + 
    #ggtitle("Case 3 ($m=10000$)") + 
    ggtitle(TeX("Case 3 $$(m=10000)$$")) + #  "Case 2 (m=10000)") + 
    theme_bw(base_size = 12)
p3         

plots <- ggarrange(p1, p2, p3, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
plots

if(F){
    #pdf("../../../../../gbtree_information/figures/boost_sim_lm_plots.pdf", width=8, height=4, paper="special")
    print(plots)
    #dev.off()
    
    # TIKZ
    #library(tikzDevice)
    #options(tz="CA")
    #tf <- "../../../gbtree_information/figures/boost_sim_lm_plots.tex"
    #tikz(file = tf, width = 6.5, height = 3.5, standAlone=TRUE)
    print(plots)
    #dev.off()
    
    #currentwd <- getwd()
    #setwd("~/Projects/Github repositories/gbtree_information/figures")
    # DO IN CMD!!!! Generates pdf
    #system("lualatex boost_sim_lm_plots.tex")
    
    #setwd(currentwd)
    
    # view output
    #tools::texi2dvi(tf,pdf=TRUE)
    #system(paste(getOption('pdfviewer'),file.path(td,'example3.pdf')))
    
}
