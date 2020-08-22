library(data.table)
library(agtboost)
library(pROC)
library(xgboost)
library(ggplot2)
library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

logloss = function(y, x) {
    x = pmin(pmax(x, 1e-14), 1-1e-14)
    -sum(y*log(x) + (1-y)*log(1-x)) / length(y)
}

setwd("~/Projects/Github repositories/gbtorch-package-article/gbtorch-package-article-scripts")
hdata <- fread("data/HIGGS.csv/HIGGS.csv")
xtr <- as.matrix(hdata[1:10000000,-1])
ytr <- as.matrix(hdata[1:10000000,1])
xte <- as.matrix(hdata[-(1:10000000),-1])
yte <- as.matrix(hdata[-(1:10000000),1])
rm(hdata);gc()

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Store results from run
nsizes <- 10^(2:5)
time_vanilla <- time_gsub <- time_xgb <- numeric(length(nsizes))
loss_vanilla <- loss_gsub <- loss_xgb <- numeric(length(nsizes))
auc_vanilla <- auc_gsub <- numeric(length(nsizes))
ntrees_vanilla <- ntrees_gsub <- numeric(length(nsizes))
nleaves_vanilla <- nleaves_gsub <- numeric(length(nsizes))
nfeatures_vanilla <- nfeatures_gsub <- numeric(length(nsizes))
ks_stat_vanilla <- ks_stat_gsub <- numeric(length(nsizes))
df_nleaves <- data.frame(iteration=numeric(),
                         nleaves=numeric(),
                         type=character())
df_convergence <- data.frame(iteration=numeric(),
                             loss=numeric(),
                             type=character())

# run
set.seed(314)
for(i in 1:length(nsizes))
{
    cat("Iter: ", i, "\n")
    gc()
    # n
    nsize <- nsizes[i]
    
    # training indices
    tr_ind <- sample(length(ytr), nsize)
    
    # training
    start <- Sys.time()
    mod_vanilla <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                             gsub_compare = F, learning_rate = 0.01)
    time_vanilla[i] <- Sys.time() - start
    
    start <- Sys.time()
    mod_gsub <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                          gsub_compare = T, learning_rate = 0.01)
    time_gsub[i] <- Sys.time() - start
    
    
    if(F)
    {
        p.xgb <- list(eta=0.01, objective="binary:logistic", eval_metric = "logloss", lambda=0,
                      tree_method="exact")
        tr_ind_xgb <- sample(tr_ind, ceiling(0.7*length(tr_ind)))
        tr_ind_xgb_val <- tr_ind[!(tr_ind %in% tr_ind_xgb)]
        dmat_xgb <- xgb.DMatrix(data=xtr[tr_ind_xgb,], label=ytr[tr_ind_xgb])
        dmat_xgb_val <- xgb.DMatrix(data=xtr[tr_ind_xgb_val,], label=ytr[tr_ind_xgb_val])
        start <- Sys.time()
        xgb_mod <- xgb.train(p.xgb, dmat_xgb, nrounds=50000, list(val=dmat_xgb_val),
                             early_stopping_rounds = 50, verbose=10)
        time_xgb[i] <- Sys.time() - start
    }
    
    # Loss
    pred_vanilla <- predict(mod_vanilla, xte)
    loss_vanilla[i] <- logloss(yte, pred_vanilla)
    
    pred_gsub <- predict(mod_gsub, xte)
    loss_gsub[i] <- logloss(yte, pred_gsub)
    
    if(F)
    {
        pred_xgb <- predict(xgb_mod, xte)
        loss_xgb[i] <- logloss(yte, pred_xgb)
    }
    
    # ksval
    ks_stat_vanilla[i] <- gbt.ksval(mod_vanilla, yte,xte)$statistic
    ks_stat_gsub[i] <- gbt.ksval(mod_gsub, yte,xte)$statistic
    
    # auc
    roc_vanilla <- roc(yte, pred_vanilla)
    auc_vanilla[i] <- auc(roc_vanilla)
    roc_gsub <- roc(yte, pred_gsub)
    auc_gsub[i] <- auc(roc_gsub)
    
    # ntrees
    ntrees_vanilla[i] <- mod_vanilla$get_num_trees()
    ntrees_gsub[i] <- mod_gsub$get_num_trees()
    
    # nfeatures
    nfeatures_vanilla[i] <- length(gbt.importance(colnames(xte), mod_vanilla))
    nfeatures_gsub[i] <- length(gbt.importance(colnames(xte), mod_gsub))
    
    # nleaves
    leaves_vanilla <- mod_vanilla$get_num_leaves()
    nleaves_vanilla[i] <- sum(leaves_vanilla)
    df_nleaves <- rbind(df_nleaves, 
                        data.frame(iteration=1:length(leaves_vanilla),
                                   nleaves = leaves_vanilla,
                                   model = paste0("vanilla",nsize)))
    leaves_gsub <- mod_gsub$get_num_leaves()
    nleaves_gsub[i] <- sum(leaves_gsub)
    df_nleaves <- rbind(df_nleaves, 
                        data.frame(iteration=1:length(leaves_gsub),
                                   nleaves = leaves_gsub,
                                   model = paste0("gsub",nsize)))
    
    # convergence
    conv_vanilla <- gbt.convergence(mod_vanilla, yte, xte)
    df_convergence <- rbind(df_convergence,
                            data.frame(iteration= 0:(length(conv_vanilla)-1),
                                       loss = conv_vanilla,
                                       model = paste0("vanilla",nsize)))
    conv_gsub <- gbt.convergence(mod_gsub, yte, xte)
    df_convergence <- rbind(df_convergence,
                            data.frame(iteration= 0:(length(conv_gsub)-1),
                                       loss = conv_gsub,
                                       model = paste0("gsub",nsize)))
    
}
if(F)
{
    save(time_vanilla, time_gsub, loss_vanilla, loss_gsub,
         auc_vanilla, auc_gsub, ntrees_vanilla, ntrees_gsub,
         nleaves_gsub, nleaves_vanilla, nfeatures_gsub, nfeatures_vanilla,
         ks_stat_gsub, ks_stat_vanilla, df_nleaves, df_convergence, file="results/res_chap_6.RData")
}

# Plot nleaves
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

ind_vanilla <- grep("vanilla", df_nleaves$model)
df_nleaves$model <- as.character(df_nleaves$model)
df_nleaves$model[ind_vanilla] <- rep(c("100", "1000", "10000", "100000"), ntrees_vanilla)

ind_gsub <- grep("gsub", df_nleaves$model)
df_nleaves$model[ind_gsub] <- rep(c("100", "1000", "10000", "100000"), ntrees_gsub)
colnames(df_nleaves)[3] <- c("#Training observations")

#names(cbp2) <- names(df[,-1])
y_axis_max <- max(df_nleaves$nleaves)
p_nleaves_vanilla <- df_nleaves[ind_vanilla,] %>%
    ggplot(aes(x=iteration, y=nleaves, group=`#Training observations`)) + 
    #geom_point() + 
    geom_line(aes(linetype=`#Training observations`, colour=`#Training observations`), size=1) + 
    theme_bw() + 
    ylab("Number of leaves") + 
    xlab("Boosting iteration") +
    ggtitle("Method: Vanilla") +
    scale_y_continuous(trans='log2', limits=c(2, y_axis_max)) + 
    scale_color_manual(values=cbp2) + 
    theme(#legend.position=c(1,1),legend.justification=c(1,1),
          #legend.direction="vertical",
          #legend.box="horizontal",
          #legend.box.just = c("top"), 
          legend.background = element_rect(fill="transparent", colour="black"))
p_nleaves_vanilla

p_nleaves_gsub <- df_nleaves[ind_gsub,] %>%
    ggplot(aes(x=iteration, y=nleaves, group=`#Training observations`)) + 
    #geom_point() + 
    geom_line(aes(linetype=`#Training observations`, colour=`#Training observations`), size=1) + 
    theme_bw() + 
    ylab("Number of leaves") + 
    xlab("Boosting iteration") + 
    ggtitle("Method: Global-subset") +
    scale_y_continuous(trans='log2', limits=c(2, y_axis_max)) + 
    scale_color_manual(values=cbp2) + 
    theme(#legend.position=c(1,1),legend.justification=c(1,1),
        #legend.direction="vertical",
        #legend.box="horizontal",
        #legend.box.just = c("top"), 
        legend.background = element_rect(fill="transparent", colour="black"))
p_nleaves_gsub

ggpubr::ggarrange(p_nleaves_vanilla, p_nleaves_gsub, ncol=2, common.legend = TRUE, legend="bottom")
#gridExtra::grid.arrange(p_nleaves_vanilla, p_nleaves_gsub, ncol=2)



if(F)
{
    p_nleaves <- ggpubr::ggarrange(p_nleaves_vanilla, p_nleaves_gsub, ncol=2, common.legend = TRUE, legend="bottom")
    
    pdf(file="../../../../gbtorch-package-article/figures/model_nleaves.pdf", width = 8,height = 4)
    p_nleaves
    dev.off()
}

# Plot convergence
#names(cbp2) <- names(df[,-1])
df_convergence$model <- as.character(df_convergence$model)
df_convergence$model[df_convergence$model == "vanilla100"] = "vanilla-100"
df_convergence$model[df_convergence$model == "vanilla1000"] = "vanilla-1000"
df_convergence$model[df_convergence$model == "vanilla10000"] = "vanilla-10000"
df_convergence$model[df_convergence$model == "vanilla1e+05"] = "vanilla-100000"
df_convergence$model[df_convergence$model == "gsub100"] = "global-subset-100"
df_convergence$model[df_convergence$model == "gsub1000"] = "global-subset-1000"
df_convergence$model[df_convergence$model == "gsub10000"] = "global-subset-10000"
df_convergence$model[df_convergence$model == "gsub1e+05"] = "global-subset-100000"
p_convergence <- df_convergence %>%
    ggplot(aes(x=iteration, y=loss, group=model)) + 
    #geom_point() + 
    geom_line(aes(linetype=model, colour=model), size=1) + 
    ylab("Test loss") + 
    xlab("Boosting iteration") + 
    theme_bw() + 
    scale_color_manual(values=cbp2)  + 
    theme(#legend.position=c(1,1),legend.justification=c(1,1),
          #legend.direction="vertical",
          #legend.box="horizontal",
          #legend.box.just = c("top"), 
          legend.background = element_rect(fill="transparent", colour="black"))
p_convergence

if(F)
{
    pdf(file="../../../../gbtorch-package-article/figures/model_convergence.pdf", width = 8,height = 3)
    p_convergence
    dev.off()
}


# \multicolumn{7}{c}{$n=100$}\\
# Table
#n & model name & loss & time & ntrees & nleaves
mod_names <- c("vanilla", "global-subset", "xgb")
for(i in 1:length(nsizes))
{
    cat(paste0("\\multicolumn{7}{c}{$n=", format(nsizes[i],scientific = FALSE), "$}\\\\ \n"))
    for(j in 1:length(mod_names))
    {
        if(j==1)
        {
            cat(paste0(
                #nsizes[i], " & ",
                mod_names[j], " & ",
                format(loss_vanilla[i], digits=4), " & ",
                format(auc_vanilla[i], digits=4), " & ",
                #format(ks_stat_vanilla[i], digits=4), " & ",
                format(time_vanilla[i], digits=4), " & ",
                ntrees_vanilla[i], " & ",
                nleaves_vanilla[i], " & ",
                nfeatures_vanilla[i],
                "\\\\ \n"  
            ))
        }else if(j==2)
        {
            cat(paste0(
                #nsizes[i], " & ",
                mod_names[j], " & ",
                format(loss_gsub[i], digits=4), " & ",
                format(auc_gsub[i], digits=4), " & ",
                #format(ks_stat_gsub[i], digits=4), " & ",
                format(time_gsub[i], digits=4), " & ",
                ntrees_gsub[i], " & ",
                nleaves_gsub[i], " & ",
                nfeatures_gsub[i],
                "\\\\ \n"  
            ))
        }
        # else
        # {
        #     cat(paste0(
        #         nsizes[i], " & ",
        #         mod_names[j], " & ",
        #         format(loss_xgb[i], digits=4), " & ",
        #         format(time_xgb[i], digits=4), "\\\\ \n"  
        #     ))
        # }
    }
    cat(paste0("\\hline \n"))
}





# gsub
ind <- sample(length(yte),1000)
pred_gsub <- predict(mod_gsub, xte)
pred_vanilla <- predict(mod_vanilla, xte)
pred_xgb <- predict(xgb_mod, xte)
imp <- gbt.importance(colnames(xtr), mod_vanilla)
cind <- which(colnames(xte)==names(imp)[1])

df_plot <- data.frame("pred"=pred_vanilla[ind], "xval"=xte[ind,cind], "res"=as.factor(yte[ind]))
p1 <- df_plot %>%
    ggplot(aes(x=xval, y=pred, colour=res)) + 
    geom_point() + 
    ylim(0, 1) +
    theme_bw()

df_plot <- data.frame("pred"=pred_gsub[ind], "xval"=xte[ind,cind], "res"=as.factor(yte[ind]))
p2 <- df_plot %>%
    ggplot(aes(x=xval, y=pred, colour=res)) + 
    geom_point() + 
    ylim(0, 1) +
    theme_bw()

df_plot <- data.frame("pred"=pred_xgb[ind], "xval"=xte[ind,cind], "res"=as.factor(yte[ind]))
p3 <- df_plot %>%
    ggplot(aes(x=xval, y=pred, colour=res)) + 
    geom_point() + 
    ylim(0, 1) +
    theme_bw()

gridExtra::grid.arrange(p1,p2,p3, ncol=3)



plot(xte[ind,cind],yte[ind])
points(xte[ind,cind], pred_vanilla[ind],col=2)
points(xte[ind,cind], pred_gsub[ind],col=3)
points(xte[ind,cind], pred_xgb[ind],col=4)










# n=1000
nsize <- 1000

# training indices
tr_ind <- sample(length(ytr), nsize)

# training
start <- Sys.time()
mod_vanilla <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                         gsub_compare = F, learning_rate = 0.01)
time_vanilla <- Sys.time() - start

start <- Sys.time()
mod_gsub <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                         gsub_compare = T, learning_rate = 0.01)
time_gsub <- Sys.time() - start

p.xgb <- list(eta=0.01, objective="binary:logistic", eval_metric = "logloss")
dmat_xgb <- xgb.DMatrix(data=xtr[tr_ind,], label=ytr[tr_ind])
start <- Sys.time()
xgb_mod <- xgboost(dmat_xgb, params = p.xgb, nrounds = 1000)
time_xgb <- Sys.time() - start

# Loss
pred_vanilla <- predict(mod_vanilla, xte)
roc_vanilla <- roc(yte, pred_vanilla)
agt_vanilla_loss[i] <- logloss(yte, pred_vanilla)
agt_vanilla_auc[i] <- auc(roc_vanilla)

pred_gsub <- predict(mod_gsub, xte)
roc_gsub <- roc(yte, pred_gsub)
agt_gsub_loss[i] <- logloss(yte, pred_gsub)
agt_gsub_auc[i] <- auc(roc_gsub)

pred_xgb <- predict(xgb_mod, xte)
roc_xgb <- roc(yte, pred_xgb)
xgb_loss <- logloss(yte, pred_xgb)
xgb_auc[i] <- auc(roc_xgb)

# ntrees
agt_vanilla_ntrees <- mod_vanilla$get_num_trees()
agt_gsub_ntrees <- mod_gsub$get_num_trees()
xgb_ntrees <- xgb_mod$niter

# nleaves
agt_vanilla_nleaves <- mod_vanilla$get_num_leaves()
plot(agt_vanilla_nleaves, type="l")
agt_gsub_nleaves <- mod_gsub$get_num_leaves()
plot(agt_gsub_nleaves, type="l")

# importance
gbt.importance(colnames(xte), mod_vanilla)
gbt.importance(colnames(xte), mod_gsub)

# ks-val
gbt.ksval(mod_vanilla, yte, xte)
gbt.ksval(mod_gsub, yte, xte)



nsize <- 10^(2:6)
gbt_loss <- xgb_loss <- gbt_auc <- xgb_auc <- numeric(length(nsize))
for(i in 1:length(nsize))
{
    # agt
    tr_ind <- sample(length(ytr), nsize[i])
    agt_mod <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                         greedy_complexities = T, learning_rate = 0.3)
    pred <- predict(gbt_mod, xte)
    prob <-  1/(1+exp(-pred))
    roc_gbt <- roc(yte, prob)
    gbt_loss[i] <- logloss(yte, prob)
    gbt_auc[i] <- auc(roc_gbt)
    
    # xgb
    p.xgb <- list(eta=0.3, objective="binary:logistic", eval_metric = "logloss")
    dmat_xgb <- xgb.DMatrix(data=xtr[tr_ind,], label=ytr[tr_ind])
    xgb_mod <- xgboost(dmat_xgb, params = p.xgb, nrounds = 100)    
    prob_xgb <- predict(xgb_mod, xte)
    roc_xgb <- roc(yte, prob_xgb)
    xgb_loss[i] <- logloss(yte, prob_xgb)
    xgb_auc[i] <- auc(roc_xgb)
    
    
    #    p.xgb <- list(eta = 0.01, lambda=0, objective="binary:logistic", eval_metric = "logloss") # lambda=0, vanilla 2'nd order gtb
    #x.train.xgb <- xgb.DMatrix(data=x.train, label = y.train)
    #xgb.n <- xgb.cv(p.xgb, x.train.xgb, nround=2000, nfold=5, early_stopping_rounds =50, verbose = F)
    #xgb.nrounds <- which.min(xgb.n$evaluation_log$test_logloss_mean)
    #xgb.mod <- xgb.train(p.xgb, x.train.xgb, xgb.nrounds)
    
}

res <- data.frame(n=nsize, gbt_auc=gbt_auc, xgb_auc=xgb_auc, 
                  gbt_loss=gbt_loss, xgb_loss=xgb_loss)

p1 <- res %>% ggplot() + 
    geom_point(aes(x=n,y=gbt_loss)) +
    geom_line(aes(x=n,y=gbt_loss)) + 
    geom_point(aes(x=n,y=xgb_loss), colour="orange") +
    geom_line(aes(x=n,y=xgb_loss), colour="orange") +
    scale_x_continuous(trans='log10')
p1

p2 <- res %>% ggplot() + 
    geom_point(aes(x=n,y=gbt_auc)) +
    geom_line(aes(x=n,y=gbt_auc)) + 
    geom_point(aes(x=n,y=xgb_auc), colour="orange") +
    geom_line(aes(x=n,y=xgb_auc), colour="orange") +
    scale_x_continuous(trans='log10')
p2






tr_ind <- sample(length(ytr), 100000)
mod <- gbt.train(ytr[tr_ind], xtr[tr_ind,], loss_function = "logloss", verbose=1,
                 gsub_compare = TRUE, learning_rate = 0.01)
plot(mod$get_num_leaves(), type="l")
prob <- predict(mod, xte)

logloss(yte, prob)
logloss(yte, mean(ytr[tr_ind]))

roc_obj <- roc(yte, prob)
auc(roc_obj)
plot(roc_obj)

gbt.importance(colnames(xtr), mod)
gbt.ksval(mod, yte, xte)













