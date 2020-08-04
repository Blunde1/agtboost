# optimism as a function of num splits

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(RcppEigen)
library(ggplot2)
library(ggpubr)
library(reshape2)
library(dplyr)
library(tidyr)
library(latex2exp)

Rcpp::sourceCpp("tree_dim.cpp")
#MSE Loss and derivatives
loss <- function(y,y.hat){
    mean((y-y.hat)^2)
}
dloss <- function(y, y.hat) -2*sum(y-y.hat)
ddloss <- function(y, y.hat) 2*length(y)

rbbridge <- function(t_vec){
    # simulate a brownian bridge at places t in t_vec
    m <- length(t_vec)
    W <- numeric(m+1)
    W[1] <- rnorm(1, 0, sd=sqrt(t_vec[1]))
    if(m>1){
        for(i in 2:m){
            W[i] <- W[i-1] + rnorm(1, 0, sqrt(t_vec[i]-t_vec[i-1]))
        }
        
    }
    W[m+1] <- W[m] + rnorm(1, 0, sqrt(1-t_vec[m])) # at t=T=1
    B <- W[1:m] - t_vec*W[m+1] # B(t) = W(t) - t/T W(T)
    return(B)
}

emax_bb <- function(t){
    
    # t should be Uniform on 0-1, preferably sorted
    B=100 # replicates
    mx <- numeric(B) # storing max on bridge
    
    for( i in 1:B ){
        # Simulate left
        b <- rbbridge(t)
        x <- b^2/(t*(1-t))
        mx[i] <- max(x)
    }
    
    return(mean(mx))
}

feature_to_unif <- function(x){
    
    # returns sorted x->u, minus last
    # x: numeric vector
    n <- length(x)
    x_unique <- unique(x)
    v <- length(x_unique)
    u <- numeric(v)
    
    for(i in 1:v){
        u[i] <- sum(x <= x_unique[i]) / n
    }
    
    return(sort(u)[-v])
}

root_bias <- function(g,h)
{
    w <- -sum(g)/sum(h)
    mean((g+h*w)^2) / mean(h) / length(g)
}

DGP <- function(n, nsplits){
    
    # create grid
    # draw n_group from each grid-point
    # sample remaining values (without replacement) from grid
    
    n_group <- floor(n/(nsplits+1))
    xvalues <- 0:nsplits / nsplits
    x <- as.vector(sapply(xvalues, function(x)rep(x,n_group)))
    x <- c(x, sample(xvalues, n-length(x)) )
    #x <- c(x, rep(last(x),n-length(x)))
    x <- matrix(x, ncol=1)
    
    y <- rnorm(n)
    
    res <- list(x=x, y=y)
    return(res)
}

n <- 100
nsplits <- 50
DGP(n, nsplits)$x

# Loop over B experiments; output optimism and cv vectors of length B
set.seed(123)
B <- 1000

# Parameters of experiment
n <- 100
depth <- 1
folds <- 10
ntest <- 1000

# num_splits
num_splits <- c(1, seq(2,n-1,by=1))
loss_tr <- cv_optimism <- leaf_optimism <- tree_optimism <- topology_optimism <- numeric(length(num_splits))

# quantiles
loss_tr_l <- loss_tr_u <- cv_optimism_l <- cv_optimism_u <- leaf_optimism_l <- 
    leaf_optimism_u <- tree_optimism_l <- tree_optimism_u <- topology_optimism_l <- topology_optimism_u <- 
    test_optimism <- test_optimism_l <- test_optimism_u <- 
    numeric(length(num_splits))


pb <- txtProgressBar(min = 0, max = length(num_splits), style = 3)
for(i in 1:length(num_splits)){
    
    # B experiments on split num
    tree_opt <- tree_opt_topology <- tree_opt_leaf <- tree_opt_cv <- tree_loss_tr <- tree_opt_test <- numeric(B)
    loss_tr_cv <- loss_te_cv <- tree_cv_opt <- numeric(folds)
    tree_opt_test_k <- numeric(ntest)
    
    for(b in 1:B){
        
        # Simulate data
        simulated_data <- DGP(n, num_splits[i])
        x <- simulated_data$x
        y <- simulated_data$y
        
        # compute derivatives
        y0.hat <- mean(y)
        g <- sapply(y, dloss, y.hat=y0.hat)
        h <- sapply(y, ddloss, y.hat=y0.hat)
        
        # Train gbtree
        tree <- new(GBTREE)
        tree$train(g, h, x, depth)
        
        # calculate loss
        pred_tr <- y0.hat + tree$predict_data(x)
        tree_loss_tr[b] <- loss(y, pred_tr)
        
        # Calculate optimism
        tree_opt_leaf[b] <- tree$getTreeBias()
        t <- feature_to_unif(x)
        
        # Calculate optimism
        t <- feature_to_unif(x)
        tree_opt[b] <- root_bias(g,h)*emax_bb(t) + root_bias(g,h)
        
        # test optimism
        for(k in 1:ntest){
            data_te <- DGP(n, num_splits[i])
            x_te <- data_te$x
            y_te <- data_te$y
            pred_te <- y0.hat + tree$predict_data(x_te)
            tree_opt_test_k[k] <- loss(y_te, pred_te) - tree_loss_tr[b] # test - train
        }
        tree_opt_test[b] <- mean(tree_opt_test_k)
        
        
        
        # Generate indices of holdout observations
        holdout <- split(sample(1:n), 1:folds)
        
        # Train CV-gbtrees
        for(k in 1:folds){
            
            # cv derivatives
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict
            pred_tr_cv_tree <- tree_k$predict_data(x[-holdout[[k]],,drop=F])
            pred_te_cv_tree <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            pred_tr_cv <- y0.hat.cv + pred_tr_cv_tree
            pred_te_cv <- y0.hat.cv + pred_te_cv_tree
            
            # loss on train and holdout
            loss_tr_cv[k] <- loss( y[-holdout[[k]]], pred_tr_cv ) 
            loss_te_cv[k] <- loss( y[holdout[[k]]], pred_te_cv ) 
            
            # optimism
            tree_cv_opt[k] <- loss_te_cv[k] - loss_tr_cv[k]
        }
        
        # Calculate CV-implied optimism
        tree_opt_cv[b] <- mean(tree_cv_opt)
        
        
    }
    tree_opt2 <- 1*tree_opt_topology + tree_opt_leaf
    
    loss_tr[i] <- mean(tree_loss_tr)
    cv_optimism[i] <- mean(tree_opt_cv)
    test_optimism[i] <- mean(tree_opt_test)
    leaf_optimism[i] <- mean(tree_opt_leaf)
    tree_optimism[i] <- mean(tree_opt)
    topology_optimism[i] <- mean(tree_opt_topology)
    
    #  quantiles
    loss_tr_l[i] <- quantile(tree_loss_tr, 0.25)
    loss_tr_u[i] <- quantile(tree_loss_tr, 0.75)
    cv_optimism_l[i] <- quantile(tree_opt_cv, 0.25)
    cv_optimism_u[i] <- quantile(tree_opt_cv, 0.75)
    test_optimism_l[i]  <- quantile(tree_opt_test, 0.25)
    test_optimism_u[i]  <- quantile(tree_opt_test, 0.75)
    leaf_optimism_l[i] <- quantile(tree_opt_leaf, 0.25)
    leaf_optimism_u[i] <- quantile(tree_opt_leaf, 0.75)
    tree_optimism_l[i] <- quantile(tree_opt, 0.25)
    tree_optimism_u[i] <- quantile(tree_opt, 0.75)
    topology_optimism_l[i] <- quantile(tree_opt_topology, 0.25)
    topology_optimism_u[i] <- quantile(tree_opt_topology, 0.75)
    
    
    setTxtProgressBar(pb, i)
}

if(F)
{
    results_df <- data.frame("a"=num_splits, 
                             "adjusted"=tree_optimism, 
                             "cv"=cv_optimism, 
                             "test" = test_optimism)
    save(results_df, file="results/stump_opt_vs_nsplits.RData")
    load("results/stump_opt_vs_nsplits.RData")
}

load("single_feature_case/results/stump_opt_vs_nsplits.RData")
df <- results_df
names(df) <- c("a", "$\\tilde{C}$", "10-fold CV", "test")#, "10-fold CV adjusted")
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
names(cbp2) <- names(df[,-1])
df_long <- df[,-5] %>% gather(type,value, -a)

p0 <- df_long %>%
    ggplot() + 
    geom_point(aes(x=a, y=value, colour=type)) + 
    geom_line(aes(x=a, y=value, colour=type), size=1) + 
    
    #geom_ribbon(data=df2,aes(a, ymax = lr_adj_u, ymin = lr_adj_l), fill = cbp2[1], alpha= 0.30) + 
    #geom_ribbon(data=df2,aes(a, ymax = cv_opt_u, ymin = cv_opt_l), fill = cbp2[2], alpha= 0.30) + 
    #geom_ribbon(data=df2,aes(a, ymax = test_optimism_u, ymin = test_optimism_l), fill = cbp2[3], alpha= 0.30) + 
    
    #labs(y="Optimism of stump-model", title="$m=1$") + 
    ylab("Optimism of stump-model") + 
    scale_color_manual(name=NULL, values=cbp2,
                       labels=list(TeX("$\\tilde{C}$"), "10-fold CV", TeX("$C$")) 
    ) +
    theme_bw()
p0


if(F){
    # PDF
    pdf(file = "../../../gbtree_information/figures/loss_reduction_vs_nsplits.pdf", 
        width = 6.5, height = 3)
    print(p0)
    dev.off()
    
    # TIKZ
    library(tikzDevice)
    options(tz="CA")
    tikz(file = "../../../gbtree_information/figures/loss_reduction_vs_nsplits.tex", 
         width = 6.5, height = 3)
    print(p0)
    dev.off()
}