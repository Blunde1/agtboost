# loss reduction histogram
# one-hot encoding

library(RcppEigen)
library(ggplot2)
library(ggpubr)
library(reshape2)
library(dplyr)
library(tidyr)
library(ggridges)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
Rcpp::sourceCpp("../tree_dim.cpp")

# loss functions
loss <- function(y,y.hat) sum((y-y.hat)^2) / length(y)
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

# returns list of (x,y) where x has nsplits possible splits and y is of type type
fun <- function(n, type, nsplits){
    
    y <- numeric(n)
    
    n_group <- floor(n/(nsplits+1))
    xvalues <- 0:nsplits
    x <- as.vector(sapply(xvalues, function(x)rep(x,n_group))) / nsplits
    x <- c(x, rep(last(x),n-length(x)))
    
    if(type==1){
        y <- rnorm(n,0,1)    
    }else if(type==2){
        y <- rnorm(n,0,10)
    }else if(type==3){
        y <- rnorm(n, round(x), 1)
    }else if(type==4){
        y <- rnorm(n,round(x),10)
    }else if(type==5){
        y <- rnorm(n, x, 1)
    }else if(type==6){
        y <- rnorm(n,x,10)
    }
    
    res <- list(x=x, y=y)
    return(res)
}


# Simulate loss reduction type 1
set.seed(123)
B <- 1000
m <- 1000
test_tmp <- numeric(m)
n <- c(30, 100, 1000)
nsplits <- 1
type <- 1

# lists for storing data
result <- list()

for(i in 1:length(n)){
    
    cat("i: ",i, "\n")
    
    # temporary storage
    cv5 <- cv10 <- cvn <- adjusted <- test <- train <- numeric(B)
    
    cv5_tmp <- numeric(5)
    cv10_tmp <- numeric(10)
    cvn_tmp <- numeric(n[i])
    
    # progress bar
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    
    for(b in 1:B){
        
        # generate data
        data <- fun(n[i], type, nsplits)
        x <- as.matrix(data$x)
        y <- as.vector(data$y)
        
        # compute derivatives
        y0.hat <- mean(y)
        g <- sapply(y, dloss, y.hat=y0.hat)
        h <- sapply(y, ddloss, y.hat=y0.hat)
        ch0 <- mean((g+h*y0.hat)^2) / mean(h)/n[i]
        
        # Train gbtree
        tree <- new(GBTREE)
        tree$train(g, h, x, 1)
        
        # calculate loss
        pred_tr <- y0.hat + tree$predict_data(x)
        train[b] <- loss(y, y0.hat) - loss(y, pred_tr) 
        
        # Calculate optimism
        #tree_opt_leaf <- tree$getTreeBias()
        t <- feature_to_unif(x)
        max_cir <- emax_bb(t)
        CRt <- ch0 * max_cir
        #ch1 <- tree_opt_leaf/2 * max_double_sq_bb
        
        # adjusted lossred
        adjusted[b] <- train[b] - CRt
        
        # Test reduction in loss
        for(k in 1:m){
            testdata <- fun(n[i], type, nsplits)
            xte <- as.matrix(testdata$x)
            yte <- as.matrix(testdata$y)
            
            pred_te <- y0.hat + tree$predict_data(xte)
            test_tmp[k] <- loss(yte, y0.hat) - loss(yte, pred_te)
            
        }
        test[b] <- mean(test_tmp)
        
        # 5-fold cv
        # Generate indices of holdout observations
        holdout <- split(sample(1:n[i]), 1:5)
        
        # Train CV-gbtrees
        for(k in 1:5){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv5_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv5[b] <- mean(cv5_tmp)
        
        # 10-fold cv
        holdout <- split(sample(1:n[i]), 1:10)
        
        # Train CV-gbtrees
        for(k in 1:10){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv10_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv10[b] <- mean(cv10_tmp)
        
        # n-fold cv
        holdout <- split(sample(1:n[i]), 1:n[i])
        
        # Train CV-gbtrees
        for(k in 1:n[i]){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cvn_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cvn[b] <- mean(cvn_tmp)
        
        # update progress bar
        setTxtProgressBar(pb, b)
        
    }
    
    result[[i]] <- data.frame("5-fold cv"=cv5, "10-fold cv"=cv10, "n-fold cv"=cvn,
                              "adjusted"=adjusted, "test1000"=test, "train"=train)
    names(result[[i]]) <- c("5-fold cv", "10-fold cv", "n-fold cv", "adjusted", "test1000", "train")
    
    close(pb)
    
}

if(F){
    save(result, file="results/loss_reduction_simple_sim_plot.RData")
    load("results/loss_reduction_simple_sim_plot.RData")
}



# scale to n
#result[[1]] <- result[[1]] * n[1]
#result[[2]] <- result[[2]] * n[2]
#result[[3]] <- result[[3]] * n[3]

# The palette with black:
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



labels_custom <- list( "$5$-fold CV",  #TeX("$5$-fold CV"), 
                       "$10$-fold CV",  #TeX("$10$-fold CV"),
                       "$n$-fold CV",  #TeX("$n$-fold CV"),
                       "$\\tilde{\\mathcal{R}}^0$", #TeX("$\\tilde{R}^0$"),
                       "$\\mathcal{R}^0$",  #TeX("$R^0$"),
                       "$\\mathcal{R}$")  #TeX("$R$") )
                       #c("10-fold cv", "n-fold cv", "adjusted", "test1000", "train")

#xlim <- c(-10,10)
#xlim1 <- xlim/n[1]
#xlim2 <- xlim/n[2]
#xlim3 <- xlim/n[3]

p1 <- result[[1]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.4,0.2,by=0.04)) + 
    #scale_x_continuous(expand = c(0.01, 0)) +
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=30$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim1[1],xlim1[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p1

p2 <- result[[2]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.1,0.08,by=0.01)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=100$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim2[1],xlim2[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p2

p3 <- result[[3]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.01,0.006,by=0.001)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=1000$") + 
    scale_fill_manual(values = cbp2, labels=labels_custom) +
    #xlim(xlim3[1],xlim3[2]) + 
    theme_bw() +
    theme(axis.text.y = element_blank())
#theme(legend.position = "none")
p3


# Relevant figure
plots <- ggarrange(p1, p2, p3, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
plots

if(F){
    #pdf("../../../gbtree_information/figures/loss_reduction_simple_sim.pdf", width=8, height=4, paper="special")
    plots
    #dev.off()
    
    # TIKZ
    #library(tikzDevice)
    #options(tz="CA")
    #tikz(file = "../../../gbtree_information/figures/loss_reduction_sim_simple.tex", width = 6.5, height = 3.5)
    print(plots)
    #dev.off()
}



############### --- TEST PLOTS: Round DGP ---- #################

################# A=9 ##############
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

# Simulate loss reduction type 1
set.seed(123)
B <- 1000 # iteration / samples
m <- 1 # test sets per iteration
test_tmp <- numeric(m)
n <- c(30, 100, 200)
nsplits <- 9
type <- 1

# lists for storing data
result <- list()

for(i in 1:length(n)){
    
    cat("i: ",i, "\n")
    
    # temporary storage
    cv5 <- cv10 <- cvn <- adjusted <- test <- train <- numeric(B)
    
    cv5_tmp <- numeric(5)
    cv10_tmp <- numeric(10)
    cvn_tmp <- numeric(n[i])
    
    # progress bar
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    
    for(b in 1:B){
        
        # generate data
        data <- DGP(n[i], nsplits)
        x <- as.matrix(data$x)
        y <- as.vector(data$y)
        
        # compute derivatives
        y0.hat <- mean(y)
        g <- sapply(y, dloss, y.hat=y0.hat)
        h <- sapply(y, ddloss, y.hat=y0.hat)
        ch0 <- mean((g+h*y0.hat)^2) / mean(h)/n[i]
        
        # Train gbtree
        tree <- new(GBTREE)
        tree$train(g, h, x, 1)
        
        # calculate loss
        pred_tr <- y0.hat + tree$predict_data(x)
        train[b] <- loss(y, y0.hat) - loss(y, pred_tr) 
        
        # Calculate optimism
        tree_opt_leaf <- tree$getTreeBias()
        t <- feature_to_unif(x)
        max_double_sq_bb <- emax_bb(t)
        ch1 <- tree_opt_leaf/2 * max_double_sq_bb
        
        # adjusted lossred
        adjusted[b] <- train[b] + ch0 - ch1
        
        # Test reduction in loss
        for(k in 1:m){
            testdata <- DGP(n[i], nsplits)
            xte <- as.matrix(testdata$x)
            yte <- as.matrix(testdata$y)
            
            pred_te <- y0.hat + tree$predict_data(xte)
            test_tmp[k] <- loss(yte, y0.hat) - loss(yte, pred_te)
            
        }
        test[b] <- mean(test_tmp)
        
        # 5-fold cv
        # Generate indices of holdout observations
        holdout <- split(sample(1:n[i]), 1:5)
        
        # Train CV-gbtrees
        for(k in 1:5){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv5_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv5[b] <- mean(cv5_tmp)
        
        # 10-fold cv
        holdout <- split(sample(1:n[i]), 1:10)
        
        # Train CV-gbtrees
        for(k in 1:10){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv10_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv10[b] <- mean(cv10_tmp)
        
        # n-fold cv
        holdout <- split(sample(1:n[i]), 1:n[i])
        
        # Train CV-gbtrees
        for(k in 1:n[i]){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cvn_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cvn[b] <- mean(cvn_tmp)
        
        # update progress bar
        setTxtProgressBar(pb, b)
        
    }
    
    result[[i]] <- data.frame("5-fold cv"=cv5, "10-fold cv"=cv10, "n-fold cv"=cvn,
                              "adjusted"=adjusted, "test1000"=test, "train"=train)
    names(result[[i]]) <- c("5-fold cv", "10-fold cv", "n-fold cv", "adjusted", "test1000", "train")
    
    close(pb)
    
}

if(F){
    save(result, file="results/loss_reduction_simple_sim_plot.RData")
    load("results/loss_reduction_simple_sim_plot.RData")
}



# scale to n
#result[[1]] <- result[[1]] * n[1]
#result[[2]] <- result[[2]] * n[2]
#result[[3]] <- result[[3]] * n[3]

# The palette with black:
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



labels_custom <- list( "$5$-fold CV",  #TeX("$5$-fold CV"), 
                       "$10$-fold CV",  #TeX("$10$-fold CV"),
                       "$n$-fold CV",  #TeX("$n$-fold CV"),
                       "$\\tilde{\\mathcal{R}}^0$", #TeX("$\\tilde{R}^0$"),
                       "$\\mathcal{R}^0$",  #TeX("$R^0$"),
                       "$\\mathcal{R}$")  #TeX("$R$") )
#c("10-fold cv", "n-fold cv", "adjusted", "test1000", "train")

xlim <- c(-10,10)
xlim1 <- xlim/n[1]
xlim2 <- xlim/n[2]
xlim3 <- xlim/n[3]

p1 <- result[[1]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.4,0.2,by=0.04)) + 
    #scale_x_continuous(expand = c(0.01, 0)) +
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=30$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim1[1],xlim1[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p1

p2 <- result[[2]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.2,0.1,by=0.01)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=100$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim2[1],xlim2[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p2

p3 <- result[[3]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=2, breaks=seq(-0.05,0.05,by=0.0025)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=1000$") + 
    scale_fill_manual(values = cbp2, labels=labels_custom) +
    #xlim(xlim3[1],xlim3[2]) + 
    theme_bw() +
    theme(axis.text.y = element_blank())
#theme(legend.position = "none")
p3



plots <- ggarrange(p1, p2, p3, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
plots

sapply(1:3, function(i) colMeans(result[[i]]))



















# NO DGP -- OK
# mathcal(r): tikzDevice -- OK
# samme simuleringar -- dersom forskjell
# hist -- Sjekk med tore
# one-hot med round() -- ekstreme forskjellar i høve cv
# faktor n -- tenk deg om
# triks: e^-3 og e^-2
# histogram fiks
# histogram: samme x-akse



# Simulate loss reduction type 3
set.seed(123)
B <- 1000
m <- 1000
test_tmp <- numeric(m)
n <- c(30, 100, 1000)
nsplits <- 1
type=3

# lists for storing data
result <- list()

for(i in 1:length(n)){
    
    cat("i: ",i, "\n")
    
    # temporary storage
    cv5 <- cv10 <- cvn <- adjusted <- test <- train <- numeric(B)
    
    cv5_tmp <- numeric(5)
    cv10_tmp <- numeric(10)
    cvn_tmp <- numeric(n[i])
    
    # progress bar
    pb <- txtProgressBar(min = 0, max = B, style = 3)
    
    for(b in 1:B){
        
        # generate data
        data <- fun(n[i], type, nsplits)
        x <- as.matrix(data$x)
        y <- as.vector(data$y)
        
        # compute derivatives
        y0.hat <- mean(y)
        g <- sapply(y, dloss, y.hat=y0.hat)
        h <- sapply(y, ddloss, y.hat=y0.hat)
        ch0 <- mean((g+h*y0.hat)^2) / mean(h)/n[i]
        
        # Train gbtree
        tree <- new(GBTREE)
        tree$train(g, h, x, 1)
        
        # calculate loss
        pred_tr <- y0.hat + tree$predict_data(x)
        train[b] <- loss(y, y0.hat) - loss(y, pred_tr) 
        
        # Calculate optimism
        tree_opt_leaf <- tree$getTreeBias()
        t <- feature_to_unif(x)
        max_double_sq_bb <- emax_bb(t)
        ch1 <- tree_opt_leaf/2 * max_double_sq_bb
        
        # adjusted lossred
        adjusted[b] <- train[b] + ch0 - ch1
        
        # Test reduction in loss
        for(k in 1:m){
            testdata <- fun(n[i], type, nsplits)
            xte <- as.matrix(testdata$x)
            yte <- as.matrix(testdata$y)
            
            pred_te <- y0.hat + tree$predict_data(xte)
            test_tmp[k] <- loss(yte, y0.hat) - loss(yte, pred_te)
            
        }
        test[b] <- mean(test_tmp)
        
        # 5-fold cv
        # Generate indices of holdout observations
        holdout <- split(sample(1:n[i]), 1:5)
        
        # Train CV-gbtrees
        for(k in 1:5){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv5_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv5[b] <- mean(cv5_tmp)
        
        # 10-fold cv
        holdout <- split(sample(1:n[i]), 1:10)
        
        # Train CV-gbtrees
        for(k in 1:10){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cv10_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cv10[b] <- mean(cv10_tmp)
        
        # n-fold cv
        holdout <- split(sample(1:n[i]), 1:n[i])
        
        # Train CV-gbtrees
        for(k in 1:n[i]){
            
            n_holdout <- length(holdout[[k]])
            
            # train
            # h0
            y0.hat.cv <- mean(y[-holdout[[k]]])
            gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
            hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
            
            # h1
            tree_k <- new(GBTREE)
            tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
            
            # predict holdout
            pred_holdout <- tree_k$predict_data(x[holdout[[k]],,drop=F])
            
            # cv loss reduction
            cvn_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], y0.hat.cv+pred_holdout)
            
        }
        
        # Calculate CV-implied optimism
        cvn[b] <- mean(cvn_tmp)
        
        # update progress bar
        setTxtProgressBar(pb, b)
        
    }
    
    result[[i]] <- data.frame("5-fold cv"=cv5, "10-fold cv"=cv10, "n-fold cv"=cvn,
                              "adjusted"=adjusted, "test1000"=test, "train"=train)
    names(result[[i]]) <- c("5-fold cv", "10-fold cv", "n-fold cv", "adjusted", "test1000", "train")
    
    close(pb)
    
}

if(F){
    save(result, file="figures/loss_reduction_simple_sim_plot_round.RData")
}
load("results/loss_reduction_simple_sim_plot_round.RData")


# scale to n
#result[[1]] <- result[[1]] * n[1]
#result[[2]] <- result[[2]] * n[2]
#result[[3]] <- result[[3]] * n[3]

# The palette with black:
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")



labels_custom <- list( "$5$-fold CV",  #TeX("$5$-fold CV"), 
                       "$10$-fold CV",  #TeX("$10$-fold CV"),
                       "$n$-fold CV",  #TeX("$n$-fold CV"),
                       "$\\tilde{\\mathcal{R}}^0$", #TeX("$\\tilde{R}^0$"),
                       "$\\mathcal{R}^0$",  #TeX("$R^0$"),
                       "$\\mathcal{R}$")  #TeX("$R$") )
#c("10-fold cv", "n-fold cv", "adjusted", "test1000", "train")

xlim <- c(-20,20)
xlim1 <- xlim/n[1]
xlim2 <- xlim/n[2]
xlim3 <- xlim/n[3]

p1 <- result[[1]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=3, breaks=seq(-0.5,1,by=0.1)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=30$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim1[1],xlim1[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p1

p2 <- result[[2]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=3, breaks=seq(0.0,0.5,by=0.03)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=100$") +
    scale_fill_manual(values = cbp2, labels=labels_custom) + 
    #xlim(xlim2[1],xlim2[2]) + 
    theme_bw() + 
    theme(axis.text.y = element_blank())
p2

p3 <- result[[3]] %>% melt() %>%
    ggplot(aes( x=value, y=variable, fill=variable)) + 
    #geom_density_ridges_gradient(scale=3, rel_min_height = 0.001) + 
    geom_density_ridges2(aes(fill=variable), stat="binline", scale=3, breaks=seq(0.15,0.35,by=0.015)) + 
    labs(x="Loss reduction", y="", fill='') + 
    ggtitle("$n=1000$") + 
    scale_fill_manual(values = cbp2, labels=labels_custom) +
    #xlim(xlim3[1],xlim3[2]) + 
    theme_bw() +
    theme(axis.text.y = element_blank())
#theme(legend.position = "none")
p3



plots <- ggarrange(p1, p2, p3, p3, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
plots$`1`

if(F){

    #pdf("../../../gbtree_information/figures/loss_reduction_simple_round_sim.pdf", width=8, height=4, paper="special")
    plots$`1`
    #dev.off()
    
    # TIKZ
    #library(tikzDevice)
    #options(tz="CA")
    #tikz(file = "../../../gbtree_information/figures/loss_reduction_sim_simple_round.tex", width = 6.5, height = 3.5)
    print(plots$`1`)
    #dev.off()
    
    
}





