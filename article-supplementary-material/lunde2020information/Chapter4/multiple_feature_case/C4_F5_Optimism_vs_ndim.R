library(tidyverse)
library(latex2exp)
library(gridExtra)
library(Rcpp)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(reshape2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
Rcpp::sourceCpp("../tree_class_gumbel2.cpp")


cir_increment <- function(dt, x0){
    # dX = a*(b-X)dt + sigma*sqrt(X) dW
    # But specified:
    # kappa=4, sigma_ou=4, d=2 -- ou
    # OU = -0.5*kappa*OU*dt + 0.5*sigma_ou*dW
    # this implies in previous ou:
    # kappa_ou=1, sigma_ou = sqrt(2)
    # kappa=2, sigma=2*sqrt(2)
    # sigma = 4, a = 4, b = d*sigma^2 / (4*kappa) =  2*4^2 / ( 4*4)
    # 4*(2-X)dt + 4*sqrt(X)dW
    
    a=2
    b=1
    sigma = 2*sqrt(2)
    
    #kappa = 2; sigma=2*sqrt(2)
    #a = kappa
    #b = 2 * sigma^2 / (4*kappa)
    
    c<-2*a/(sigma^2*(1-exp(-a*dt)))
    Y<-rchisq(n=1,df=4*a*b/sigma^2,ncp=2*c*x0*exp(-a*dt))
    X<-Y/(2*c) # X = X_{t+dt}
    return(X)
}

rcir_process <- function(u){
    k <- length(u)
    eps <- 1e-15
    t_ <- 0.5*log( u*(1-eps)/(eps*(1-u)) )
    dt <- diff(c(eps,t_))
    X <- numeric(k)
    X[1] <- rgamma(1, shape=0.5, scale=2)
    if(k>1){
        for(i in 2:k){
            X[i] <- cir_increment(dt[i],X[i-1])
        }
    }
    return(X)
}

if(F){
    # prepare cir_sim
    n <- 1000 # number of cir-simulations
    m <- 1000 # number of split-points
    eps <- 1e-7
    delta <- 1/(m+1)
    cir_sim <- matrix(ncol = m, nrow = n)
    u <- seq(delta,1-delta, length.out=m)
    for(i in 1:n){
        cir_sim[i,] <- rcir_process(u)
    }
}

#MSE Loss and derivatives
loss <- function(y,y.hat){
    mean((y-y.hat)^2) 
}
dloss <- function(y, y.hat) -2*sum(y-y.hat)
ddloss <- function(y, y.hat) 2*length(y)
generateY <- function(n, age, groups, sd){
    rnorm(n,10*age, sd)
    #rnorm(n,ceiling(age),sd)
}

# perhaps use this?
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

# DGP
generate_data <- function(n,m, type=1){
    
    # continuous
    if(type==1){
        x <- matrix(rnorm(n*m), ncol = m, nrow = n)
    }
    
    # a+1 = 10
    if(type==2){
        x <- matrix(sample(1:10, n*m, replace = TRUE), ncol=m, nrow=n)
    }
    
    # one hot
    if(type==3){
        x <- matrix(rbinom(n*m,1,0.5), ncol = m, nrow = n)
    }
    y <- rnorm(n)
    list(x = x, y=y)
}

# Simulate loss reductions vs number of dimensions
# R^0: test from 1000 simulations
# R: train
# R.tilde^0: adjusted train

plots <- list()
# create new cir
# prepare cir_sim
n_sim <- 100 # number of cir-simulations
n_col <- 100 # number of split-points
delta <- 1/(n_col+1)
cir_sim <- matrix(ncol = n_col, nrow = n_sim)
u <- seq(delta,1-delta, length.out=n_col)
for(i in 1:n_sim){
    cir_sim[i,] <- rcir_process(u)
}

for(dgp_type in 1:3){
    #dgp_type <- 1
    B <- 100
    depth <- 1
    n <- 100
    m <- c(1,seq(5,100, 5))
    ntest <- 1
    lr_tr <- lr_te <- lr_adj <- data.frame(matrix(nrow=length(m), ncol=B))#  numeric(length(m))
    lr_te_tmp <- numeric(ntest)
    #pb <- txtProgressBar(min = 0, max = B, style = 3)
    for(b in 1:B){
        
        cat(b, "\n")
        pb2 <- txtProgressBar(min = 0, max = length(m), style = 3)
        
        for(j in 1:length(m)){
            
            # generate data: update
            #random_vector_tr <- rnorm(n*j)
            data <- generate_data(n,m[j], dgp_type)
            x_tr <- data$x
            y_tr <- data$y
            
            # derivatives
            y_0 <- mean(y_tr)
            g <- sapply(y_tr, dloss, y.hat=y_0)
            h <- sapply(y_tr, ddloss, y.hat=y_0)
            
            # Build tree with one split
            tree <- new(GBTREE)
            tree$train(g, h, x_tr, cir_sim, depth)
            
            # loss reduction: train
            pred_tr <- y_0 + tree$predict_data(x_tr)
            lr_tr[j,b] <- loss(y_tr, y_0) - loss(y_tr, pred_tr) 
            
            # loss reduction: adjusted train
            ch0 <- mean((g+h*y_0)^2) / mean(h)/n
            #ch1 <- tree$getFeatureMapOptimism() + tree$getConditionalOptimism()
            CR <- tree$getTreeOptimism()
            lr_adj[j,b] <- lr_tr[j,b] - CR #- ch0
            
            # loss reduction: test
            for(k in 1:ntest){
                data_te <- generate_data(n,m[j], dgp_type)
                x_te <- data_te$x
                y_te <- data_te$y
                pred_te <- y_0 + tree$predict_data(x_te)
                lr_te_tmp[k] <- loss(y_te, y_0) - loss(y_te, pred_te)
            }
            lr_te[j,b] <- mean(lr_te_tmp)
            
            setTxtProgressBar(pb2, j)
            
        }   
        close(pb2)
        
        #setTxtProgressBar(pb, b)
    }
    #close(pb)
    
    rowSD <- function(df){
        n <- nrow(df)
        sapply(1:n, function(i) sd(df[i,]))
    }
    
    df <- data.frame("m"=m,
                     "lr_tr" = rowMeans(lr_tr),
                     "lr_te" = rowMeans(lr_te),
                     "lr_adj" = rowMeans(lr_adj)
    )
    
    df2 <- data.frame("m"=m,
                      "lr_tr" = rowMeans(lr_tr),
                      "lr_tr_u" = rowMeans(lr_tr) + rowSD(lr_tr),
                      "lr_tr_l" = rowMeans(lr_tr) - rowSD(lr_tr),
                      "lr_te" = rowMeans(lr_te),
                      "lr_te_u" = rowMeans(lr_te) + rowSD(lr_te),
                      "lr_te_l" = rowMeans(lr_te) - rowSD(lr_te),
                      "lr_adj" = rowMeans(lr_adj),
                      "lr_adj_u" = rowMeans(lr_adj) + rowSD(lr_adj),
                      "lr_adj_l" = rowMeans(lr_adj) - rowSD(lr_adj)
    )
    
    #df <- data.frame(cbind(m, lr_tr, lr_te, lr_adj))
    
    df_long <- df %>%
        gather(type, reduction, lr_tr:lr_adj, factor_key=TRUE)
    head(df_long)
    
    cbp2 <- c("lr_tr" = "#000000", 
              "lr_te" = "#E69F00", 
              "lr_adj" = "#56B4E9", 
              "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
              "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
    names(cbp2) <- colnames(df)[-1]
    label_names <- c("$\\mathcal{R}$", "$\\mathcal{R}^0$", "$\\tilde{\\mathcal{R}}^0$")
    
    plots[[dgp_type]] <- df_long %>%
        ggplot() + 
        geom_point(aes(x=m, y=reduction, colour=type)) + 
        geom_line(aes(x=m, y=reduction, colour=type)) + 
        geom_ribbon(data=df2,aes(m, ymax = lr_tr_u, ymin = lr_tr_l), fill = cbp2[1], alpha= 0.30) + 
        geom_ribbon(data=df2,aes(m, ymax = lr_te_u, ymin = lr_te_l), fill = cbp2[2], alpha= 0.30) + 
        geom_ribbon(data=df2,aes(m, ymax = lr_adj_u, ymin = lr_adj_l), fill = cbp2[3], alpha= 0.30) + 
        xlab("$m$") +
        ylab("Loss reduction") +
        ggtitle("$a+1=2$") + 
        #scale_shape_manual(name = NULL,
        #                   values = c(1:20)) + 
        scale_color_manual(name=NULL, 
                           values=cbp2,
                           labels = label_names) +
        theme_bw(base_size = 12) +
        theme(legend.position = "bottom")
    
    #plots[[dgp_type]]
    
}

if(F){
    save(plots, file="results/loss_reduction_vs_dim_plots.RData")
    load("results/loss_reduction_vs_dim_plots.RData")
}
#label_names <- c("$\\mathcal{R}$", "$\\mathcal{R}^0$", "$\\tilde{\\mathcal{R}}^0$")

p1 <- plots[[1]] + ggtitle("$a+1=100$")
p2 <- plots[[2]] + ggtitle("$a+1=10$")
p3 <- plots[[3]] + ggtitle("$a+1=2$")

plot <- ggarrange(p3, p2, p1, ncol=3, nrow=1, common.legend = TRUE, legend="bottom")
plot

if(F){
    # TIKZ
    #library(tikzDevice)
    #options(tz="CA")
    #tikz(file = "../../../gbtree_information/figures/loss_reduction_vs_dim.tex", width = 6.5, height = 3)
    print(plot)
    #dev.off()
}
