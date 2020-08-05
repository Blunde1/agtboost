# Iterative tree-procedure

# tree boost
# linreg sim
# const pred
# pred tree
# pred from f^(k-1)
# f_k = (f^(k) - f^(k-1)) / delta 
# errors can be computed afterwards after sim from f^(k-1)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(agtboost)
n <- 100
d <- 0.01
set.seed(123)
x <- as.matrix(runif(n,0,5))
y <- rnorm(n, x, 1)
df <- data.frame(x,y)

p <- df %>%
    ggplot() + 
    geom_point(aes(x, y), shape=15, colour="black", size=1.5, alpha=1) + 
    theme_bw()
p

f0 <- mean(y)
f <- f0
pred <- rep(f, n)
pred.f <- function(x){
    f
}
p + stat_function(fun = pred.f, colour="black", size=1.5) 

#parameters
param <- list("learning_rate" = d, "loss_function" = "mse", "nrounds"=5000)
mod <- gbt.train(y=y, x=x, learning_rate = d, verbose=1, gsub_compare = FALSE)
pred.f.gbt <- function(x){
    mod$predict(as.matrix(x))
}
p_gbt_full <- p + stat_function(fun=pred.f.gbt, colour="blue", size=1) + 
    ggtitle("Full model fit") + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p_gbt_full

# convergence
ntrees <- seq(1, max(mod$get_num_trees()), by=1)
losstr <- numeric(length(ntrees))
pb <- txtProgressBar(min=1, max=length(ntrees), style=3)
for(i in 1:length(losstr)){
    predtr <- mod$predict2( as.matrix(x), ntrees[i] )
    losstr[i] <- mean((y - predtr)^2)
    setTxtProgressBar(pb, i)
}
close(pb)
df <- data.frame("Number of trees in ensemble"=ntrees, "Training loss"=losstr) 
names(df) = c("Number of trees in ensemble", "Training loss")
p_convergence <- 
    df %>%
    ggplot(aes(x=`Number of trees in ensemble`, y=`Training loss`)) + 
    #geom_point() + 
    geom_line() +
    #ggtitle("Convergence") + 
    theme_bw() + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p_convergence

# number of trees
num_leaves <- mod$get_num_leaves()
df2 <- data.frame("Number of trees in ensemble"=1:mod$get_num_trees(), "Number of leaves"=mod$get_num_leaves())
colnames(df2) <- c("k'th tree in ensemble", "Number of leaves")
p_nleaves <- df2 %>% 
    ggplot(aes(x=`k'th tree in ensemble`, y=`Number of leaves`)) + 
    geom_point() + 
    geom_line() +
    #scale_x_continuous(trans='log10') + 
    theme_bw()
p_nleaves
gridExtra::grid.arrange(p_gbt_full,p_convergence,p_nleaves, ncol=3)

K <- mod$get_num_trees()
res.list <- list()
p.res.list <- list()
for(k in 1:K){
    res.list[[k]] <- (y-pred)
    
    p.res.list[[k]] <- data.frame(Residuals=res.list[[k]], x) %>%
        ggplot() +     
        geom_point(aes(x, Residuals), shape=15, colour="black", size=1.5, alpha=1) + 
        theme_bw()
    
    # update pred
    pred <- mod$predict2(x, k)
}

pred.f.tree <- function(x, k){
    #pred <- rep(f,n)
    if(k==1){
        (mod$predict2(as.matrix(x),1)-f) / d
    }else{
        (mod$predict2(as.matrix(x),k)-mod$predict2(as.matrix(x),k-1)) / d
    }
}
pred.f.gbt <- function(x, k){
    mod$predict2(as.matrix(x), k)
}

p_tree_1 <- p.res.list[[1]] + 
    stat_function(fun=pred.f.tree,args = list(k=1), colour="blue", size=1) + 
    ggtitle("First tree fit") + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p_tree_2 <- p.res.list[[100]] + 
    stat_function(fun=pred.f.tree,args = list(k=100), colour="blue", size=1) + 
    ggtitle("100'th tree fit") + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p_tree_3 <- p.res.list[[200]] + 
    stat_function(fun=pred.f.tree,args = list(k=200), colour="blue", size=1) + 
    ggtitle("200'th tree fit") + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
gridExtra::grid.arrange(p_tree_1,p_tree_2,p_tree_3,ncol=3)

gridExtra::grid.arrange(
    p_gbt_full,p_convergence,p_nleaves,
    p_tree_1,p_tree_2,p_tree_3,
    ncol=3)

if(F)
{
    pdf(file="../../../../gbtorch-package-article/figures/tree_model_fits.pdf", width = 10,height = 5)
    gridExtra::grid.arrange(
        p_gbt_full,p_convergence,p_nleaves,
        p_tree_1,p_tree_2,p_tree_3,
        ncol=3)
    dev.off()
}

