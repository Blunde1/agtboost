# greedy trees vs greedy ensemble
# Berent Lunde
# 26.09.2019

# Description: Illustrates the difference of standard gradient tree boosting 
# which is implemented in V1 GBTorch (greedy_complexities = FALSE) and the 
# V2 GBTorch greedy complexities algorithm (greedy_complexities = TRUE).
# References to future paper is coming...!


# Load GBTorch library
library(tidyverse)
library(agtboost)


# Simulate data
## A simple gtb.train example with linear regression:
set.seed(123)
x <- as.matrix(runif(100, 0, 5))
y <- rnorm(100, 1*x, 1)
x.test <- as.matrix(runif(100, 0, 5))
y.test <- rnorm(100, 1*x.test, 1)


# -- Train models --
greedy_tree_mod <- gbt.train(y, x, verbose=1, algorithm="vanilla")
greedy_complexities_mod <- gbt.train(y, x, verbose=1, algorithm="global_subset")


# -- Predict on test --
# greedy tree
y.pred.vanilla <- predict( greedy_tree_mod, as.matrix( x.test ) )
plot(x.test, y.test)
points(x.test, y.pred.vanilla, col="red")
mean((y.test - y.pred.vanilla)^2)
# greedy complexities
y.pred.modified <- predict( greedy_complexities_mod, as.matrix( x.test ) )
#plot(x.test, y.test)
points(x.test, y.pred.modified, col="blue")
mean((y.test - y.pred.modified)^2)


# Plotting
p_data <- data.frame(y=y, x=x) %>%
    ggplot(aes(x=x, y=y), colour="black") + 
    geom_point() + 
    theme_bw()
p_data

pred_mod1 <- function(x, k){
    greedy_tree_mod$predict2(as.matrix(x), k)
}
pred_mod2 <- function(x, k){
    greedy_complexities_mod$predict2(as.matrix(x), k)
}


# ordinary model plots
k1 <- ceiling(seq(30,greedy_tree_mod$get_num_trees(), length.out = 3))
p1_mod1 <- p_data + 
    stat_function(fun=pred_mod1,args = list(k=k1[1]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k1[1])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p2_mod1 <- p_data + 
    stat_function(fun=pred_mod1,args = list(k=k1[2]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k1[2])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p3_mod1 <- p_data + 
    stat_function(fun=pred_mod1,args = list(k=k1[3]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k1[3])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))

# new model plots
k2 <- ceiling(seq(30,greedy_complexities_mod$get_num_trees(), length.out = 3))
p1_mod2 <- p_data + 
    stat_function(fun=pred_mod2,args = list(k=k2[1]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k2[1])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p2_mod2 <- p_data + 
    stat_function(fun=pred_mod2,args = list(k=k2[2]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k2[2])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))
p3_mod2 <- p_data + 
    stat_function(fun=pred_mod2,args = list(k=k2[3]), colour="blue", size=1) + 
    ggtitle(paste0("Iteration ",k2[3])) + 
    theme(plot.title = element_text(margin = margin(t = 10, b=-20)))

gridExtra::grid.arrange(
    p1_mod1,p2_mod1,p3_mod1,
    p1_mod2,p2_mod2,p3_mod2,
    ncol=3)

if(F)
{
    pdf(file="../../../../gbtorch-package-article/figures/greedy_fun_fits.pdf", width = 10,height = 5)
    gridExtra::grid.arrange(
        p1_mod1,p2_mod1,p3_mod1,
        p1_mod2,p2_mod2,p3_mod2,
        ncol=3)
    dev.off()
}



# -- ILLUSTRATION --
# Greedy tree algorithm
plot(x,y)
k1 <- greedy_tree_mod$get_num_trees()
for(k in ceiling(seq(1,k1, length.out = 4))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- greedy_tree_mod$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}

# Greedy complexities algorithm
plot(x,y)
k2 <- greedy_complexities_mod$get_num_trees()
for(k in ceiling(seq(1,k2, length.out = 10))){
    # pred
    cat("Predictions from the ", k, " first trees in the ensemble \n")
    preds <- greedy_complexities_mod$predict2(x, k)
    points(x,preds, col=k)
    Sys.sleep(1)
}
