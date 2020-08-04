library(gbtorch)
library(dplyr)
library(reshape2)
library(ggplot2)
library(tidyr)
library(gridExtra)
library(ggpubr)

## A simple gtb.train example with linear regression:
set.seed(2)
n <- 1000
x <- matrix(runif(n, 0, 4), ncol=1)
y <- rnorm(n, x, 1)
x.test <- matrix(runif(n, 0, 4), ncol=1)
y.test <- rnorm(n, x.test, 1)

learning_rate <- 0.01
mod <- gbt.train(y, x, learning_rate = learning_rate, greedy_complexities = F, verbose=1)
mod$get_num_trees()
mod$get_num_leaves()
y.pred <- predict( mod, x.test  )

plot(x.test[,1], y.test)
points(x.test[,1], y.pred, col="red")

mod2 <- gbt.train(y, x, learning_rate = learning_rate, greedy_complexities = F, 
                  force_continued_learning = TRUE, nrounds = 5000, verbose = 100)
y.pred2 <- predict( mod2, x.test )
points(x.test[,1], y.pred2, col=3)

mean((y.test-y.pred)^2)
mean((y.test-y.pred2)^2)

ntrees <- seq(1, max(mod2$get_num_trees()), by=1)
losste <- losstr <- numeric(length(ntrees))
pb <- txtProgressBar(min=1, max=length(ntrees), style=3)
for(i in 1:length(losste)){
    predte <- mod2$predict2( as.matrix(x.test), ntrees[i] )
    predtr <- mod2$predict2( as.matrix(x), ntrees[i] )
    losstr[i] <- mean((y - predtr)^2)
    losste[i] <- mean((y.test-predte)^2)
    setTxtProgressBar(pb, i)
}
close(pb)

lossgr_est <- sapply(1:mod2$get_num_trees(), mod2$estimate_generalization_loss)
nleaves <- mod2$get_num_leaves()
df <- data.frame(
    ntrees, losstr, losste, nleaves#, lossgr_est
)
names(df) <- c("Number of trees in ensemble", "Training loss", "Test loss", "Number of leaves")#, "Gen loss est")
colnames(df)

# colours and legend
cbp2 <- c("#000000", "#E69F00", "#56B4E9", "#009E73",
          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
names(cbp2) <- colnames(df)[-1]

ntrees[which.min(df$`Test loss`)]
mod$get_num_trees()

df_long <- df %>%
    gather(type, loss,  'Training loss':'Test loss')
p <- df_long %>%
    ggplot() + 
    geom_point(aes(x=`Number of trees in ensemble` , y=loss, colour=type)) + 
    geom_line(aes(x=`Number of trees in ensemble` , y=loss, colour=type)) + 
    geom_vline(xintercept=which.min(df$`Test loss`), colour="orange", size=1) +
    geom_text(data=data.frame(x=ntrees[which.min(df$`Test loss`)], y=2.3), map=aes(x=x, y=y), 
              label="Min test loss", hjust=-0.1 ) + 
    geom_vline(xintercept=mod$get_num_trees(), colour="black", size=1) +
    geom_text(data=data.frame(x=mod$get_num_trees(), y=2.3), map=aes(x=x, y=y), 
              label="Stopping criterion", hjust=1.1 ) + 
    scale_x_continuous(trans='log2', limits=c(1,5000)) +
    ylab("Loss") + 
    #ylim(0.8, 2.5) + 
    scale_color_manual(name=NULL, 
                       values=cbp2) +
    theme_bw() +
    theme(legend.position = c(0.2, 0.3), 
          legend.background = element_rect(fill=alpha('white', 0)))
    #theme(legend.position = "bottom")
p   

df2 <- data.frame("Number of trees in ensemble"=1:mod$get_num_trees(), "Number of leaves"=mod$get_num_leaves())
colnames(df2) <- c("k'th tree in ensemble", "Number of leaves")
p2 <- df2 %>% 
    ggplot(aes(x=`k'th tree in ensemble`, y=`Number of leaves`)) + 
    geom_point() + 
    geom_line() +
    #scale_x_continuous(trans='log10') + 
    theme_bw()
p2

gridExtra::grid.arrange(p,p2, ncol=2, legend)
ggpubr::ggarrange(p, p2, ncol=2)#, common.legend = FALSE, legend="bottom")

if(F){
    # TIKZ
    #library(tikzDevice)
    #options(tz="CA")
    #tikz(file = "boost_convergence.tex", width = 6.5, height = 3)
    #print(p)
    #print(ggpubr::ggarrange(p, p2, ncol=2, common.legend = FALSE, legend="bottom"))
    #dev.off()
}
if(F){
    # PDF
    #pdf("../../../gbtree_information/figures/boost_convergence.pdf", width=8, height=3.5, paper="special")
    ggpubr::ggarrange(p, p2, ncol=2)
    #dev.off()
    
}
