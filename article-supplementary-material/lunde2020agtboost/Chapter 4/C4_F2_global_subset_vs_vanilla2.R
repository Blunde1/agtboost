setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
library(agtboost)

nsize <- c(1000, 10000, 100000)
convergence <- list()
nleaves <- list()
for(i in 1:length(nsize))
{
    # generate date
    n <- nsize[i]
    d <- 0.01
    set.seed(123)
    x <- as.matrix(runif(n,0,5))
    y <- rnorm(n, x, 1)
    xte <- as.matrix(runif(n,0,5))
    yte <- rnorm(n, xte, 1)
    
    # modelling
    mod_vanilla <- gbt.train(y=y, x=x, learning_rate = d, verbose=100, algorithm = "vanilla")
    mod_gsub <- gbt.train(y=y, x=x, learning_rate = d, verbose=100, algorithm = "global_subset")
    
    # convergence
    losste_v100 <- gbt.convergence(mod_vanilla, y, x)
    losste_g100 <- gbt.convergence(mod_gsub, y, x)
    
    df <- data.frame("Iteration"= c(1:length(losste_v100), 1:length(losste_g100)),
                     "Test loss"= c(losste_v100, losste_g100),
                     "Model"= c(rep("Vanilla", length(losste_v100)), rep("Global-subset", length(losste_g100))))
    names(df) <- c("Iteration", "Test loss", "Model")
    
    p_convergence <- df %>% ggplot(aes(x=`Iteration`, y=`Test loss`, group=`Model`)) +
        geom_line(aes(linetype=`Model`, color=`Model`))+
        ggtitle(paste0("n=",format(n,scientific = FALSE))) + 
        theme_bw() + 
        theme(legend.title = element_blank(),
              legend.position = c(0.75, 0.8),
              legend.background = element_rect(fill="transparent"))
    
    # number of trees
    num_leaves_v <- mod_vanilla$get_num_leaves()
    num_leaves_g <- mod_gsub$get_num_leaves()
    df2 <- data.frame("iter"=c(1:mod_vanilla$get_num_trees(), 1:mod_gsub$get_num_trees()), 
                      "nleaves"= c(mod_vanilla$get_num_leaves(), mod_gsub$get_num_leaves()),
                      "Model"= c(rep("Vanilla", length(num_leaves_v)), rep("Global-subset", length(num_leaves_g)))
    )
    colnames(df2) <- c("k'th tree in ensemble", "Number of leaves", "Model")
    p_nleaves <- df2 %>% 
        ggplot(aes(x=`k'th tree in ensemble`, y=`Number of leaves`, group=`Model`)) + 
        geom_point(aes(colour=`Model`)) + 
        geom_line(aes(linetype=`Model`, colour=`Model`)) +
        theme_bw() + 
        theme(legend.title = element_blank(),
              legend.position = c(0.75, 0.8),
              legend.background = element_rect(fill="transparent"))

    # Store
    convergence[[i]] <- p_convergence
    nleaves[[i]] <- p_nleaves
}

gridExtra::grid.arrange(
    convergence[[1]], convergence[[2]], convergence[[3]],
    nleaves[[1]], nleaves[[2]], nleaves[[3]],
    ncol=3)

if(F)
{
    pdf(file="../../../../gbtorch-package-article/figures/global_subset_illustration.pdf", width = 10,height = 5)
    gridExtra::grid.arrange(
        convergence[[1]], convergence[[2]], convergence[[3]],
        nleaves[[1]], nleaves[[2]], nleaves[[3]],
        ncol=3)
    dev.off()
}