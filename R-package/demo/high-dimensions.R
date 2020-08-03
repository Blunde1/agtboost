# P >> N Comparison to lm(), Ridge and Lasso regression (glmnet)
# 15.10.2019
# Berent Lunde

# load agtboost and other libraries needed
library(agtboost)
library(glmnet)
library(ggplot2)
library(dplyr)
library(gridExtra)

# One dimensional case ####
# Generate data
set.seed(314)
cat("Generate data")
n <- 200
x.train <- as.matrix( runif(n, 0, 4))
y.train <- rnorm(n, x.train[,1], 1)
x.test <- as.matrix( runif(n, 0, 4))
y.test <- rnorm(n, x.test[,1], 1)

# train model
cat("Training an agtboost model on one feature")
mod <- gbt.train(y.train, x.train, verbose=10, gsub_compare =T, learning_rate = 0.01)
mod$get_param() # standard parameters
mod$get_num_trees()
pred.test <- predict( mod, x.test)
plot(x.test,y.test, main="Predictions on test-data")
points(x.test,pred.test,col=2)

# lm
cat("Comparing with a linear model (holds the truth)")
mod.lm <- lm(y~., data=data.frame(y=y.train, x=x.train))
pred.test.lm <- predict(mod.lm, newdata = data.frame(y=y.test, x=x.test))
points(x.test, pred.test.lm, col=3) # obviously better


# p >> n: dependence ####

# Create new design matrix for p>>n problem
cat("Adding 9999 correlated/dependent features to the training and test sets")
cat("The data is not independent, but predictive to a different degree")
cat("Importantly, all features are predictive through the first feature!")

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

cat("The dimensions of training and test:")
dim(x.train2); dim(x.test2) 

cat("The data is not independent, but noisy to a different degree")
df_sub <- data.frame(x1=x.train2[,1], x50=x.train2[,50], #x100=x.train2[,100],
                     x200=x.train2[,200], x1000=x.train2[,1000], x10000=x.train2[,10000])
pairs(~., data=df_sub)

cat("Training a second agtboost model: give it a few seconds")
mod2 <- gbt.train(y.train, x.train2, verbose=1, gsub_compare=T)
pred.test2 <- predict( mod2, x.test2 )


# Training a GLMNET ridge
cat("Training a second (penalized) linear regression: Ridge with glmnet")
mod.ridge <- cv.glmnet(y=y.train, x=x.train2, alpha=0, nfolds=5, lambda=exp(seq(-2,3,length.out=100)))
plot(mod.ridge)
pred.ridge <- predict(mod.ridge, x.test2, s=mod.ridge$lambda.min)

# plotting
cat("All features are predictive through the first feature. \n
    Thus, any adaption to randomness outside of the first feature is adaption to noise!")

theme_set(theme_bw())
cols <- c("Observations"="black", 
          "lm-1d predictions"="orange", 
          "ridge-p>>n predictions"="red", 
          "gbt-1d predictions" = "green",
          "gbt-p>>n predictions" = "blue")
preds <- data.frame(predictive_feature <- x.test, test_obs <- y.test, pred.lm = pred.test.lm, 
           pred.ridge = pred.ridge, pred.gbt.1d = pred.test, pred.gbt.md = pred.test2)

plot1 <- preds %>%
    ggplot() +
    geom_point(aes(predictive_feature, test_obs, colour="Observations"), size=1 ) + 
    geom_point(aes(predictive_feature, pred.lm, colour="lm-1d predictions"), size=2) +  
    geom_point(aes(predictive_feature, pred.ridge, colour="ridge-p>>n predictions"), size=2) + 
    xlab("x") +
    ylab("y") +
    scale_color_manual(name=NULL, values=cols) + 
    theme(legend.position=c(.25, .8),legend.background=element_rect(colour='black'))

plot2 <- preds %>%
    ggplot() +
    geom_point(aes(predictive_feature, test_obs, colour="Observations"), size=1 ) + 
    geom_point(aes(predictive_feature, pred.gbt.1d, colour="gbt-1d predictions"), size=2) +  
    geom_point(aes(predictive_feature, pred.gbt.md, colour="gbt-p>>n predictions"), size=2) + 
    xlab("x") +
    ylab("y") +
    scale_color_manual(name=NULL, values=cols) + 
    theme(legend.position=c(.25, .8),legend.background=element_rect(colour='black'))

grid.arrange(plot1, plot2, ncol=2, 
             top="Test data and predictions, vs. the predictive feature: 
             Ridge seem to adapt slightly more to noise than agtboost")
    

# p >> n: independence ####
cat("What about when features are independent?")

x.train3 <- x.train
x.test3 <- x.test
pb <- txtProgressBar(min = 0, max = mdim, style = 3)
for(i in 1:mdim){
    x.train3 <- cbind(x.train3, rnorm(n, 0, 1) )
    x.test3 <- cbind(x.test3,  rnorm(n, 0, 1) )
    setTxtProgressBar(pb, i)
}
close(pb)

cat("The dimensions of training and test:")
dim(x.train3); dim(x.test3) 

cat("Training a third agtboost model: give it a few seconds")
mod3 <- gbt.train(y.train, x.train3, verbose=1, gsub_compare=T)
pred.test3 <- predict( mod3, x.test3 )

# Training a GLMNET Lasso
cat("Training a third (penalized) linear regression: Ridge with glmnet")
mod.lasso <- cv.glmnet(y=y.train, x=x.train3, alpha=1, nfolds=5, lambda=exp(seq(-5,3,length.out=100)))
plot(mod.lasso)
pred.lasso <- predict(mod.lasso, x.test3, s=mod.lasso$lambda.min)

# plotting
cat("No other than the first feature is predictive!")

cols <- c("Observations"="black", 
          "lm-1d predictions"="orange", 
          "lasso-p>>n predictions"="red", 
          "gbt-1d predictions" = "green",
          "gbt-p>>n predictions" = "blue")
preds <- data.frame(predictive_feature <- x.test, test_obs <- y.test, pred.lm = pred.test.lm, 
                    pred.lasso = pred.lasso, pred.gbt.1d = pred.test, pred.gbt.md = pred.test3)

plot3 <- preds %>%
    ggplot() +
    geom_point(aes(predictive_feature, test_obs, colour="Observations"), size=1 ) + 
    geom_point(aes(predictive_feature, pred.lm, colour="lm-1d predictions"), size=2) +  
    geom_point(aes(predictive_feature, pred.lasso, colour="lasso-p>>n predictions"), size=2) + 
    xlab("x") +
    ylab("y") +
    scale_color_manual(name=NULL, values=cols) + 
    theme(legend.position=c(.25, .8),legend.background=element_rect(colour='black'))

plot4 <- preds %>%
    ggplot() +
    geom_point(aes(predictive_feature, test_obs, colour="Observations"), size=1 ) + 
    geom_point(aes(predictive_feature, pred.gbt.1d, colour="gbt-1d predictions"), size=2) +  
    geom_point(aes(predictive_feature, pred.gbt.md, colour="gbt-p>>n predictions"), size=2) + 
    xlab("x") +
    ylab("y") +
    scale_color_manual(name=NULL, values=cols) + 
    theme(legend.position=c(.25, .8),legend.background=element_rect(colour='black'))

grid.arrange(plot3, plot4, ncol=2, 
             top="Test data and predictions, vs. the predictive feature: 
             Lasso seem to adapt similarly to noise in comparison to agtboost")

cat("The information theoretic approach behind agtboost is developed under independence assumptions")
cat("But luckily also works under dependence. :) ")
