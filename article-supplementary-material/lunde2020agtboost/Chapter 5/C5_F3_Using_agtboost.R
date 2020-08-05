# aGTBoost: Adaptive and automatic GTB computations
# Chapter 5: Using agtboost
# Berent Lunde
# 05082020

# working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Packages
library(agtboost)

# Data
data("caravan.test")
data("caravan.train")
?caravan.train # documentation
dim(caravan.train$x)

# Training
mod <- gbt.train(y=caravan.train$y, x=caravan.train$x,
                 loss_function="logloss", verbose=100)


# Predictions
prob_te <- predict(mod, caravan.test$x)

# Store plots
#setwd("C:/Users/lunde/OneDrive/Dokumenter/Projects/Github repositories/gbtorch-package-article/gbtorch-package-article-scripts")
pdf(file="../../../../gbtorch-package-article/figures/using_agtboost_validation.pdf", width = 9,height = 5)
par(mfrow=c(1,2))

# Model validation
gbt.ksval(object=mod, y=caravan.test$y, x=caravan.test$x)
#gbt.ksval(object=mod, y=caravan.train$y, x=caravan.train$x)

# Feature importance
gbt.importance(feature_names=colnames(caravan.train$x), object=mod)

dev.off()
