# # Creation of Caravan datasets
# if(F){
#     # Store Wage data into gbtorch R-package
#     library(ISLR)
#     data(Caravan)
#     dim(Caravan)
#     Caravan = na.omit(Caravan)
#     dim(Caravan)
#     n_full <- nrow(Caravan)
#     ind_train <- sample(n_full, 0.7*n_full)
#     data <- model.matrix(Purchase~., data=Caravan)[,-1]
#     dim(data)
#     x.train <- as.matrix(data[ind_train, ])
#     y.train <- as.matrix(ifelse(Caravan[ind_train, "Purchase"]=="Yes",1,0))
#     x.test <- as.matrix(data[-ind_train, ])
#     y.test <- as.matrix(ifelse(Caravan[-ind_train, "Purchase"]=="Yes", 1, 0))
#     
#     caravan.train <- list(y=y.train, x=x.train)
#     caravan.test <- list(y=y.test, x=x.test)
#     
#     library(devtools)
#     use_data(caravan.train, overwrite=T)
#     use_data(caravan.test, overwrite=T)
#     
# }