# Poisson regression
# 09.11.2019
# Berent Lunde

data(mtcars)
head(mtcars)
dim(mtcars)

y.train <- mtcars[,11]
x.train <- as.matrix(mtcars[,-11])

gbt_mod <- gbt.train(y.train, x.train, loss_function = "poisson", verbose=10)
pred <- predict(gbt_mod, x.train) # predict log(lambda)

plot(pred, y.train)
