# save-load a model

# generate data
set.seed(123)
n <- 1000
x.tr <- as.matrix(runif(n, -10,10))
y.tr <- rnorm(n,sin(x.tr),0.3)
plot(x.tr, y.tr)

# train model
mod <- gbt.train(y.tr, x.tr, verbose=1, learning_rate = 0.2)
pred.tr <- predict(mod, x.tr)
points(x.tr, pred.tr, col=2)

# save model 1 -- saved using std::fixed
gbt.save(mod, "mod_save_example.txt")

# load model -- roundoff error can occur
mod2 <- gbt.load("mod_save_example.txt")
pred.tr2 <- predict(mod2, x.tr)
points(x.tr, pred.tr2, col=3)
