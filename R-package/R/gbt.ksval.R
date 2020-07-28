#' Kolmogorov-Smirnov validation of model
#'
#' \code{gbt.ksval} transforms observations to U(0,1) if the model
#' is correct and performs a Kolmogorov-Smirnov test for uniformity.
#'
#' @param object Object or pointer to object of class \code{ENSEMBLE}
#' @param y Observations to be tested
#' @param x design matrix for training. Must be of type \code{matrix}.
#'
#' @details
#' 
#' Model validation of model passed as \code{object} using observations \code{y}.
#' Assuming the loss is a negative log-likelihood and thus a probabilistic model, 
#' the transformation 
#' \deqn{u = F_Y(y;x,\theta) \sim U(0,1),}
#' is usually valid. 
#' One parameter, \eqn{\mu=g^{-1}(f(x))}, is given by the model. Remaining parameters 
#' are estimated globally over feature space, assuming they are constant.
#' This then allow the above transformation to be exploited, so that the 
#' Kolmogorov-Smirnov test for uniformity can be performed.
#' 
#' If the response is a count model (\code{poisson} or \code{negbinom}), the transformation
#' \deqn{u_i = F_Y(y_i-1;x,\theta) + Uf_Y(y_i,x,\theta), ~ U \sim U(0,1)}
#' is used to obtain a continuous transformation to the unit interval, which, if the model is
#' correct, will give standard uniform random variables.
#'
#' @return
#' Kolmogorov-Smirnov test of model
#' 
#' @examples
#' ## Gaussian regression:
#' x_tr <- as.matrix(runif(500, 0, 4))
#' y_tr <- rnorm(500, x_tr, 1)
#' x_te <- as.matrix(runif(500, 0, 4))
#' y_te <- rnorm(500, x_te, 1)
#' mod <- gbt.train(y_tr, x_tr)
#' gbt.ksval(mod, y_te, x_te)
#'
#' @rdname gbt.ksval
#' @export
gbt.ksval <- function(object, y, x)
{
    
    # Check input
    
    
    # Model specifics
    loss_type <- object$get_loss_function()
    mu_pred <- predict(mod, x)
    
    
    # cdf transform
    n <- length(y)
    u <- numeric(n)
    if(loss_type == "mse")
    {
        cat("Gaussian regression \n")
        cat("Assuming constant variance \n")
        lsigma <- 0.0
        lsigma <- stats::nlminb(lsigma, nll_norm, y=y, mu_pred=mu_pred)$par
        cat("Variance estimate is: ", exp(2*lsigma), "\n")
        u <- pnorm(y, mean=mu_pred, sd=exp(lsigma))
    }else if(loss_type %in% c("gamma::neginv", "gamma::log"))
    {
        cat("Gamma regression \n")
        cat("Assuming constant shape \n")
        lshape <- 0.0
        lshape <- stats::nlminb(lshape, nll_gamma, y=y, mu_pred=mu_pred)$par
        cat("Shape estimate is: ", exp(lshape), "\n")
        u <- pgamma(y, shape=exp(lshape), scale=mu_pred/exp(lshape))
    }else if(loss_type == "negbinom")
    {
        cat("Overdispersed count (negative binomial) regression \n")
        cat("Assuming constant dispersion \n")
        ldisp <- 0.0
        ldisp <- stats::nlminb(ldisp, nll_nbinom, y=y, mu_pred=mu_pred)$par
        cat("Dispersion estimate is: ", exp(ldisp), "\n")
        u <- unbinom(y, mu=mu_pred, dispersion=exp(ldisp))
    }else if(loss_type == "poisson")
    {
        cat("Count (Poisson) regression \n")
        u <- upois(y, lambda=mu_pred)
    }else if(loss_type == "logloss")
    {
        cat("Classification \n")
        u <- ubernoulli(y, p=mu_pred)   
    }
    
    # ks.test and histogram
    res <- ks.test(u, "punif")
    hist(u, freq=FALSE, oma=c(2, 3, 5, 2)+0.1, main=NULL, xlab="CDF transformed observations")
    mytitle="Histogram: Model-CDF transfored observations" 
    mysubtitle=paste0(res$method, ": ", format(res$p.value))
    mtext(side=3, line=3, at=-0.07, adj=0, cex=1, mytitle)
    mtext(side=3, line=2, at=-0.07, adj=0, cex=0.7, mysubtitle)
    
    return(res)
}


# --- Likelihoods ---

# negative binomial nll
nll_nbinom <- function(ldisp, y, mu_pred)
{
    disp <- exp(ldisp)
    -sum(dnbinom(y, size=disp, mu=mu_pred, log=TRUE))
}

# normal nll
nll_norm <- function(lsigma, y, mu_pred)
{
    sigma <- exp(lsigma)
    -sum(dnorm(y, mean=mu_pred, sd=sigma, log=TRUE))
}

# gamma nll
nll_gamma <- function(lshape, y, mu_pred)
{
    shape <- exp(lshape)
    scale <- mu_pred / shape
    -sum(dgamma(y, shape=shape, scale=scale, log=TRUE))
}


# --- Uniform count transforms ---

# transform binomial r.v. to U(0,1)
# assuming dispersion
unbinom <- function(X, mu, dispersion)
{
    n <- length(X)
    U <- rep(0,n)
    v <- runif(n)
    for(i in 1:n){
        U[i] <- pnbinom(X[i]-1, size=dispersion, mu=mu[i]) + v[i]*dnbinom(X[i], size=dispersion, mu=mu[i])
    }
    # avoid exact zeroes and unity
    ind0 <- U==0
    U[ind0] <- U[ind0] + 1e-9*runif(1)
    ind1 <- U==1
    U[ind1] <- U[ind1] - 1e-9*runif(1)
    return(U)
}

# Converts Poisson count data to continuous uniform (0,1)
upois <- function(X, lambda){
    
    n <- length(X)
    U <- rep(0, n)
    v <- runif(n)
    for(i in 1:n)
    {
        U[i] <- ppois(X[i]-1, lambda[i]) + v[i]*dpois(X[i], lambda[i])
    }
    # avoid exact zeroes and unity
    ind0 <- U==0
    U[ind0] <- U[ind0] + 1e-9*runif(1)
    ind1 <- U==1
    U[ind1] <- U[ind1] - 1e-9*runif(1) # U to avoid ties
    return(U)
}

# Converts Bernooulli RVs to continuous uniform (0,1)
ubernoulli <- function(X, p)
{
    n <- length(X)
    U <- numeric(n)
    v <- runif(n)
    ind_zero <- X==0
    U[ind_zero] = v[ind_zero]*(1-p[ind_zero])
    U[!ind_zero] = v[!ind_zero]*p[!ind_zero]
    return(U)
}
