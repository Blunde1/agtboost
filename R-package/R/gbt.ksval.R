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
#' @importFrom graphics hist mtext
#' @importFrom stats dgamma dnbinom dnorm dpois ks.test pgamma pnbinom pnorm ppois runif dbinom pbinom
#' @rdname gbt.ksval
#' @export
gbt.ksval <- function(object, y, x)
{
    
    # Check input
    error_messages <- c()
    error_messages_type <- c(
        "object_type" = "Error: object must be a GBTorch ensemble \n",
        "model_not_trained" = "Error: GBTorch ensemble must be trained, see function documentation gbt.train \n",
        "response_not_vec" = "Error: y must be a vector of type numeric or matrix with dimension 1 \n",
        "dmat_not_mat" = "Error: x must be a matrix \n",
        "y_x_correspondance" = "Error: length of y must correspond to the number of rows in x \n"
    )
    # check object
    if(class(object)!="Rcpp_ENSEMBLE"){
        error_messages <- c(error_messages, error_messages_type["object_type"])
    }else{
        # test if trained
        if(object$get_num_trees()==0)
            error_messages <- c(error_messages, error_messages_type["model_not_trained"])
    }
    
    # Check y, x
    if(!is.vector(y, mode="numeric")){
        if(is.matrix(y) && ncol(y)>1 ){
            error_messages <- c(error_messages, error_messages_type["response_not_vec"])
        }
    }
    if(!is.matrix(x))
        error_messages <- c(error_messages, error_messages_type["dmat_not_mat"])
    # dimensions
    if(length(y) != nrow(x))
        error_messages <- c(error_messages, error_messages_type["y_x_correspondance"])
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    
    # Model specifics
    loss_type <- object$get_loss_function()
    mu_pred <- predict(object, x)
    
    # cdf transform
    n <- length(y)
    u <- numeric(n)
    if(loss_type == "mse")
    {
        msg1 <- c("Gaussian regression \n")
        msg2 <- c("Assuming constant variance \n")
        lsigma <- 0.0
        lsigma <- stats::nlminb(lsigma, nll_norm, y=y, mu_pred=mu_pred)$par
        msg3 <- c("Variance estimate is: ", exp(2*lsigma), "\n")
        message(msg1, msg2, msg3)
        u <- pnorm(y, mean=mu_pred, sd=exp(lsigma))
    }else if(loss_type %in% c("gamma::neginv", "gamma::log"))
    {
        msg1 <- c("Gamma regression \n")
        msg2 <- c("Assuming constant shape \n")
        lshape <- 0.0
        lshape <- stats::nlminb(lshape, nll_gamma, y=y, mu_pred=mu_pred)$par
        msg3 <- c("Shape estimate is: ", exp(lshape), "\n")
        message(msg1, msg2, msg3)
        u <- pgamma(y, shape=exp(lshape), scale=mu_pred/exp(lshape))
    }else if(loss_type == "negbinom")
    {
        msg1 <- c("Overdispersed count (negative binomial) regression \n")
        msg2 <- c("Assuming constant dispersion \n")
        ldisp <- 0.0
        ldisp <- stats::nlminb(ldisp, nll_nbinom, y=y, mu_pred=mu_pred)$par
        msg3 <- c("Dispersion estimate is: ", exp(ldisp), "\n")
        message(msg1, msg2, msg3)
        u <- unbinom(y, mu=mu_pred, dispersion=exp(ldisp))
    }else if(loss_type == "poisson")
    {
        message("Count (Poisson) regression \n")
        u <- upois(y, lambda=mu_pred)
    }else if(loss_type == "logloss")
    {
        message("Classification \n")
        u <- ubernoulli(y, p=mu_pred)   
    }
    
    # ks.test and histogram
    res <- ks.test(u, "punif")
    hist(u, freq=FALSE, oma=c(2, 3, 5, 2)+0.1, main=NULL, xlab="CDF transformed observations")
    mytitle="Histogram: Model-CDF transformed observations" 
    mysubtitle=paste0(res$method, ": ", format(res$p.value))
    mtext(side=3, line=2.2, at=-0.07, adj=0, cex=1.1, mytitle)
    mtext(side=3, line=1.2, at=-0.07, adj=0, cex=0.8, mysubtitle)
    
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
    U <- pbinom(X-1, 1, p) + runif(n)*dbinom(X, 1, p)
    return(U)
}
