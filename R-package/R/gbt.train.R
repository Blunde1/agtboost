#' GBTorch Training.
#'
#' \code{gbt.train} is an interface for training a \code{gbtorch} model.
#'
#' @param y response vector for training. Must correspond to the design matrix \code{x}.
#' @param x design matrix for training. Must be of type \code{matrix}.
#' @param learning_rate control the learning rate: scale the contribution of each tree by a factor of \code{0 < learning_rate < 1} when it is added to the current approximation. Lower value for \code{learning_rate} implies an increase in the number of boosting iterations: low \code{learning_rate} value means model more robust to overfitting but slower to compute. Default: 0.01
#' @param loss_function specify the learning objective (loss function). Only pre-specified loss functions are currently supported.
#'   \itemize{
#'   \item \code{mse} regression with squared error loss (Default).
#'   \item \code{logloss} logistic regression for binary classification, output score before logistic transformation.
#'   \item \code{poisson} Poisson regression for count data using a log-link, output score before natural transformation.
#'   \item \code{gamma::neginv} gamma regression using the canonical negative inverse link. Scaling independent of y.
#'   \item \code{gamma::log} gamma regression using the log-link. Constant information parametrisation. 
#'   \item \code{negbinom} Negative binomial regression for count data with overdispersion. Log-link.
#'   \item \code{poisson::zip} Conditional Zero-Inflated Poisson (ZIP) regression, for modelling the Poisson intensity in a ZIP regression model. Log-link.
#'   \item \code{zero_inflation::poisson} Zero-inflated Poisson. Mean predictions.
#'   \item \code{zero_inflation::negbinom} Zero-inflated negative binomial (Poisson-gamma mixture). Mean predictions.
#'   \item \code{zero_inflation::auto} Zero inflation that automatically chooses between the ordinary Poisson and a mixture as the conditional count process. Mean predictions.
#'   \item \code{count::auto} Chooses automatically between Poisson or negative binomial regression.
#'   }
#' @param nrounds a just-in-case max number of boosting iterations. Default: 50000
#' @param verbose Enable boosting tracing information at i-th iteration? Default: \code{0}.
#' @param gsub_compare Boolean: Global-subset comparisons. \code{FALSE} means standard GTB, \code{TRUE} compare subset-splits with global splits (next root split). Default: \code{TRUE}.
#' @param previous_pred prediction vector for training. Boosted training given predictions from another model.
#' @param weights weights vector for scaling contributions of individual observations. Default \code{NULL} (the unit vector).
#' @param force_continued_learning Boolean: \code{FALSE} (default) stops at information stopping criterion, \code{TRUE} stops at \code{nround} iterations.
#'
#' @details
#' These are the training functions for \code{gbtorch}.
#' 
#' Explain the philosophy and the algorithm and a little math
#' 
#' \code{gbt.train} learn trees with adaptive complexity given by an information criterion, 
#' until the same (but scaled) information criterion tells the algorithm to stop. The data used 
#' for training at each boosting iteration stems from a second order Taylor expansion to the loss 
#' function, evaluated at predictions given by ensemble at the previous boosting iteration.
#' 
#' Formally, ....
#'
#' @return
#' An object of class \code{ENSEMBLE} or \code{GBT_ZI_MIX} with some or all of the following elements:
#' \itemize{
#'   \item \code{handle} a handle (pointer) to the gbtorch model in memory.
#'   \item \code{initialPred} a field containing the initial prediction of the ensemble.
#'   \item \code{set_param} function for changing the parameters of the ensemble.
#'   \item \code{get_param} function for looking up the parameters of the ensemble.
#'   \item \code{train} function for re-training (or from scratch) the ensemble directly on vector \code{y} and design matrix \code{x}.
#'   \item \code{predict} function for predicting observations given a design matrix
#'   \item \code{predict2} function as above, but takes a parameter max number of boosting ensemble iterations.
#'   \item \code{estimate_generalization_loss} function for calculating the (approximate) optimism of the ensemble.
#'   \item \code{get_num_trees} function returning the number of trees in the ensemble.
#' }
#'
#' @seealso
#' \code{\link{predict.Rcpp_ENSEMBLE}}
#'
#' @references
#'
#' B. Å. S. Lunde, T. S. Kleppe and H. J. Skaug, "An information criterion for gradient boosted trees"
#' publishing details, 
#'
#' @examples
#' ## A simple gtb.train example with linear regression:
#' x <- runif(500, 0, 4)
#' y <- rnorm(500, x, 1)
#' x.test <- runif(500, 0, 4)
#' y.test <- rnorm(500, x.test, 1)
#' 
#' mod <- gbt.train(y, as.matrix(x))
#' y.pred <- predict( mod, as.matrix( x.test ) )
#' 
#' plot(x.test, y.test)
#' points(x.test, y.pred, col="red")
#'
#'
#' @rdname gbt.train
#' @export
#' @importFrom methods new
gbt.train <- function(y, x, learning_rate = 0.01,
                      loss_function = "mse", nrounds = 50000,
                      verbose=0, gsub_compare=TRUE, 
                      previous_pred=NULL,
                      weights = NULL,
                      force_continued_learning=FALSE,
                      ...){
    
    error_messages <- c()
    error_messages_type <- c(
        "\n Error: y must be a vector of type numeric or matrix with dimension 1",
        "\n Error: x must be a matrix",
        "\n Error: length of y must correspond to the number of rows in x \n",
        #"Error: param must be provided as a list \n",
        "\n Error: learning_rate must be a number between 0 and 1 \n",
        "\n Error: loss_function must be a valid loss function. See documentation for valid parameters \n",
        "\n Error: nrounds must be an integer >= 1 \n",
        "\n Error: verbose must be of type numeric with length 1",
        "\n Error: gsub_compare must be of type logical with length 1",
        "\n Error: previous_pred must be a vector of type numeric",
        "\n Error: previous_pred must correspond to length of y",
        "\n Error: force_continued_learning must be of type logical with length 1",
        "negbinom"= "\n Error: if loss_function is 'negbinom', dispersion must be provided in ..."
    )
    # Check y, x
    if(!is.vector(y, mode="numeric")){
        if(is.matrix(y) && ncol(y)>1 ){
            error_messages <- c(error_messages, error_messages_type[1])
        }
    }
    if(!is.matrix(x))
        error_messages <- c(error_messages, error_messages_type[2])
    # dimensions
    if(length(y) != nrow(x))
        error_messages <- c(error_messages, error_messages_type[3])
    
    # learning_rate
    if(is.numeric(learning_rate) && length(learning_rate)==1){
        # ok
        if(0 < learning_rate && learning_rate <=1){
            #ok
        }else{
            #error
            error_messages <- c(error_messages, error_messages_type[4])
        }
    }else{
        #error
        error_messages <- c(error_messages, error_messages_type[4])
    }
    
    # loss function
    if(is.character(loss_function) && length(loss_function) == 1){
        if(
            loss_function %in% c("mse", "logloss", "poisson", "gamma::neginv", 
                                 "gamma::log", "negbinom", 
                                 "poisson::zip", "zero_inflation", "zero_inflation::poisson",
                                 "zero_inflation::negbinom", "zero_inflation::auto",
                                 "count::auto")
        ){}else{
            error_messages <- c(error_messages, error_messages_type[5])
        }   
    }else{
        error_messages <- c(error_messages, error_messages_type[5])
    }
    
    # nrounds
    if(is.numeric(nrounds) && length(nrounds) == 1){
        if(nrounds >= 1){}else{
            error_messages <- c(error_messages, error_messages_type[6])
        }   
    }else{
        error_messages <- c(error_messages, error_messages_type[6])
    }
    
    
    # verbose
    if(is.numeric(verbose) && length(verbose)==1){
        #ok
    }else{
        error_messages <- c(error_messages, error_messages_type[7])
    }

    # gsub_compare
    if(is.logical(gsub_compare) && length(gsub_compare)==1){
        #ok
    }else{
        # error
        error_messages <- c(error_messages, error_messages_type[8])
    }
    
    if(!is.null(previous_pred)){
        if(!is.vector(y, mode="numeric")){
            if(is.matrix(y) && ncol(y)>1 ){
                error_messages <- c(error_messages, error_messages_type[9])
            }
        }
        # dimensions
        if(length(previous_pred) != nrow(x))
            error_messages <- c(error_messages, error_messages_type[10])
    }
    
    # force_continued_learning
    if(is.logical(force_continued_learning) && length(force_continued_learning)==1){
        #ok
    }else{
        # error
        error_messages <- c(error_messages, error_messages_type[11])
    }
    
    # Check for dispersion if negbinom
    if(loss_function=="negbinom"){
        input_list <- list(...)
        dispersion <- input_list$dispersion
        if(is.null(dispersion) || !is.numeric(dispersion) || length(dispersion)>1){
            error_messages <- c(error_messages, error_messages["negbinom"])
        }
        extra_param <- dispersion
    }else{
        extra_param <- 0 # default without meaning
    }
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # Weights vector?
    if(is.null(weights))
        weights = rep(1,nrow(x))
    
    param <- list("learning_rate" = learning_rate, 
                  "loss_function" = loss_function, 
                  "nrounds"=nrounds,
                  "extra_param" = extra_param)
    
    if(loss_function %in% c("zero_inflation::poisson", "zero_inflation::negbinom", "zero_inflation::auto")){
        
        mod <- new(GBT_ZI_MIX)
        mod$set_param(param)
        mod$train(y,x, verbose, gsub_compare)   
        
    }else if(loss_function %in% c("count::auto")){
        mod <- new(GBT_COUNT_AUTO)
        mod$set_param(param)
        mod$train(y,x, verbose, gsub_compare)   
        
    }else{
        # create gbtorch ensemble object
        mod <- new(ENSEMBLE)
        mod$set_param(param)
        
        # train ensemble
        if(is.null(previous_pred)){
            
            # train from scratch
            mod$train(y,x, verbose, gsub_compare, force_continued_learning, weights)   
        }else{
            
            # train from previous predictions
            mod$train_from_preds(previous_pred,y,x, verbose, gsub_compare, weights)
        }
        
    }
    
    
    # return trained gbtorch ensemble
    return(mod)
}
