#' GBTorch Training.
#'
#' \code{gbt.train} is an interface for training a \code{gbtorch} model.
#'
#' @param param the list of parameters:
#'
#' 1. Task Parameters
#'
#' \itemize{
#'   \item \code{loss_function} specify the learning objective (loss function). Only pre-specified loss functions are currently supported.
#'   \itemize{
#'   \item \code{mse} regression with squared error loss (Default).
#'   \item \code{logloss} logistic regression for binary classification, output score before logistic transformation.
#'   }
#' }
#'
#' 2. Tree Booster Parameters
#'
#' \itemize{
#'   \item \code{learning_rate} control the learning rate: scale the contribution of each tree by a factor of \code{0 < learning_rate < 1} when it is added to the current approximation. Lower value for \code{learning_rate} implies an increase in the number of boosting iterations: low \code{learning_rate} value means model more robust to overfitting but slower to compute. Default: 0.01
#'   \item \code{nrounds} a just-in-case max number of boosting iterations. Default: 5000
#' }
#'
#'
#' @param y response vector for training. Must correspond to the design matrix \code{x}.
#' @param x design matrix for training. Must be of type \code{matrix}.
#' @param verbose Boolean: Enable boosting tracing information? Default: \code{TRUE}.
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
#' An object of class \code{ENSEMBLE} with the following elements:
#' \itemize{
#'   \item \code{handle} a handle (pointer) to the xgboost model in memory.
#'   \item \code{initialPred} a field containing the initial prediction of the ensemble.
#'   \item \code{set_param} function for changing the parameters of the ensemble.
#'   \item \code{get_param} function for looking up the parameters of the ensemble.
#'   \item \code{train} function for re-training (or from scratch) the ensemble directly on vector \code{y} and design matrix \code{x}.
#'   \item \code{predict} function for predicting observations given a design matrix
#'   \item \code{predict2} function as above, but takes a parameter max number of boosting ensemble iterations.
#'   \item \code{get_ensemble_bias} function for calculating the (approximate) optimism of the ensemble.
#'   \item \code{get_num_trees} function returning the number of trees in the ensemble.
#' }
#'
#' @seealso
#' \code{\link{gbt.pred}}
#'
#' @references
#'
#' B. Å. S. Lunde, T. S. Kleppe and H. J. Skaug, "An information criterion for gradient boosted trees"
#' publishing details, \url{}
#'
#' @examples
#' ## A simple gtb.train example with linear regression:
#' x <- runif(500, 0, 4)
#' y <- rnorm(500, x, 1)
#' x.test <- runif(500, 0, 4)
#' y.test <- rnorm(500, x.test, 1)
#' 
#' param <- list("learning_rate" = 0.03, "loss_function" = "mse", "nrounds"=2000)
#' mod <- gbt.train(param, y, as.matrix(x))
#' y.pred <- predict( mod, as.matrix( x.test ) )
#' 
#' plot(x.test, y.test)
#' points(x.test, y.pred, col="red")
#'
#'
#' @rdname gbt.train
#' @export
gbt.train <- function(param = list(), y, x, verbose=TRUE){
    
    error_messages <- c()
    error_messages_type <- c(
        "Error: y must be a vector of type numeric or matrix with dimension 1 \n",
        "Error: x must be a matrix \n",
        "Error: length of y must correspond to the number of rows in x \n",
        "Error: param must be provided as a list \n",
        "Error: learning_rate in param must be a number between 0 and 1 \n",
        "Error: loss_function in param must be a valid loss function. See documentation for valid parameters \n",
        "Error: nrounds in param must be an integer >= 1 \n"
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
    
    # Check param else default
    if(!is.list(param))
        error_messages <- c(error_messages, error_messages_type[4])
    
    if("learning_rate" %in% names(param)){
        if(is.numeric(param$learning_rate) && length(param$learning_rate) == 1){
            if(0 < param$learning_rate && param$learning_rate <=1){}else{
                error_messages <- c(error_messages, error_messages_type[5])
            }   
        }else{
            error_messages <- c(error_messages, error_messages_type[5])
        }
    }else{
        error_messages <- c(error_messages, error_messages_type[5])
    }
    
    if("loss_function" %in% names(param)){
        if(is.character(param$loss_function) && length(param$loss_function) == 1){
            if(
                param$loss_function %in% c("mse", "logloss")
            ){}else{
                error_messages <- c(error_messages, error_messages_type[6])
            }   
        }else{
            error_messages <- c(error_messages, error_messages_type[6])
        }
    }else{
        error_messages <- c(error_messages, error_messages_type[6])
    }
    
    
    if("nrounds" %in% names(param)){
        if(is.numeric(param$nrounds) && length(param$nrounds) == 1){
            if(param$nrounds >= 1){}else{
                error_messages <- c(error_messages, error_messages_type[7])
            }   
        }else{
            error_messages <- c(error_messages, error_messages_type[7])
        }
    }else{
        error_messages <- c(error_messages, error_messages_type[7])
    }
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # create gbtorch ensemble object
    mod <- new(ENSEMBLE)
    mod$set_param(param)
    
    # train ensemble
    mod$train(y,x, verbose)
    
    # return trained gbtorch ensemble
    return(mod)
}
