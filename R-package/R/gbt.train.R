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
#'   }
#' @param nrounds a just-in-case max number of boosting iterations. Default: 50000
#' @param verbose Enable boosting tracing information at i-th iteration? Default: \code{0}.
#' @param greedy_complexities Boolean: \code{FALSE} means standard GTB, \code{TRUE} means greedy complexity tree-building. Default: \code{TRUE}.
#' @param previous_pred prediction vector for training. Boosted training given predictions from another model.
#' @param weights weights vector for scaling contributions of individual observations. Default \code{NULL} (the unit vector).
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
#' mod <- gbt.train(y, as.matrix(x))
#' y.pred <- predict( mod, as.matrix( x.test ) )
#' 
#' plot(x.test, y.test)
#' points(x.test, y.pred, col="red")
#'
#'
#' @rdname gbt.train
#' @export
gbt.train <- function(y, x, learning_rate = 0.01,
                      loss_function = "mse", nrounds = 50000,
                      verbose=0, greedy_complexities=TRUE, 
                      previous_pred=NULL,
                      weights = NULL){
    
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
        "\n Error: greedy_complexities must be of type logical with length 1",
        "\n Error: previous_pred must be a vector of type numeric",
        "\n Error: previous_pred must correspond to length of y"
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
            loss_function %in% c("mse", "logloss", "poisson", "gamma::neginv", "gamma::log")
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

    # greedy_complexities
    if(is.logical(greedy_complexities) && length(greedy_complexities)==1){
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
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # Weights vector?
    if(is.null(weights))
        weights = rep(1,nrow(x))
    
    # create gbtorch ensemble object
    mod <- new(ENSEMBLE)
    param <- list("learning_rate" = learning_rate, 
                  "loss_function" = loss_function, 
                  "nrounds"=nrounds)
    mod$set_param(param)
    
    # train ensemble
    if(is.null(previous_pred)){
        
        # train from scratch
        mod$train(y,x, verbose, greedy_complexities, weights)   
    }else{
        
        # train from previous predictions
        mod$train_from_preds(previous_pred,y,x, verbose, greedy_complexities, weights)
    }
    
    # return trained gbtorch ensemble
    return(mod)
}
