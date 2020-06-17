#' GBTorch Prediction
#'
#' \code{predict} is an interface for predicting from a \code{gbtorch} model.
#'
#' @param object Object or pointer to object of class \code{ENSEMBLE}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
#' @param ... additional parameters passed. Currently not in use.
#'
#' @details
#' 
#' The prediction function for \code{gbtorch}.
#' Using the generic \code{predict} function in R is also possible, using the same arguments.
#' 
#'
#' @return
#' For regression or binary classification, it returns a vector of length \code{nrows(newdata)}.
#'
#' @seealso
#' \code{\link{gbt.train}}
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
#' 
#' ## predict is overloaded
#' y.pred <- predict( mod, as.matrix( x.test ) )
#' 
#' plot(x.test, y.test)
#' points(x.test, y.pred, col="red")
#'
#'
#' @rdname predict.Rcpp_ENSEMBLE
#' @export

#' @export
predict.Rcpp_ENSEMBLE <- function(object, newdata, ...){
    # object - pointer to class ENSEMBLE
    # newdata - design matrix of type matrix
    
    # checks on newdata and e.ptr
    error_messages <- c()
    error_messages_type <- c(
        "Error: object must be a GBTorch ensemble \n",
        "Error: GBTorch ensemble must be trained, see function documentation gbt.train \n",
        "Error: newdata must be a matrix \n"
    )
    # check object
    if(class(object)!="Rcpp_ENSEMBLE"){
        error_messages <- c(error_messages, error_messages_type[1])
    }else{
        # test if trained
        if(object$get_num_trees()==0)
            error_messages <- c(error_messages, error_messages_type[2])
    }
    
    # check x
    if(!is.matrix(newdata))
        error_messages <- c(error_messages, error_messages_type[3])
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # predict
    pred <- object$predict(newdata)
    res <- NULL
    
    # Check if transformation of input
    # Get and check input
    input_list <- list(...)
    type <- ""
    if(length(input_list)>0){
        if("type" %in% names(input_list)){
            type <- input_list$type
            if(!(type %in% c("response", "link_response"))){
                warning(paste0("Ignoring unknown input type: ", type))
                type <- ""
            }
        }else{
            warning(paste0("Ignoring unknown input: ", names(input_list)))
        }
    }
    
    if(type %in% c("", "response")){
        
        # This is default: Predict g^{-1}(preds)
        # Get link function
        loss_fun_type <- object$get_param()$loss_function
        link_type = ""
        if(loss_fun_type %in% c("mse")){
            link_type = "identity"
        }else if(loss_fun_type %in% c("logloss")){
            link_type = "logit"
        }else if(loss_fun_type %in% c("poisson", "gamma::log", "negbinom")){
            link_type = "log"
        }else if(loss_fun_type %in% c("gamma::neginv")){
            link_type = "neginv"
        }else{
            # if no match
            warning(paste0("No link-function match for loss: ", loss_fun_type, " Using identity"))
            link_type = "identity"
        }
        
        if(link_type == "mse"){
            res <- pred
        }else if(link_type == "logit"){
            res <- 1/(1+exp(-pred))
        }else if(link_type == "log"){
            res <- exp(pred)
        }else if(link_type == "neginv"){
            res <- -1/pred
        }
    }else if(type == "link_response"){
        # predict response on log (link) level
        res <- object$predict(newdata)
    }
    
    
    return(res)
} 

#' GBTorch Zero-Inflated Mixture Prediction
#'
#' \code{predict} is an interface for predicting from a \code{gbtorch} model.
#'
#' @param object Object or pointer to object of class \code{GBT_ZI_MIX}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
#' @param ... additional parameters passed. Currently not in use.
#'
#' @details
#' 
#' The prediction function for \code{gbtorch}.
#' Using the generic \code{predict} function in R is also possible, using the same arguments.
#' 
#'
#' @return
#' For regression or binary classification, it returns a vector of length \code{nrows(newdata)}.
#'
#' @seealso
#' \code{\link{gbt.train}}
#'
#' @references
#'
#' B. Å. S. Lunde, T. S. Kleppe and H. J. Skaug, "An information criterion for gradient boosted trees"
#' publishing details,
#'
#' @examples
#' ## A simple gtb.train example with linear regression:
#' ## Random generation of zero-inflated poisson
#' rzip <- function(n, lambda, prob){
#' bin <- rbinom(n, 1, prob)
#' y <- numeric(n)
#' for(i in 1:n){
#'     y[i] <- ifelse(bin[i]==1, 0, rpois(1, lambda))
#' }
#' return(y)
#' }
#' 
#' prob <- 0.4
#' lambda <- 3
#' y <- rzip(1000, lambda, prob)
#' x <- as.matrix(runif(1000, 1,5))
#' 
#' mod <- gbt.train(y, x, loss_function = "zero_inflation::poisson", verbose=1)
#' pred <- predict(mod, x)
#' range(pred) # mean predictions
#' (1-prob)*lambda # true mean
#' 
#' plot(y, pred, ylim=c(0,max(y)))
#'
#' @rdname predict.Rcpp_GBT_ZI_MIX
#' @export

#' @export
predict.Rcpp_GBT_ZI_MIX <- function(object, newdata, ...){
    # object - pointer to class ENSEMBLE
    # newdata - design matrix of type matrix
    
    # checks on newdata and e.ptr
    error_messages <- c()
    error_messages_type <- c(
        "Error: object must be a GBTorch GBT_ZI_MIX \n",
        "Error: GBTorch model must be trained, see function documentation gbt.train \n",
        "Error: newdata must be a matrix \n"
    )
    # check object
    if(class(object)!="Rcpp_GBT_ZI_MIX"){
        error_messages <- c(error_messages, error_messages_type[1])
    }#else{
    #   # test if trained
    #   if(object$get_num_trees()==0)
    #       error_messages <- c(error_messages, error_messages_type[2])
    #}
    
    # check x
    if(!is.matrix(newdata))
        error_messages <- c(error_messages, error_messages_type[3])
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # Get input
    input_list <- list(...)
    type <- ""
    if(length(input_list)>0){
        if("type" %in% names(input_list)){
            type <- input_list$type
            if(!(type %in% c("response", "separate"))){
                warning(paste0("Ignoring unknown input type: ", type))
                type <- ""
            }
        }else{
            warning(paste0("Ignoring unknown input: ", names(input_list)))
        }
    }
    
    if(type %in% c("", "response")){
        # predict mean
        res <- object$predict(newdata)
    }else if(type == "separate"){
        # predict probability and non-zero mean separate
        res <- object$predict_separate(newdata)
        colnames(res) <- c("probability", "non_zero_mean")
    }
    
    

    return(res)
} 



#' GBTorch Count-Regression Auto Prediction
#'
#' \code{predict} is an interface for predicting from a \code{gbtorch} model.
#'
#' @param object Object or pointer to object of class \code{GBT_ZI_MIX}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
#' @param ... additional parameters passed. Currently not in use.
#'
#' @details
#' 
#' The prediction function for \code{gbtorch}.
#' Using the generic \code{predict} function in R is also possible, using the same arguments.
#' 
#'
#' @return
#' For regression or binary classification, it returns a vector of length \code{nrows(newdata)}.
#'
#' @seealso
#' \code{\link{gbt.train}}
#'
#' @references
#'
#' B. Å. S. Lunde, T. S. Kleppe and H. J. Skaug, "An information criterion for gradient boosted trees"
#' publishing details,
#'
#' @examples
#' ## A simple gtb.train example with linear regression:
#' ## Random generation of zero-inflated poisson
#' 2+2
#'
#' @rdname predict.Rcpp_GBT_COUNT_AUTO
#' @export

#' @export
predict.Rcpp_GBT_COUNT_AUTO <- function(object, newdata, ...){
    # object - pointer to class ENSEMBLE
    # newdata - design matrix of type matrix
    
    # checks on newdata and e.ptr
    error_messages <- c()
    error_messages_type <- c(
        "Error: object must be a GBTorch GBT_COUNT_AUTO \n",
        "Error: GBTorch model must be trained, see function documentation gbt.train \n",
        "Error: newdata must be a matrix \n"
    )
    # check object
    if(class(object)!="Rcpp_GBT_COUNT_AUTO"){
        error_messages <- c(error_messages, error_messages_type[1])
    }#else{
    #   # test if trained
    #   if(object$get_num_trees()==0)
    #       error_messages <- c(error_messages, error_messages_type[2])
    #}
    
    # check x
    if(!is.matrix(newdata))
        error_messages <- c(error_messages, error_messages_type[3])
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # Get input
    input_list <- list(...)
    type <- ""
    if(length(input_list)>0){
        if("type" %in% names(input_list)){
            type <- input_list$type
            if(!(type %in% c("response", "link_response"))){
                warning(paste0("Ignoring unknown input type: ", type))
                type <- ""
            }
        }else{
            warning(paste0("Ignoring unknown input: ", names(input_list)))
        }
    }
    
    if(type %in% c("", "response")){
        # predict mean
        res <- object$predict(newdata)
        res <- exp(res)
    }else if(type == "link_response"){
        # predict response on log (link) level
        res <- object$predict(newdata)
    }
    
    
    
    return(res)
} 
