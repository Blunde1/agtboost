#' aGTBoost Prediction
#'
#' \code{predict} is an interface for predicting from a \pkg{agtboost} model.
#'
#' @param object Object or pointer to object of class \code{ENSEMBLE}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
#' @param ... additional parameters passed. Currently not in use.
#'
#' @details
#' 
#' The prediction function for \pkg{agtboost}.
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
#' Berent Ånund Strømnes Lunde, Tore Selland Kleppe and Hans Julius Skaug,
#' "An Information Criterion for Automatic Gradient Tree Boosting", 2020, 
#' \url{https://arxiv.org/abs/2008.05926}
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
        "Error: object must be a agtboost ensemble \n",
        "Error: agtboost ensemble must be trained, see function documentation gbt.train \n",
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
    
    # Check if transformation of input
    # Get and check input
    offset <- rep(0, nrow(newdata))
    input_list <- list(...)
    type <- ""
    if(length(input_list)>0){
        if("type" %in% names(input_list)){
            type <- input_list$type
            if(!(type %in% c("response", "link_response"))){
                warning(paste0("Ignoring unknown input type: ", type))
                type <- ""
            }else{
                warning(paste0("Ignoring unknown input: ", names(input_list)))
            }
        }
        if("offset" %in% names(input_list)){
            offset <- input_list$offset
        }
    }
    
    # predict
    pred <- object$predict(newdata, offset)
    res <- NULL
    
    if(type %in% c("", "response")){
        
        # This is default: Predict g^{-1}(preds)
        # Get link function
        loss_fun_type <- object$get_loss_function()
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
        
        if(link_type == "identity"){
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
        res <- pred
    }
    return(res)
} 


#' aGTBoost Count-Regression Auto Prediction
#'
#' \code{predict} is an interface for predicting from a \code{agtboost} model.
#'
#' @param object Object or pointer to object of class \code{GBT_ZI_MIX}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
#' @param ... additional parameters passed. Currently not in use.
#'
#' @details
#' 
#' The prediction function for \code{agtboost}.
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
#' Berent Ånund Strømnes Lunde, Tore Selland Kleppe and Hans Julius Skaug,
#' "An Information Criterion for Automatic Gradient Tree Boosting", 2020, 
#' \url{https://arxiv.org/abs/2008.05926}
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
        "Error: object must be a agtboost GBT_COUNT_AUTO \n",
        "Error: agtboost model must be trained, see function documentation gbt.train \n",
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
