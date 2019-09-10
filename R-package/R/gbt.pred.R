#' GBTorch Prediction
#'
#' \code{gbt.pred} is an interface for predicting from a \code{gbtorch} model.
#'
#' @param object Object or pointer to object of class \code{ENSEMBLE}
#' @param newdata Design matrix of data to be predicted. Type \code{matrix}
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
#' y.pred.1 <- gbt.pred( mod, as.matrix( x.test ) )
#' 
#' ## predict i overloaded
#' y.pred.2 <- predict( mod, as.matrix( x.test ) )
#' 
#' plot(x.test, y.test)
#' points(x.test, y.pred.1, col="red")
#' points(x.test, y.pred.2, col="blue")
#'
#'
#' @rdname gbt.pred
#' @export
gbt.pred <- function(object, newdata){
    # object - pointer to class ENSEMBLE
    # newdata - design matrix of type matrix
    
    # checks on newdata and e.ptr
    
    # predict
    res <- object$predict(newdata)
    
    return(res)
}

#' @export
predict.Rcpp_ENSEMBLE <- function(object, newdata){
    # object - pointer to class ENSEMBLE
    # newdata - design matrix of type matrix
    
    # checks on newdata and e.ptr
    
    # predict
    res <- object$predict(newdata)
    
    return(res)
} 
    