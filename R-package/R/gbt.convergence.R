#' Convergence of agtboost model.
#'
#' \code{gbt.convergence} calculates loss of data over iterations in the model
#'
#' @param object Object or pointer to object of class \code{ENSEMBLE}
#' @param y response vector
#' @param x design matrix for training. Must be of type \code{matrix}.
#'
#' @details
#' 
#' Computes the loss on supplied data at each boosting iterations of the model passed as object.
#' This may be used to visually test for overfitting on test data, or the converce, to check for underfitting
#' or non-convergence.
#'
#' @return
#' \code{vector} with $K+1$ elements with loss at each boosting iteration and at the first constant prediction
#' 
#' @examples
#' ## Gaussian regression:
#' x_tr <- as.matrix(runif(500, 0, 4))
#' y_tr <- rnorm(500, x_tr, 1)
#' x_te <- as.matrix(runif(500, 0, 4))
#' y_te <- rnorm(500, x_te, 1)
#' mod <- gbt.train(y_tr, x_tr)
#' convergence <- gbt.convergence(mod, y_te, x_te)
#' which.min(convergence) # Should be fairly similar to boosting iterations + 1
#' mod$get_num_trees() +1 # num_trees does not include initial prediction
#'
#' @importFrom graphics mtext plot
#' @rdname gbt.convergence
#' @export
gbt.convergence <- function(object, y, x)
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
    
    loss_vec <- object$convergence(y, x)
    
    # Plot
    plot(1:length(loss_vec), loss_vec,
         main=NULL,
         xlab="Boosting iterations",
         ylab="Loss")
    mtext(side=3, line=1.5, at=-0.02, adj=0, cex=1.1, "Convergence of model")
    
    return(loss_vec)
}