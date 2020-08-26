#' Importance of features in a model.
#'
#' \code{gbt.importance} creates a \code{data.frame} of feature importance in a model 
#'
#' @param feature_names character vector of feature names
#' @param object object or pointer to object of class \code{ENSEMBLE}
#'
#' @details
#' 
#' Sums up "expected reduction" in generalization loss (scaled using \code{learning_rate}) 
#' at each node for each tree in the model, and attributes it to 
#' the feature the node is split on. Returns result in terms of percents.
#'
#' @return
#' \code{data.frame} with percentwise reduction in loss of total attributed to each feature.
#' 
#' @examples
#' \donttest{
#' ## Load data
#' data(caravan.train, package = "agtboost")
#' train <- caravan.train
#' mod <- gbt.train(train$y, train$x, loss_function = "logloss", verbose=10)
#' feature_names <- colnames(train$x)
#' imp <- gbt.importance(feature_names, mod)
#' imp
#' }
#' 
#' @importFrom graphics barplot mtext par
#' @rdname gbt.importance
#' @export
gbt.importance <- function(feature_names, object)
{
    # Check input
    error_messages <- c()
    error_messages_type <- c(
        "object_type" = "Error: object must be a GBTorch ensemble \n",
        "model_not_trained" = "Error: GBTorch ensemble must be trained, see function documentation gbt.train \n"
    )
    # check object
    if(class(object)!="Rcpp_ENSEMBLE"){
        error_messages <- c(error_messages, error_messages_type["object_type"])
    }else{
        # test if trained
        if(object$get_num_trees()==0)
            error_messages <- c(error_messages, error_messages_type["model_not_trained"])
    }
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    
    m <- length(feature_names) # should have a check that $m > max_j in ensemble$
    importance_vec <- object$importance(m);
    names(importance_vec) <- feature_names
    
    # Plot
    importance_vec <- importance_vec[order(importance_vec, decreasing = FALSE)]
    importance_vec <- importance_vec[importance_vec != 0] * 100
    opar <- par(no.readonly =TRUE)       # code line i
    on.exit(par(opar))                   # code line i+1
    par(las=2)
    par(mar=c(5,6.5,3.5,2))
    barplot(importance_vec, main=NULL, horiz = TRUE, names.arg = names(importance_vec), 
            cex.names = 0.8, xlab = "Importance in percent")
    par(las=1)
    mtext(side=3, line=1.5, at=-0.02, adj=0, cex=1.1, "Feature importance")

    # Return val
    importance_vec <- importance_vec[order(importance_vec, decreasing = TRUE)]
    
    
    return(importance_vec)
}