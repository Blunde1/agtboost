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
#' ## Load data
#' data(caravan.train, package = "gbtorch")
#' train <- caravan.train
#' mod <- gbt.train(train$y, train$x, loss_function = "logloss", verbose=10)
#' feature_names <- colnames(train$x)
#' imp <- gbt.importance(feature_names, mod)
#' imp
#'
#' @rdname gbt.importance
#' @export
gbt.importance <- function(feature_names, object)
{
    m <- length(feature_names)
    importance_vec <- object$importance(m);
    names(importance_vec) <- feature_names
    
    # Plot
    importance_vec <- importance_vec[order(importance_vec, decreasing = FALSE)]
    importance_vec <- importance_vec[importance_vec != 0]
    old.par <- par(mar=c(0,0,0,0))
    par(las=2)
    par(mar=c(5,6.5,3.5,2))
    barplot(importance_vec, main=NULL, horiz = TRUE, names.arg = names(importance_vec), 
            cex.names = 0.8, xlab = "Importance in percent")
    par(las=1)
    mtext(side=3, line=1.5, at=-0.02, adj=0, cex=1.1, "Feature importance")
    par(old.par)
    
    # Return val
    importance_vec <- importance_vec[order(importance_vec, decreasing = TRUE)]
    
    
    return(importance_vec)
}