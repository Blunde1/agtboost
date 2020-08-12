#' Adaptive and Automatic Gradient Boosting Computations
#' 
#' \pkg{agtboost} is a lightning fast gradient boosting library designed to avoid 
#' manual tuning and cross-validation by utilizing an information theoretic 
#' approach. This makes the algorithm adaptive to the dataset at hand; it is 
#' completely automatic, and with minimal worries of overfitting. 
#' Consequently, the speed-ups relative to state-of-the-art implementations 
#' are in the thousands while mathematical and technical knowledge required 
#' on the user are minimized.
#' 
#' Important functions:
#' 
#' \itemize{
#' \item \code{\link{gbt.train}}: function for training an \pkg{agtboost} ensemble
#' \item \code{\link{predict.Rcpp_ENSEMBLE}}: function for predicting from an \pkg{agtboost} ensemble
#' }
#' 
#' See individual function documentation for usage.
#'
#' @docType package
#' @name agtboost
#' @title Adaptive and automatic gradient boosting computations.
#'
#' @author Berent Ånund Strømnes Lunde
#' @useDynLib agtboost, .registration = TRUE
NULL