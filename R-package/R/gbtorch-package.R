#' \code{gbtorch} is a lightning fast gradient boosting library designed to avoid 
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
#' \item \code{gtb.train}: function for training a \code{gbtorch} ensemble
#' \item \code{gtb.pred}: function for predicting from a \code{gbtorch} ensemble
#' }
#' 
#' See individual function documentation for usage.
#'
#' @docType package
#' @name gbtorch
#' @title Adaptive and automatic gradient boosting computations.
#'
#' @author Berent Ånund Strømnes Lunde
#' @useDynLib gbtorch
NULL