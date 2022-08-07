#' Load an aGTBoost Model
#'
#' \code{gbt.load} is an interface for loading a \pkg{agtboost} model.
#'
#' @param file Valid file-path to a stored aGTBoost model
#'
#' @details
#' 
#' The load function for \pkg{agtboost}.
#' Loades a GTB model from a txt file.
#' 
#'
#' @return
#' Trained aGTBoost model.
#'
#' @seealso
#' \code{\link{gbt.save}}
#'
#'
#' @rdname gbt.load
#' @export
gbt.load <- function(file)
{
    # check valid file path
    
    if (grepl("rds$", file)) {
        mod <- readRDS(file)
        mod
    } else {
        mod <- new(ENSEMBLE)
        mod$load_model(file)
        mod
    }
}
