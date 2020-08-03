#' Load a GBTorch Model
#'
#' \code{gbt.load} is an interface for loading a \code{agtboost} model.
#'
#' @param file Valid file-path to a stored GBTorch model
#'
#' @details
#' 
#' The load function for \code{agtboost}.
#' Loades a GBT model from a txt file.
#' 
#'
#' @return
#' Trained GBTorch model.
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
    
    mod <- new(ENSEMBLE)
    mod$load_model(file)
    return(mod)
}