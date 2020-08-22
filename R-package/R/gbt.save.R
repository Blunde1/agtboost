#' Save an aGTBoost Model
#'
#' \code{gbt.save} is an interface for storing a \pkg{agtboost} model.
#'
#' @param gbt_model Model object or pointer to object of class \code{ENSEMBLE}
#' @param file Valid file-path
#'
#' @details
#' 
#' The model-storage function for \pkg{agtboost}.
#' Saves a GTB model as a txt file. Might be retrieved using \code{gbt.load}
#' 
#'
#' @return
#' Txt file that can be loaded using \code{gbt.load}.
#'
#' @seealso
#' \code{\link{gbt.load}}
#'
#'
#' @rdname gbt.save
#' @export
gbt.save <- function(gbt_model, file)
{
    # gbt_model - pointer to class ENSEMBLE
    # file - valid path to file
    
    # checks on newdata and e.ptr
    error_messages <- c()
    error_messages_type <- c(
        "Error: gbt_model must be a GBTorch ensemble \n",
        "Error: GBTorch ensemble must be trained, see function documentation gbt.train \n",
        "Error: file must be a valid file path \n"
    )
    # check gbt_model
    if(class(gbt_model)!="Rcpp_ENSEMBLE"){
        error_messages <- c(error_messages, error_messages_type[1])
    }else{
        # test if trained
        if(gbt_model$get_num_trees()==0)
            error_messages <- c(error_messages, error_messages_type[2])
    }
    
    # check file path
    
    # Any error messages?
    if(length(error_messages)>0)
        stop(error_messages)
    
    # Store model in file
    gbt_model$save_model(file)
    
}
