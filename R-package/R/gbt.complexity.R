transform_prediction <- function(pred, loss_fun_type){
  # This is default: Predict g^{-1}(preds)
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
  return(res)
}

loss_to_xgbloss <- function(loss_function){
  if(loss_function == "mse"){
    xgbloss = "reg:squarederror"
  }else if(loss_function == "logloss"){
    xgbloss = "binary:logistic"
  }else if(loss_function == "gamma::log"){
    xgbloss = "reg:gamma"
  }else if(loss_function == "poisson"){
    xgbloss = "count:poisson"
  }
  return(xgbloss)
}

loss_to_lgbloss <- function(loss_function){
  if(loss_function == "mse"){
    lgbloss = "regression"
  }else if(loss_function == "logloss"){
    lgbloss = "binary"
  }else if(loss_function == "gamma::log"){
    lgbloss = "gamma"
  }else if(loss_function == "poisson"){
    lgbloss = "poisson"
  }
  return(lgbloss)
}

#' Return complexity of model in terms of hyperparameters.
#'
#' \code{gbt.complexity} creates a list of hyperparameters from a model
#'
#' @param model object or pointer to object of class \code{ENSEMBLE}
#' @param type currently supports "xgboost" or "lightgbm"
#'
#' @details
#' 
#' Returns the complexity of \code{model} in terms of hyperparameters associated
#' to model \code{type}.
#'
#' @return
#' \code{list} with \code{type} hyperparameters.
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
#' @rdname gbt.complexity
#' @export
gbt.complexity <- function(model, type){
  
  # # ensemble parameters
  loss_function <- model$get_loss_function()
  nrounds <- model$get_num_trees()
  learning_rate <- model$get_learning_rate()
  initial_raw_prediction <- model$initialPred # needs to be transformed?
  initial_prediction <- transform_prediction(initial_raw_prediction, loss_function)
  # # tree parameters
  max_depth <- max(model$get_tree_depths())
  min_loss_reductions <- model$get_max_node_optimism()
  sum_hessian_weights <- model$get_min_hessian_weights()
  number_of_leaves <- max(model$get_num_leaves())
  # # objective
  l1_regularization <- 0.0
  l2_regularization <- 0.0
  # # sampling
  row_subsampling <- 1.0
  column_subsampling <- 1.0
  
  if(type=="xgboost"){
    parameters = list(
      # ensemble param
      "base_score" = initial_prediction,
      "nrounds" = nrounds,
      "learning_rate" = learning_rate,
      # tree param
      "max_depth" = max_depth,
      "gamma" = min_loss_reductions,
      "min_child_weight" = sum_hessian_weights,
      "max_leaves" = number_of_leaves,
      "grow_policy" = "lossguide",
      # objective
      "objective" = loss_to_xgbloss(loss_function),
      "alpha" = 0.0,
      "lambda" = 0.0,
      # subsampling
      "subsample" = 1.0,
      "colsample_bytree" = 1.0
    )
  }else if(type=="lightgbm"){
    parameters = list(
      # ensemble param
      "init_score" = initial_prediction,
      "nrounds" = nrounds,
      "learning_rate" = learning_rate,
      # tree param
      "max_depth" = max_depth,
      "min_gain_to_split" = min_loss_reductions,
      "min_sum_hessian_in_leaf" = sum_hessian_weights,
      "num_leaves" = number_of_leaves,
      # objective
      "objective" = loss_to_lgbloss(loss_function),
      "lambda_l1" = 0.0,
      "lambda_l2" = 0.0,
      # subsampling
      "bagging_fraction" = 1.0,
      "feature_fraction" = 1.0
    ) 
  }
  return(parameters)
}
