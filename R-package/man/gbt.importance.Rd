% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gbt.importance.R
\name{gbt.importance}
\alias{gbt.importance}
\title{Importance of features in a model.}
\usage{
gbt.importance(feature_names, object)
}
\arguments{
\item{feature_names}{character vector of feature names}

\item{object}{object or pointer to object of class \code{ENSEMBLE}}
}
\value{
\code{data.frame} with percentwise reduction in loss of total attributed to each feature.
}
\description{
\code{gbt.importance} creates a \code{data.frame} of feature importance in a model
}
\details{
Sums up "expected reduction" in generalization loss (scaled using \code{learning_rate}) 
at each node for each tree in the model, and attributes it to 
the feature the node is split on. Returns result in terms of percents.
}
\examples{
\donttest{
## Load data
data(caravan.train, package = "agtboost")
train <- caravan.train
mod <- gbt.train(train$y, train$x, loss_function = "logloss", verbose=10)
feature_names <- colnames(train$x)
imp <- gbt.importance(feature_names, mod)
imp
}

}
