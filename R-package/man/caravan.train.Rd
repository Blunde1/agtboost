\name{caravan.train}
\alias{caravan.train}
\alias{caravan.test}
\docType{data}
\title{The Insurance Company (TIC) Benchmark
%%   ~~ data name/kind ... ~~
}
\description{
  \code{caravan.train} and \code{caravan.test} both contain a design
  matrix with 85 columns and a response vector. The train set consists
  of 70\% of the data, with 4075 rows. The test set consists of the
  remaining 30\% with 1747 rows. The following references the documentation
  within the \pkg{ISLR} package: 
  The original data contains 5822 real customer records. Each record
  consists of 86 variables, containing sociodemographic data (variables
  1-43) and product ownership (variables 44-86). The sociodemographic
  data is derived from zip codes. All customers living in areas with the
  same zip code have the same sociodemographic attributes. Variable 86
  (\code{Purchase}) indicates whether the customer purchased a caravan
  insurance policy. Further information on the individual variables can
  be obtained at  http://www.liacs.nl/~putten/library/cc2000/data.html
%%  ~~ A concise (1-5 lines) description of the dataset. ~~
}
\usage{caravan.train; caravan.test}
\format{
  Lists with a design matrix \code{x} and response \code{y}
  }
\source{The data was originally supplied by Sentient Machine Research
  and was used in the CoIL Challenge 2000.
}
\references{P. van der Putten and M. van Someren (eds) . CoIL Challenge
  2000: The Insurance Company Case.  Published by Sentient Machine
  Research, Amsterdam. Also a Leiden Institute of Advanced Computer
  Science Technical Report 2000-09. June 22, 2000. See
  http://www.liacs.nl/~putten/library/cc2000/\cr
  P. van der Putten and M. van Someren. A Bias-Variance Analysis of a Real World Learning Problem: The CoIL Challenge 2000. Machine Learning, October 2004, vol. 57, iss. 1-2, pp. 177-195, Kluwer Academic Publishers\cr
  James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013)
  \emph{An Introduction to Statistical Learning with applications in R},
  \url{https://trevorhastie.github.io/ISLR/},
  Springer-Verlag, New York
}
\examples{
summary(caravan.train)
summary(caravan.test)
}
\keyword{datasets}