#ifndef __EXTERNALRCPP_HPP_INCLUDED__
#define __EXTERNALRCPP_HPP_INCLUDED__


#include <RcppEigen.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// Enables Eigen
// [[Rcpp::depends(RcppEigen)]]


using namespace Eigen;
using namespace Rcpp;

template <class T>
using Tvec = Eigen::Matrix<T,Dynamic,1>;

template <class T>
using Tmat = Eigen::Matrix<T,Dynamic,Dynamic>;

template<class T>
using Tavec = Eigen::Array<T,Eigen::Dynamic,1>; 


#endif // __EXTERNALRCPP_HPP_INCLUDED__
