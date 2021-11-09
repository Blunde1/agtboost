// gbt_zimix.hpp
// gbtorch count auto

#ifndef __GBT_COUNT_AUTO_HPP_INCLUDED__
#define __GBT_COUNT_AUTO_HPP_INCLUDED__


#include "ensemble.hpp"


class GBT_COUNT_AUTO
{
public:
    
    ENSEMBLE* count_mod; // Poisson or negative binomial so far
    
    double learning_rate;
    double extra_param; // Needed for certain distributions s.a. negative binomial, typically a dispersion param
    Rcpp::List param;
    
    // constructors
    GBT_COUNT_AUTO();
    
    // Functions
    void set_param(Rcpp::List par_list);
    Rcpp::List get_param();
    std::string get_model_name();
    
    double get_overdispersion(); // Retrieve extra param (overdispersion)
    
    ENSEMBLE* get_count_mod();
    
    void train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities);
    
    Tvec<double> predict(Tmat<double> &X);
};



#endif
