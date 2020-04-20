// gbt_zimix.hpp
// gbtorch zero-inflated mixture model

#ifndef __GBT_ZI_MIX_HPP_INCLUDED__
#define __GBT_ZI_MIX_HPP_INCLUDED__


#include "ensemble.hpp"


class GBT_ZI_MIX
{
public:
    
    ENSEMBLE* count_conditional; // Count model conditioned on y>0 -- Poisson or negative binomial so far
    ENSEMBLE* zero_inflation;
    
    double learning_rate;
    double extra_param; // Needed for certain distributions s.a. negative binomial, typically a dispersion param
    Rcpp::List param;
    
    // constructors
    GBT_ZI_MIX();

    // Functions
    void set_param(Rcpp::List par_list);
    Rcpp::List get_param();
    
    double get_overdispersion(); // Retrieve extra param (overdispersion)
    
    ENSEMBLE* get_count_conditional();
    ENSEMBLE* get_zero_inflation();
    
    void train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities);
    
    Tvec<double> predict(Tmat<double> &X);
};



#endif