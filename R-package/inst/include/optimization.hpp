// Optimization
// Berent Lunde
// 19.04.2020


#ifndef __OPTIMIZATION_HPP_INCLUDED__
#define __OPTIMIZATION_HPP_INCLUDED__


#include <Rcpp.h>


// Use optimization to find solution lambda = lambertW1(-a*exp(-a))+a, where a=mean(y)
double poisson_zip_start(double a){
    // a is mean
    // Optimize on log-level
    
    double pred = log(a);
    int NITER = 100;
    double TRESH = 1e-9;
    double step, g, h;
    for(int i=0; i<NITER; i++){
        
        // Grad and Hess of likelihood
        g = exp(pred) - a + exp(pred)/(exp(exp(pred))-1.0);
        h = exp(pred) + exp(pred)*(exp(exp(pred))-exp(pred+exp(pred))-1.0) / 
            ( (exp(exp(pred))-1.0)*(exp(exp(pred))-1.0) );
        
        step = -g/h;
        pred += step;
        Rcpp::Rcout << "lambda est: " << exp(pred) << " - step: " << step << std::endl;
        
        if(std::abs(step)<TRESH){
            break;
        }
        
    }
    return pred;
}

#endif