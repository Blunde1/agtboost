// Optimization
// Berent Lunde
// 19.04.2020


#ifndef __OPTIMIZATION_HPP_INCLUDED__
#define __OPTIMIZATION_HPP_INCLUDED__

#include "external_rcpp.hpp"
#include "ensemble.hpp"

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
        Rcpp::Rcout << "lambda initial: " << exp(pred) << " - step: " << step << std::endl;
        
        if(std::abs(step)<TRESH){
            break;
        }
        
    }
    return pred;
}

double zero_inflation_start(Tvec<double>& y, ENSEMBLE* ens_ptr)
{
    // retrieve lambda_est
    // Calculate pi = (sum(lambda_est) - mean(y))/sum(lambda_est)
    int n = y.size();
    double lambda_sum=0, y_sum=0;
    Tvec<double> preds_cond_count = ens_ptr->param["preds_cond_count"];
    //Rcpp::Rcout << "y: \n" << y << "\n lambda: \n" <<  preds_cond_count << std::endl;
    for(int i=0; i<n; i++){
        y_sum += y[i];
        lambda_sum += preds_cond_count[i];
    }
    //Rcpp::Rcout <<"lambda_sum" << lambda_sum << " - y_sum: " << y_sum << std::endl;
    double prob = (lambda_sum - y_sum)/lambda_sum;
    Rcpp::Rcout << "Initial probability: " << prob << std::endl;
    return log(prob) - log(1.0-prob); // logit transform
}

#endif