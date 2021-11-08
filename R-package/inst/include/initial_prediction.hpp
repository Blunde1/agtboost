// initial_prediction.hpp

#ifndef __INITIAL_PREDICTION_HPP_INCLUDED__
#define __INITIAL_PREDICTION_HPP_INCLUDED__


#include "external_rcpp.hpp"


double learn_initial_prediction(
        Tvec<double>& y, 
        Tvec<double>& offset,
        std::function<Tvec<double> (Tvec<double>&,Tvec<double>&)> dloss,
        std::function<Tvec<double> (Tvec<double>&,Tvec<double>&)> ddloss,
        std::function<double (double)> link_function,
        std::function<double (double)> inverse_link_function,
        int verbose
    ){
    // Newton opt settings
    double tolerance = 1E-9;
    double step_length = 0.2;
    double step=0.0;
    int niter = 50; // Max iterations
    // Data specific settings
    int n = y.size();
    double y_average = y.sum() / n;
    double initial_prediction = link_function(y_average);
    Tvec<double> pred = offset.array() + initial_prediction;
    // Iterate until optimal starting point found
    for(int i=0; i<niter; i++){
        // Gradient descent
        step = - step_length * dloss(y, pred).sum() / ddloss(y, pred).sum();
        initial_prediction += step;
        pred = pred.array() + step;
        // Check precision
        if(std::abs(step) <= tolerance){
            break;
        }
    }
    // Verbose?
    if(verbose>0){
        Rcpp::Rcout << 
            std::setprecision(4) <<
            "Initial prediction and raw-prediction estimated to :" << 
                inverse_link_function(initial_prediction) <<
                    " and " <<
                initial_prediction << 
                    " respectively" <<
                        std::endl;
    }
    // Retun optimal starting point
    return initial_prediction;
}


#endif
