// loss_functions

#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__

#include "external_rcpp.hpp"

// ----------- LOSS --------------
namespace loss_functions {

    enum LossFunction 
    {
        MSE, // Gaussian distribution with identity-link (mean squared error)
        LOGLOSS, // Bernoulli distribution with logit-link
        POISSON, // Poisson distribution with log-link
        GAMMANEGINV, // Gamma distribution with negative-inverse link
        GAMMALOG, // Gamma dsitribution with log-link
        NEGBINOM // Negative binomial distribution with log-link
    };


    double link_function(double pred_observed, LossFunction loss_function){
        // Returns g(mu)
        double pred_transformed=0.0;
        switch(loss_function)
        {
        case MSE:
            pred_transformed = pred_observed;
        case LOGLOSS:
            pred_transformed = log(pred_observed) - log(1 - pred_observed);
        case POISSON:
            pred_transformed = log(pred_observed);
        case GAMMANEGINV:
            pred_transformed = - 1.0 / pred_observed;
        case GAMMALOG:
            pred_transformed = log(pred_observed);
        case NEGBINOM:
            pred_transformed = log(pred_observed);
        }
        return pred_transformed;
    }


    double inverse_link_function(double pred_transformed, LossFunction loss_function){
        // Returns g^{-1}(pred)
        double pred_observed = 0.0;
        switch(loss_function)
        {
        case MSE:
            pred_observed = pred_transformed;
        case LOGLOSS:
            pred_observed = 1.0 / (1.0+exp(-pred_transformed));
        case POISSON:
            pred_observed = exp(pred_transformed);
        case GAMMANEGINV:
            pred_observed = -1.0 / pred_transformed;
        case GAMMALOG:
            pred_observed = exp(pred_transformed);
        case NEGBINOM:
            pred_observed = exp(pred_transformed);
        }
        return pred_observed;
    }


    double loss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            LossFunction loss_function, 
            Tvec<double> &w, 
            double extra_param=0.0
    ){
        // Evaluates the loss function at pred
        int n = y.size();
        double res = 0;
        switch(loss_function)
        {
        case MSE:
            for(int i=0; i<n; i++){
                res += pow(y[i]*w[i]-pred[i],2);
            }
        case LOGLOSS:
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*log(1.0+exp(-pred[i])) + (1.0-y[i]*w[i])*log(1.0 + exp(pred[i]));
            }
        case POISSON:
            for(int i=0; i<n; i++){
                res += exp(pred[i]) - y[i]*w[i]*pred[i]; // skip normalizing factor log(y!)
            }
        case GAMMANEGINV:
            // shape=1, only relevant part of negative log-likelihood
            for(int i=0; i<n; i++){
                res += -y[i]*w[i]*pred[i] - log(-pred[i]);
            }
        case GAMMALOG:
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*exp(-pred[i]) + pred[i];
            }
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                // log-link, mu=exp(pred[i])
                res += -y[i]*pred[i] + (y[i]*dispersion)*log(1.0+exp(pred[i])/dispersion); // Keep only relevant part
            }
        }
        return res/n;
        
    }
    
    
    Tvec<double> dloss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            LossFunction loss_function,
            double extra_param=0.0
    ){
        // Returns the first order derivative of the loss function at pred
        int n = y.size();
        Tvec<double> g(n);
        switch(loss_function)
        {
        case MSE:
            for(int i=0; i<n; i++){
                g[i] = -2*(y[i]-pred[i]);
            }
        case LOGLOSS:
            for(int i=0; i<n; i++){
                g[i] = ( exp(pred[i]) * (1.0-y[i]) - y[i] ) / ( 1.0 + exp(pred[i]) );
            }
        case POISSON:
            for(int i=0; i<n; i++){
                g[i] = exp(pred[i]) - y[i];
            }
        case GAMMANEGINV:
            for(int i=0; i<n; i++){
                g[i] = -(y[i]+1.0/pred[i]);
            }
        case GAMMALOG:
            for(int i=0; i<n; i++){
                g[i] = -y[i]*exp(-pred[i]) + 1.0;
            }
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                g[i] = -y[i] + (y[i]+dispersion)*exp(pred[i]) / (dispersion + exp(pred[i]));
            }
        }
        return g;
    }
    
    
    Tvec<double> ddloss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            LossFunction loss_function,
            double extra_param=0.0
    ){
        // Returns the second order derivative of the loss function at pred
        int n = y.size();
        Tvec<double> h(n);
        switch(loss_function)
        {
        case MSE:
            for(int i=0; i<n; i++){
                h[i] = 2.0;
            }
        case LOGLOSS:
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0) ) ;
            }
        case POISSON:
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]);
            }
        case GAMMANEGINV:
            for(int i=0; i<n; i++){
                h[i] = 1.0/(pred[i]*pred[i]);
            }
        case GAMMALOG:
            for(int i=0; i<n; i++){
                h[i] = y[i] * exp(-pred[i]);
            }
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                h[i] = (y[i]+dispersion)*dispersion*exp(pred[i]) / 
                    ( (dispersion + exp(pred[i]))*(dispersion + exp(pred[i])) );
            }
        }
        return h;    
    }
}


#endif
