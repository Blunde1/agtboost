// loss_functions

#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__

#include "external_rcpp.hpp"


// Define enum class for loss functions
enum LossFunction 
{
    MSE, // Gaussian distribution with identity-link (mean squared error)
    LOGLOSS, // Bernoulli distribution with logit-link
    POISSON, // Poisson distribution with log-link
    GAMMANEGINV, // Gamma distribution with negative-inverse link
    GAMMALOG, // Gamma dsitribution with log-link
    NEGBINOM // Negative binomial distribution with log-link
};
// Define stream operators for enum class
// https://stackoverflow.com/questions/21691354/enum-serialization-c
std::istream& operator >> (std::istream& in, LossFunction& loss_function)
{
    unsigned u = 0;
    in >> u;
    //TODO: check that u is a valid LossFunction value
    loss_function = static_cast<LossFunction>(u);
    return in;
}

std::ostream& operator << (std::ostream& out, LossFunction loss_function)
{
    //TODO: check that loss_function is a valid LossFunction value
    unsigned u = loss_function;
    out << u;
    return out;
}


// ----------- LOSS --------------
namespace loss_functions {


    double link_function(double pred_observed, LossFunction loss_function){
        // Returns g(mu)
        double pred_transformed=0.0;
        switch(loss_function)
        {
        case MSE:
            pred_transformed = pred_observed;
            break;
        case LOGLOSS:
            pred_transformed = log(pred_observed) - log(1 - pred_observed);
            break;
        case POISSON:
            pred_transformed = log(pred_observed);
            break;
        case GAMMANEGINV:
            pred_transformed = - 1.0 / pred_observed;
            break;
        case GAMMALOG:
            pred_transformed = log(pred_observed);
            break;
        case NEGBINOM:
            pred_transformed = log(pred_observed);
            break;
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
            break;
        case LOGLOSS:
            pred_observed = 1.0 / (1.0+exp(-pred_transformed));
            break;
        case POISSON:
            pred_observed = exp(pred_transformed);
            break;
        case GAMMANEGINV:
            pred_observed = -1.0 / pred_transformed;
            break;
        case GAMMALOG:
            pred_observed = exp(pred_transformed);
            break;
        case NEGBINOM:
            pred_observed = exp(pred_transformed);
            break;
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
            break;
        case LOGLOSS:
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*log(1.0+exp(-pred[i])) + (1.0-y[i]*w[i])*log(1.0 + exp(pred[i]));
            }
            break;
        case POISSON:
            for(int i=0; i<n; i++){
                res += exp(pred[i]) - y[i]*w[i]*pred[i]; // skip normalizing factor log(y!)
            }
            break;
        case GAMMANEGINV:
            // shape=1, only relevant part of negative log-likelihood
            for(int i=0; i<n; i++){
                res += -y[i]*w[i]*pred[i] - log(-pred[i]);
            }
            break;
        case GAMMALOG:
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*exp(-pred[i]) + pred[i];
            }
            break;
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                // log-link, mu=exp(pred[i])
                res += -y[i]*pred[i] + (y[i]*dispersion)*log(1.0+exp(pred[i])/dispersion); // Keep only relevant part
            }
            break;
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
            break;
        case LOGLOSS:
            for(int i=0; i<n; i++){
                g[i] = ( exp(pred[i]) * (1.0-y[i]) - y[i] ) / ( 1.0 + exp(pred[i]) );
            }
            break;
        case POISSON:
            for(int i=0; i<n; i++){
                g[i] = exp(pred[i]) - y[i];
            }
            break;
        case GAMMANEGINV:
            for(int i=0; i<n; i++){
                g[i] = -(y[i]+1.0/pred[i]);
            }
            break;
        case GAMMALOG:
            for(int i=0; i<n; i++){
                g[i] = -y[i]*exp(-pred[i]) + 1.0;
            }
            break;
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                g[i] = -y[i] + (y[i]+dispersion)*exp(pred[i]) / (dispersion + exp(pred[i]));
            }
            break;
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
            break;
        case LOGLOSS:
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0) ) ;
            }
            break;
        case POISSON:
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]);
            }
            break;
        case GAMMANEGINV:
            for(int i=0; i<n; i++){
                h[i] = 1.0/(pred[i]*pred[i]);
            }
            break;
        case GAMMALOG:
            for(int i=0; i<n; i++){
                h[i] = y[i] * exp(-pred[i]);
            }
            break;
        case NEGBINOM:
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                h[i] = (y[i]+dispersion)*dispersion*exp(pred[i]) / 
                    ( (dispersion + exp(pred[i]))*(dispersion + exp(pred[i])) );
            }
            break;
        }
        return h;    
    }
}


#endif
