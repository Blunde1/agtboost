// loss_functions

#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__

#include "external_rcpp.hpp"

// ----------- LOSS --------------
namespace loss_functions {


    double link_function(double pred_observed, std::string loss_function){
        // Returns g(mu)
        double pred_transformed=0.0;
        if(loss_function=="mse"){
            pred_transformed = pred_observed;
        }else if(loss_function=="logloss"){
            pred_transformed = log(pred_observed) - log(1 - pred_observed);
        }else if(loss_function=="poisson"){
            pred_transformed = log(pred_observed);
        }else if(loss_function=="gamma::neginv"){
            pred_transformed = - 1.0 / pred_observed;
        }else if(loss_function=="gamma::log"){
            pred_transformed = log(pred_observed);
        }else if(loss_function=="negbinom"){
            pred_transformed = log(pred_observed);
        }
        return pred_transformed;
    }


    double inverse_link_function(double pred_transformed, std::string loss_function){
        // Returns g^{-1}(pred)
        double pred_observed = 0.0;
        if(loss_function=="mse"){
            pred_observed = pred_transformed;
        }else if(loss_function=="logloss"){
            pred_observed = 1.0 / (1.0+exp(-pred_transformed));
        }else if(loss_function=="poisson"){
            pred_observed = exp(pred_transformed);
        }else if(loss_function=="gamma::neginv"){
            pred_observed = -1.0 / pred_transformed;;
        }else if(loss_function=="gamma::log"){
            pred_observed = exp(pred_transformed);
        }else if(loss_function=="negbinom"){
            pred_observed = exp(pred_transformed);
        }
        return pred_observed;
    }


    double loss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            std::string loss_type, 
            Tvec<double> &w, 
            double extra_param=0.0
    ){
        // Evaluates the loss function at pred
        int n = y.size();
        double res = 0;
        
        if(loss_type=="mse"){
            // MSE
            for(int i=0; i<n; i++){
                res += pow(y[i]*w[i]-pred[i],2);
            }
            
        }else if(loss_type=="logloss"){
            // LOGLOSS
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*log(1.0+exp(-pred[i])) + (1.0-y[i]*w[i])*log(1.0 + exp(pred[i]));
            }
        }else if(loss_type=="poisson"){
            // POISSON
            for(int i=0; i<n; i++){
                res += exp(pred[i]) - y[i]*w[i]*pred[i]; // skip normalizing factor log(y!)
            }
        }else if(loss_type=="gamma::neginv"){
            // GAMMA::NEGINV
            // shape=1, only relevant part of negative log-likelihood
            for(int i=0; i<n; i++){
                res += -y[i]*w[i]*pred[i] - log(-pred[i]);
            }
        }else if(loss_type=="gamma::log"){
            // GAMMA::LOG
            for(int i=0; i<n; i++){
                res += y[i]*w[i]*exp(-pred[i]) + pred[i];
            }
        }else if(loss_type=="negbinom"){
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                // log-link, mu=exp(pred[i])
                res += -y[i]*pred[i] + (y[i]*dispersion)*log(1.0+exp(pred[i])/dispersion); // Keep only relevant part
            }
        }
        // else if(loss_type=="poisson::zip"){
        //     // POISSON COND Y>0, LOG LINK
        //     for(int i=0; i<n; i++){
        //         res += exp(pred[i]) - y[i]*pred[i] + log(1.0-exp(-exp(pred[i]))); // Last is conditional p(y>0)
        //     }
        // }else if(loss_type=="zero_inflation"){
        //     // ZERO-INFLATION PROBABILITY MIX
        //     Tvec<double> lprob_weights = ens_ptr->param["log_prob_weights"];
        //     for(int i=0; i<n; i++){
        //         if(y[i] > 0){
        //             // avoid comparing equality to zero...
        //             res += pred[i] + log(1.0+exp(-pred[i])) - lprob_weights[i]; // Weight is log probability weight!!
        //         }else{
        //             // get y[i] == 0
        //             res += -log(1.0/(1.0+exp(-pred[i])) + (1.0 - 1.0/(1.0+exp(-pred[i])))*exp(lprob_weights[i]) );
        //         }
        //     }
        // }else if(loss_type=="negbinom::zinb"){
        //     // NEGBINOM COND Y>0, LOG LINK
        //     double dispersion = ens_ptr -> extra_param;
        //     for(int i=0; i<n; i++){
        //         res += -y[i]*pred[i] + (y[i]*dispersion)*log(1.0+exp(pred[i])/dispersion) + 
        //             log(1.0-(exp(-dispersion*log(1.0+exp(pred[i])/dispersion)))); // Last is conditional p(y>0)
        //     }
        // }
        
        return res/n;
        
    }
    
    
    Tvec<double> dloss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            std::string loss_type, 
            double extra_param=0.0
    ){
        // Returns the first order derivative of the loss function at pred
        int n = y.size();
        Tvec<double> g(n);
        
        if(loss_type == "mse"){
            // MSE
            for(int i=0; i<n; i++){
                g[i] = -2*(y[i]-pred[i]);
            }
        }else if(loss_type == "logloss"){
            // LOGLOSS
            for(int i=0; i<n; i++){
                g[i] = ( exp(pred[i]) * (1.0-y[i]) - y[i] ) / ( 1.0 + exp(pred[i]) );
            }
        }else if(loss_type == "poisson"){
            // POISSON REG
            for(int i=0; i<n; i++){
                g[i] = exp(pred[i]) - y[i];
            }
        }else if(loss_type == "gamma::neginv"){
            // GAMMA::NEGINV
            for(int i=0; i<n; i++){
                g[i] = -(y[i]+1.0/pred[i]);
            }
        }else if(loss_type == "gamma::log"){
            // GAMMA::LOG
            for(int i=0; i<n; i++){
                g[i] = -y[i]*exp(-pred[i]) + 1.0;
            }
        }else if(loss_type == "negbinom"){
            // NEGATIVE BINOMIAL, LOG LINK
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                g[i] = -y[i] + (y[i]+dispersion)*exp(pred[i]) / (dispersion + exp(pred[i]));
            }
        }
        // else if(loss_type == "poisson::zip"){
        //     // POISSON COND Y>0, LOG LINK
        //     for(int i=0; i<n; i++){
        //         g[i] = exp(pred[i]) - y[i] + exp(pred[i])/(exp(exp(pred[i]))-1.0);
        //     }
        // }else if(loss_type=="zero_inflation"){
        //     // ZERO-INFLATION PROBABILITY MIX
        //     Tvec<double> lprob_weights = ens_ptr->param["log_prob_weights"];
        //     for(int i=0; i<n; i++){
        //         if(y[i] > 0){
        //             // avoid comparing equality to zero...
        //             g[i] = exp(pred[i]) / (exp(pred[i]) + 1.0);
        //         }else{
        //             // get y[i] == 0
        //             g[i] = (exp(lprob_weights[i])-1.0)*exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(lprob_weights[i])+exp(pred[i])) );
        //         }
        //     }
        // }else if(loss_type=="negbinom::zinb"){
        //     // NEGBINOM COND Y>0, LOG LINK
        //     double dispersion = ens_ptr -> extra_param;
        //     for(int i=0; i<n; i++){
        //         g[i] = -y[i] + (y[i]+dispersion)*exp(pred[i]) / (dispersion + exp(pred[i])) + 
        //             dispersion*exp(pred[i]) / 
        //             ( (dispersion+exp(pred[i]))*( exp(dispersion*(log(dispersion+exp(pred[i]))-log(dispersion))) -1.0 ));
        //     }
        // }
        
        return g;
    }
    
    
    Tvec<double> ddloss(
            Tvec<double> &y, 
            Tvec<double> &pred, 
            std::string loss_type, 
            double extra_param=0.0
    ){
        // Returns the second order derivative of the loss function at pred
        int n = y.size();
        Tvec<double> h(n);
        
        if( loss_type == "mse" ){
            // MSE
            for(int i=0; i<n; i++){
                h[i] = 2.0;
            }
        }else if(loss_type == "logloss"){
            // LOGLOSS
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0) ) ;
            }
        }else if(loss_type == "poisson"){
            // POISSON REG
            for(int i=0; i<n; i++){
                h[i] = exp(pred[i]);
            }
        }else if(loss_type == "gamma::neginv"){
            // GAMMA::NEGINV
            for(int i=0; i<n; i++){
                h[i] = 1.0/(pred[i]*pred[i]);
            }
        }else if(loss_type == "gamma::log"){
            // GAMMA::LOG
            for(int i=0; i<n; i++){
                h[i] = y[i] * exp(-pred[i]);
            }
        }else if( loss_type == "negbinom" ){
            // NEGATIVE BINOMIAL, LOG LINK
            double dispersion = extra_param;
            for(int i=0; i<n; i++){
                h[i] = (y[i]+dispersion)*dispersion*exp(pred[i]) / 
                    ( (dispersion + exp(pred[i]))*(dispersion + exp(pred[i])) );
            }
        }
        // else if(loss_type == "poisson::zip"){
        //     // POISSON COND Y>0, LOG LINK
        //     for(int i=0; i<n; i++){
        //         h[i] = exp(pred[i]) + 
        //             exp(pred[i])*(exp(exp(pred[i]))-exp(pred[i]+exp(pred[i]))-1.0) / 
        //             ( (exp(exp(pred[i]))-1.0)*(exp(exp(pred[i]))-1.0) );
        //     }
        // }else if(loss_type=="zero_inflation"){
        //     // ZERO-INFLATION PROBABILITY MIX
        //     Tvec<double> lprob_weights = ens_ptr->param["log_prob_weights"];
        //     for(int i=0; i<n; i++){
        //         if(y[i] > 0){
        //             // avoid comparing equality to zero...
        //             h[i] = exp(pred[i]) / ((exp(pred[i]) + 1.0)*(exp(pred[i]) + 1.0));
        //         }else{
        //             // get y[i] == 0
        //             h[i] = -(exp(lprob_weights[i])-1.0)*exp(pred[i])*(exp(2.0*pred[i])-exp(lprob_weights[i])) / 
        //                 ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0)*(exp(lprob_weights[i])+exp(pred[i]))*(exp(lprob_weights[i])+exp(pred[i])) );
        //         }
        //     }
        // }else if(loss_type=="negbinom::zinb"){
        //     // NEGBINOM COND Y>0, LOG LINK
        //     double dispersion = ens_ptr -> extra_param;
        //     for(int i=0; i<n; i++){
        //         h[i] = (y[i]+dispersion)*dispersion*exp(pred[i]) / 
        //             ( (dispersion + exp(pred[i]))*(dispersion + exp(pred[i])) ) - 
        //             // d^2/dx^2 log(p(y>0))
        //             -dispersion*dispersion*exp(pred[i])*
        //             ((exp(pred[i])-1.0)*exp(dispersion*(log(dispersion+exp(pred[i]))-log(dispersion))) +1.0 ) / 
        //             (exp(2.0*log(dispersion+exp(pred[i]))) * 
        //              pow(exp(dispersion*(log(dispersion+exp(pred[i]))-log(dispersion))) - 1.0, 2.0 )  );
        //     }
        // }
        
        return h;    
    }
}


#endif
