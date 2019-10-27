// loss_functions

#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__

#include "external_rcpp.hpp"

// ----------- LOSS --------------
double loss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type, Tvec<double> &w){
    int n = y.size();
    double res = 0;
    
    if(loss_type=="mse"){
        // MSE
        for(int i=0; i<n; i++){
            res += pow(y[i]*w[i]-pred[i],2);
        }
        
    }else if(loss_type=="logloss"){
        for(int i=0; i<n; i++){
            res += y[i]*w[i]*log(1.0+exp(-pred[i])) + (1.0-y[i]*w[i])*log(1.0 + exp(pred[i]));
        }
    }
    
    return res/n;
    
}
Tvec<double> dloss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type){
    
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
    }
    
    return g;
}
Tvec<double> ddloss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type="mse"){
    int n = y.size();
    Tvec<double> h(n);
    
    if( loss_type == "mse" ){
        for(int i=0; i<n; i++){
            h[i] = 2.0;
        }
    }else if(loss_type == "logloss"){
        // LOGLOSS
        for(int i=0; i<n; i++){
            h[i] = exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0) ) ;
        }
    }
    
    return h;    
}


#endif