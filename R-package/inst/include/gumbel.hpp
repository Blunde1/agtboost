// gumbel.hpp

#ifndef __GUMBEL_HPP_INCLUDED__
#define __GUMBEL_HPP_INCLUDED__

// Distribution function templated
template<class T>
T pgumbel(double q, T location, T scale, bool lower_tail, bool log_p){
    
    T z = (q-location)/scale;
    T log_px = -exp(-z); // log p(X <= x)
    T res;
    
    if(lower_tail && log_p){
        res = log_px;
    }else if(lower_tail && !log_p){
        res = exp(log_px);
    }else if(!lower_tail && log_p){
        res = log(1.0 - exp(log_px));
    }else{
        res = 1.0 - exp(log_px);
    }
    
    if( std::isnan(res) ){
        return 1.0;
    }else{
        return res;
    }
    
}

// Gradient of estimating equation for scale
double grad_scale_est_obj(double scale, Tavec<double> &x){
    
    int n = x.size();
    Tavec<double> exp_x_beta = (-1.0*x/scale).exp();
    //exp_x_beta = exp_x_beta.array().exp();
    double f = scale + (x*exp_x_beta).sum()/exp_x_beta.sum() - x.sum()/n;
    double grad = 2.0*f* ( 1.0 + 
                         ( (x*x*exp_x_beta).sum() * exp_x_beta.sum() - 
                         pow((x*exp_x_beta).sum(),2.0) ) / 
                         pow(scale*exp_x_beta.sum(), 2.0));
    return grad;
    
}

// ML Estimate of scale
double scale_estimate(Tavec<double> &x){
    
    // Start in variance estimate -- already pretty good
    int n = x.size();
    int mean = x.sum()/n;
    double var = 0.0;
    for(int i=0; i<n; i++){
        var += (x[i]-mean)*(x[i]-mean)/n;
    }
    double scale_est = sqrt(var*6.0)/M_PI;
    
    // do some gradient iterations to obtain ML estimate
    int NITER = 50; // max iterations
    double EPS = 1e-2; // precision
    double step_length = 0.2; //conservative
    double step;
    for(int i=0; i<NITER; i++){
        
        // gradient descent
        step = - step_length * grad_scale_est_obj(scale_est, x);
        scale_est += step;
        
        //Rcpp::Rcout << "iter " << i << ", step: " << std::abs(step) << ", estimate: " <<  scale_est << std::endl;
        
        // check precision
        if(std::abs(step) <= EPS){
            break;
        }
        
    }
    
    return scale_est;
    
}

// ML Estimates
Tvec<double> par_gumbel_estimates(Tavec<double> &x){
    
    int n = x.size();
    
    double scale_est = scale_estimate(x);
    double location_est = scale_est * ( log((double)n) - log( (-1.0*x/scale_est).exp().sum() ) );
    
    Tvec<double> res(2);
    res << location_est, scale_est;
    
    return res;
    
}

#endif
