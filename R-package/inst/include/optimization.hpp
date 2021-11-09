// Optimization
// Berent Lunde
// 19.04.2020


#ifndef __OPTIMIZATION_HPP_INCLUDED__
#define __OPTIMIZATION_HPP_INCLUDED__

#include "external_rcpp.hpp"
#include "ensemble.hpp"

// // Use optimization to find solution lambda = lambertW1(-a*exp(-a))+a, where a=mean(y)
// double poisson_zip_start(double a){
//     // a is mean
//     // Optimize on log-level
//     
//     double pred = log(a);
//     int NITER = 100;
//     double TRESH = 1e-9;
//     double step, g, h;
//     int i=0;
//     for(i=0; i<NITER; i++){
//         
//         // Grad and Hess of likelihood
//         g = exp(pred) - a + exp(pred)/(exp(exp(pred))-1.0);
//         h = exp(pred) + exp(pred)*(exp(exp(pred))-exp(pred+exp(pred))-1.0) / 
//             ( (exp(exp(pred))-1.0)*(exp(exp(pred))-1.0) );
//         
//         step = -g/h;
//         pred += step;
//         //Rcpp::Rcout << "lambda initial: " << exp(pred) << " - step: " << step << std::endl;
//         
//         if(std::abs(step)<TRESH){
//             break;
//         }
//         
//     }
//     Rcpp::Rcout << "Estimated Poisson intensity: " << exp(pred) << " - After " << i << " iterations" << std::endl;
//     return pred;
// }
// 
// // Use optimization to find starting point of negbinom::zinb (minimize likelihood w.r.t. mu:= exp(pred))
// double negbinom_zinb_start(double a, double dispersion){
//     
//     // a is mean
//     // optimize on log-level
//     double pred = log(a);
//     int NITER = 100;
//     double TRESH = 1e-9;
//     double step, g, h;
//     int i=0;
//     for(i=0; i<NITER; i++){
//         
//         // Grad and Hess of likelihood
//         g = -a + (a+dispersion)*exp(pred) / (dispersion + exp(pred)) + 
//             dispersion*exp(pred) / 
//             ( (dispersion+exp(pred))*( exp(dispersion*(log(dispersion+exp(pred))-log(dispersion))) -1.0 ));
//         h = (a+dispersion)*dispersion*exp(pred) / 
//             ( (dispersion + exp(pred))*(dispersion + exp(pred)) ) - 
//             // d^2/dx^2 log(p(y>0))
//             -dispersion*dispersion*exp(pred)*
//             ((exp(pred)-1.0)*exp(dispersion*(log(dispersion+exp(pred))-log(dispersion))) +1.0 ) / 
//             (exp(2.0*log(dispersion+exp(pred))) * 
//             pow(exp(dispersion*(log(dispersion+exp(pred))-log(dispersion))) - 1.0, 2.0 )  );
//         
//         step = -g/h;
//         pred += step;
//         //Rcpp::Rcout << "lambda initial: " << exp(pred) << " - step: " << step << std::endl;
//         
//         if(std::abs(step)<TRESH){
//             break;
//         }
//         
//     }
//     Rcpp::Rcout << "Estimated negbinom mean: " << exp(pred) << " - After " << i << " iterations" << std::endl;
//     
//     return pred;
// }

/*
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
    
    // Error handling
    if(prob <= 0){ // needs to be between 0 and 1, likely due to no zero-inflation
        throw std::range_error("Predicted Poisson intensity smaller than observations: Try ordinary Poisson regression");
    }
    
    return log(prob) - log(1.0-prob); // logit transform
}
*/

// nll negbinom
double nll_negbinom(Tvec<double>& y, Tvec<double>& lambda, double alpha)
{
    // lambda := log(mu)
    // alpha := log(dispersion) = log(size)
    double dispersion = exp(alpha);
    double nll = 0.0;
    int n = y.size();
    for(int i=0; i<n; i++)
    {
        nll += y[i]*log(dispersion) - y[i]*lambda[i] + 
            (y[i]+dispersion)*log(1.0+exp(lambda[i])/dispersion) - 
            R::lgammafn(y[i]+dispersion) + R::lgammafn(y[i]+1.0) + R::lgammafn(dispersion);
    }
    return nll/n;
}

// grad w.r.t. alpha := log(dispersion)
double gdnbinom(Tvec<double>& y, Tvec<double>& lambda, double alpha)
{
    // lambda := log(mu)
    // alpha := log(dispersion) = log(size)
    double g = 0.0;
    int n = y.size();
    for(int i=0; i<n; i++){
        g += y[i]-exp(lambda[i]-alpha)*(exp(alpha)+y[i])/(exp(lambda[i]-alpha)+1.0) + 
            exp(alpha)*log(exp(lambda[i]-alpha)+1.0) - 
            exp(alpha)*R::digamma(y[i]+exp(alpha)) + exp(alpha)*R::digamma(exp(alpha));
    }
    return g/n;
}

// // grad w.r.t. alpha := log(dispersion)
// double gdnbinom_zi(Tvec<double>& y, Tvec<double>& lambda, double alpha)
// {
//     // lambda := log(mu)
//     // alpha := log(dispersion) = log(size)
//     double g = 0.0;
//     int n = y.size();
//     for(int i=0; i<n; i++){
//         g += y[i]-exp(lambda[i]-alpha)*(exp(alpha)+y[i])/(exp(lambda[i]-alpha)+1) + 
//             exp(alpha)*log(exp(lambda[i]-alpha)+1) - 
//             exp(alpha)*R::digamma(y[i]+exp(alpha)) + exp(alpha)*R::digamma(exp(alpha)) -
//             (exp(lambda[i])/(exp(lambda[i]-alpha)+1.0)-exp(alpha)*log(exp(lambda[i]-alpha) + 1.0)) / 
//             (exp(exp(alpha)*log(exp(lambda[i]-alpha)+1.0)) - 1.0);
//     }
//     return g/n;
// }

// hess w.r.t. alpha := log(dispersion)
double hdnbinom(Tvec<double>& y, Tvec<double>& lambda, double alpha){
    
    // lambda := log(mu)
    // alpha := log(dispersion) = log(size)
    double h = 0.0;
    int n = y.size();
    for(int i=0; i<n; i++){
        h += (exp(lambda[i]-alpha)/(exp(lambda[i]-alpha)+1) - exp(2*(lambda[i]-alpha))/pow(exp(lambda[i]-alpha)+1,2))*(exp(alpha)+y[i]) - 
            2*exp(lambda[i])/(exp(lambda[i]-alpha)+1) + exp(alpha)*log(exp(lambda[i]-alpha)+1) - 
            exp(alpha)*R::digamma(y[i]+exp(alpha)) - exp(2*alpha)*R::trigamma(y[i]+exp(alpha)) + 
            exp(alpha)*R::digamma(exp(alpha)) + exp(2*alpha)*R::trigamma(exp(alpha));
    }
    
    return h/n;
}

// // hess w.r.t. alpha := log(dispersion)
// double hdnbinom_zi(Tvec<double>& y, Tvec<double>& lambda, double alpha){
//     
//     // lambda := log(mu)
//     // alpha := log(dispersion) = log(size)
//     double h = 0.0;
//     int n = y.size();
//     for(int i=0; i<n; i++){
//         h += (exp(lambda[i]-alpha)/(exp(lambda[i]-alpha)+1) - exp(2*(lambda[i]-alpha))/pow(exp(lambda[i]-alpha)+1,2))*(exp(alpha)+y[i]) - 
//             2*exp(lambda[i])/(exp(lambda[i]-alpha)+1) + exp(alpha)*log(exp(lambda[i]-alpha)+1) - 
//             exp(alpha)*R::digamma(y[i]+exp(alpha)) - exp(2*alpha)*R::trigamma(y[i]+exp(alpha)) + 
//             exp(alpha)*R::digamma(exp(alpha)) + exp(2*alpha)*R::trigamma(exp(alpha)) -
//         // d^2/dx^2 log(p(y>0)) where x is log(dispersion)
//         // First fraction
//         exp(-2.0*exp(alpha)*log(exp(lambda[i]-alpha)+1.0))*(exp(lambda[i])/(exp(lambda[i]-alpha)+1.0)-exp(alpha)*log(exp(lambda[i]-alpha)+1.0)) * 
//         (exp(lambda[i])/(exp(lambda[i]-alpha)+1.0)-exp(alpha)*log(exp(lambda[i]-alpha)+1.0)) / 
//         pow(1.0-exp(-exp(lambda[i])*log(exp(lambda[i]-alpha)+1.0)), 2.0) - 
//         // Second fraction
//         exp(-exp(alpha)*log(exp(lambda[i]-alpha)+1.0)) * 
//         (exp(lambda[i])/(exp(lambda[i]-alpha)+1.0) + exp(2.0*lambda[i]-alpha)/pow(exp(lambda[i]-alpha)+1.0,2.0)  -exp(alpha)*log(exp(lambda[i]-alpha)+1.0)) / 
//         (1.0-exp(-exp(lambda[i])*log(exp(lambda[i]-alpha)+1.0)));
//         // Third fraction
//         
//     }
//         
//     return h/n;
// }

double learn_dispersion(Tvec<double>& y, Tvec<double>& lambda, double disp_init=0.5)
{
    // lambda := log(mu)
    // alpha := log(dispersion) = log(size)
    
    // 1.0 Profile search on log(dispersion) in (log(0.01), log(1000))
    Tvec<double> ldisp_check = Tvec<double>::LinSpaced(200, -2.0, 10.0);
    //Tvec<double> disp = exp(ldisp_check.array()).matrix();
    int nd=ldisp_check.size();
    double best_ldisp=1.0, best_nll, nll_check;
    for(int i=0; i<nd; i++)
    {
        // Check all values og log-likelihood for values of disp
        nll_check = nll_negbinom(y, lambda, ldisp_check[i]);
        if(i==0){
            // first
            best_nll = nll_check;
            best_ldisp = ldisp_check[i];
        }else{
            // Check if better result
            if(nll_check < best_nll){
                // Update
                best_nll = nll_check;
                best_ldisp = ldisp_check[i];
            }
        }
    }
    Rcpp::Rcout << "Estimated dispersion after profile " << exp(best_ldisp) << std::endl;
    
    
    double ldisp = best_ldisp; //log(disp_init); // Start value
    int NITER = 100;
    double TRESH = 1e-9;
    double MAXTRESH = 1e9;
    int i=0;
    double step, g, h;
    for(i=0; i<NITER; i++){
        
        // Grad and Hess of likelihood
        g = gdnbinom(y, lambda, ldisp);
        h = hdnbinom(y, lambda, ldisp); // Hopefully okay...
        
        step = -g/h;
        ldisp += step;
        //Rcpp::Rcout << "dispersion initial: " << exp(ldisp) << " - step: " << step << std::endl;
        
        if(std::abs(step)<TRESH){
            // Convergence
            break;
        }
        if(std::isnan(exp(ldisp)) || exp(ldisp)>MAXTRESH){
            // Divergence -- When Poisson is correct
            ldisp = log(MAXTRESH);
            break;
        }
        
    }
    Rcpp::Rcout << "Estimated dispersion: " << exp(ldisp) << " - After " << i << " iterations" << std::endl;
    return exp(ldisp);
}

// double learn_dispersion_zi(Tvec<double>& y, Tvec<double>& lambda)
// {
//     // lambda := log(mu)
//     // alpha := log(dispersion) = log(size)
//     double ldisp = log(1.0);
//     int NITER = 100;
//     double TRESH = 1e-9;
//     double MAXTRESH = 1e7;
//     int i=0;
//     double step, g, h;
//     for(i=0; i<NITER; i++){
//         
//         // Grad and Hess of likelihood
//         g = gdnbinom_zi(y, lambda, ldisp);
//         h = hdnbinom_zi(y, lambda, ldisp); // Hopefully okay...
//         
//         step = -g/h;
//         ldisp += step;
//         //Rcpp::Rcout << "dispersion initial: " << exp(ldisp) << " - step: " << step << std::endl;
//         
//         if(std::abs(step)<TRESH){
//             // Convergence
//             break;
//         }
//         if(std::isnan(exp(ldisp)) || exp(ldisp)>MAXTRESH){
//             // Divergence -- When Poisson is correct
//             ldisp = log(MAXTRESH);
//             break;
//         }
// 
//     }
//     Rcpp::Rcout << "Estimated dispersion: " << exp(ldisp) << " - After " << i << " iterations" << std::endl;
//     return exp(ldisp);
// }

#endif
