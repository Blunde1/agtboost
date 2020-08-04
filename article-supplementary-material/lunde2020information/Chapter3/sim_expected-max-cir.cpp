#include <RcppEigen.h>

// Enable C++11 via this plugin (Rcpp 0.10.3 or later)
// [[Rcpp::plugins("cpp11")]]

// Enables Eigen
// [[Rcpp::depends(RcppEigen)]]


using namespace Eigen;
using namespace Rcpp;

template <class T>
using Tvec = Eigen::Matrix<T,Dynamic,1>;

template <class T>
using Tmat = Eigen::Matrix<T,Dynamic,Dynamic>;

// [[Rcpp::export]]
Tvec<double> cir_sim_vec(int m, double EPS)
{
    //double EPS = 1e-7;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( m+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(m, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    
    // Find cir delta
    Tvec<double> tau_delta = tau.tail(m-1) - tau.head(m-1);
    
    // Simulate first observation
    Tvec<double> res(m);
    res[0] = R::rgamma( 0.5, 2.0 );
    
    // Simulate remaining observatins
    
    double a = 2.0;
    double b = 1.0;
    double sigma = 2.0*sqrt(2.0);
    double c = 0;
    double ncchisq;
    
    for(int i=1; i<m; i++){
        
        c = 2.0 * a / ( sigma*sigma * (1.0 - exp(-a*tau_delta[i-1])) );
        ncchisq =  R::rnchisq( 4.0*a*b/(sigma*sigma), 2.0*c*res[i-1]*exp(-a*tau_delta[i-1]) );
        res[i] = ncchisq/(2.0*c);
        
    }
    
    return res;
    
}

Tmat<double> cir_sim_mat(int n_obs, int n_sim, double EPS)
{
    Tmat<double> res(n_sim, n_obs);
    
    for(int i=0; i<n_sim; i++){
            res.row(i) = cir_sim_vec(n_obs, EPS);
    }
    
    return res;
    
}

// [[Rcpp::export]]
Tmat<double> cir_max(int n_obs, int n_sim, double EPS)
{
    // Generate cir_sim_mat
    Tmat<double> cir_mat = cir_sim_mat(n_obs, n_sim, EPS);
    
    // Find vector of max
    Tvec<double> cir_max = cir_mat.rowwise().maxCoeff();
    
    // return vector of max
    return cir_max;
}



/*** R

if(F){
    
    library(ggplot2)
    library(dplyr)
    library(tidyr)
    
    EPS <- 1e-7
    m <- 1000
    n <- seq(1, 1000000, by = 10)
    mean_mcir <- sd_mcir <- numeric(length(n))
    
    x <- cir_max(max(n), m, EPS)   
    mean(x)
    
    pb <- txtProgressBar(min = 0, max = length(n), style = 3)
    for(i in 1:length(n)){
        x <- 0.5 * cir_max(n[i], m, EPS)   
        mean_mcir[i] <- mean(x)
        sd_mcir[i] <- sd(x)
        setTxtProgressBar(pb, i)
    }
    close(pb)
    
    df <- data.frame(
        "n" = n,
        "mean max cir" = mean_mcir,
        "upper" = mean_mcir + sd_mcir,
        "lower" = mean_mcir - sd_mcir
    )
    
    
    dflong <- gather(df, group, value, -n)
    dflong %>%
        ggplot() + 
        geom_line(aes(x=n, y=value, colour=group), size=1.5) + 
        xlab("Number of observations") + 
        ylab("Max CIR") + 
        theme_minimal() + 
        theme(legend.position = "bottom")
    
    
    EPS <- 1e-100
    0.5 * (log((1-EPS)^2) - log(EPS^2))
    
    
       
    
}

*/
