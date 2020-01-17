// cir.hpp

#ifndef __CIR_HPP_INCLUDED__
#define __CIR_HPP_INCLUDED__


#include "external_rcpp.hpp"


/*
 * cir_sim_vec:
 * Returns cir simulation on transformed equidistant grid tau = f(u)
 */
Tvec<double> cir_sim_vec(int m)
{
    double EPS = 1e-7;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( m+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(m, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    
    // Find cir delta
    Tvec<double> tau_delta = tau.tail(m-1) - tau.head(m-1);
    
    // Simulate first observation
    Tvec<double> res(m);
    res[0] = R::rgamma( 1.0, 2.0 );
    
    // Simulate remaining observatins
    
    double kappa=2.0, sigma = 2.0*sqrt(2.0);
    double a = kappa;
    double b = 2.0 * sigma*sigma / (4.0 * kappa);
    double c = 0;
    double ncchisq;
    
    for(int i=1; i<m; i++){
        
        c = 2.0 * a / ( sigma*sigma * (1.0 - exp(-a*tau_delta[i-1])) );
        ncchisq =  R::rnchisq( 4.0*a*b/(sigma*sigma), 2.0*c*res[i-1]*exp(-a*tau_delta[i-1]) );
        res[i] = ncchisq/(2.0*c);
        
    }
    
    return res;
    
}

/*
 * cir_sim_mat:
 * Returns 100 by 100 cir simulations
 */
Tmat<double> cir_sim_mat()
{
    int n=100, m=100;
    Tmat<double> res(n, m);
    
    for(int i=0; i<n; i++){
        res.row(i) = cir_sim_vec(m);
    }
    
    return res;
    
}


/*
 * interpolate_cir:
 * Returns interpolated observations between an observation vector, u, 
 * and pre-simlated cir observations in cir_sim
 */
Tmat<double> interpolate_cir(const Tvec<double>&u, const Tmat<double>& cir_sim)
{
    // cir long-term mean is 2.0 -- but do not use this! 
    double EPS = 1e-7;
    int cir_obs = cir_sim.cols();
    int n_timesteps = u.size();
    int n_sim = cir_sim.rows();
    int j=0;
    int i=0;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( cir_obs+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(cir_obs, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau_sim = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    Tvec<double> tau = 0.5 * log( (u.array()*(1-EPS))/(EPS*(1.0-u.array())) );
    
    // Find indices and weights of for simulations
    Tvec<int> lower_ind(n_timesteps), upper_ind(n_timesteps);
    Tvec<double> lower_weight(n_timesteps), upper_weight(n_timesteps);
    
    // Surpress to lower boundary
    for( ; i<n_timesteps; i++ ){
        if(tau[i] <= tau_sim[0]){
            lower_ind[i] = 0;
            lower_weight[i] = 1.0;
            upper_ind[i] = 0;
            upper_weight[i] = 0.0;
        }else{
            break;
        }
    }
    
    // Search until in-between cir_obs timepoints are found
    for( ; i<n_timesteps; i++ ){
        // If at limit and tau > tau_sim --> at limit --> break
        if( tau_sim[cir_obs-1] < tau[i] ){
            break;
        }
        for( ; j<(cir_obs-1); j++){
            if( tau_sim[j] < tau[i] && tau[i] <= tau_sim[j+1] ){
                lower_ind[i] = j;
                upper_ind[i] = j+1;
                lower_weight[i] = 1.0 - (tau[i]-tau_sim[j]) / (tau_sim[j+1]-tau_sim[j]);
                upper_weight[i] = 1.0 - lower_weight[i];
                break; // stop search
            }
        }
    }
    
    // Surpress to upper boundary
    for( ; i<n_timesteps; i++ ){
        if(tau[i] > tau_sim[cir_obs-1]){
            lower_ind[i] = cir_obs-1;
            upper_ind[i] = 0;
            lower_weight[i] = 1.0;
            upper_weight[i] = 0.0;
        }
    }
    
    // Populate the return matrix
    Tmat<double> cir_interpolated(n_sim, n_timesteps);
    cir_interpolated.setZero();
    
    for(i=0; i<n_sim; i++){
        for(j=0; j<n_timesteps; j++){
            cir_interpolated(i,j) = cir_sim( i, lower_ind[j] ) * lower_weight[j] + 
                cir_sim( i, upper_ind[j] ) * upper_weight[j];
            
        }
    }
    
    return cir_interpolated;
}

/*
 * rmax_cir:
 * Simulates maximum of cir observations on an observation vector u
 */
Tvec<double> rmax_cir(const Tvec<double>& u, const Tmat<double>& cir_sim)
{
    // Simulate maximum of observations on a cir process
    // u: split-points on 0-1
    // cir_sim: matrix of pre-simulated cir-processes on transformed interval

    int nsplits = u.size();
    int simsplits = cir_sim.cols();
    int nsims = cir_sim.rows();
    Tvec<double> max_cir_obs(nsims);
    
    if(nsplits < simsplits){
        double EPS = 1e-7;
        //int nsplits = u.size();
        
        // Transform interval: 0.5*log( (b*(1-a)) / (a*(1-b)) )
        Tvec<double> tau = 0.5 * log( (u.array()*(1-EPS))/(EPS*(1.0-u.array())) );
        
        // Interpolate cir-simulations 
        Tmat<double> cir_obs = interpolate_cir(u, cir_sim);

        // Calculate row-wise maximum (max of each cir process)
        max_cir_obs = cir_obs.rowwise().maxCoeff();
    }else{
        max_cir_obs = cir_sim.rowwise().maxCoeff();
    }
    
    return max_cir_obs;
}

/*
 * estimate_shape_scale: 
 * Estimates shape and scale for a Gamma distribution
 * Note: Approximately!
 */
Tvec<double> estimate_shape_scale(const Tvec<double> &max_cir)
{
    
    // Fast estimation through mean and var
    int n = max_cir.size();
    double mean = max_cir.sum() / n;
    double var = 0;
    for(int i=0; i<n; i++){
        var += (max_cir[i] - mean) * (max_cir[i] - mean) / (n-1);
    }
    double shape = mean * mean / var;
    // Surpress scale to minimum be 1.0? not necessary, bug elsewhere
    //double scale = std::max(1.0, var/mean);
    double scale = var / mean;
    Tvec<double> res(2);
    res << shape, scale;
    return res;
    
}

/*
 * pmax_cir:
 * \prod_j p( max_cir_j < s)
 */
double pmax_cir(double s, const Tmat<double> &gamma_param)
{
    
    int m = gamma_param.rows();
    double log_res = 0;
    for(int i=0; i<m; i++){
        log_res += R::pgamma( s, gamma_param(i,0), gamma_param(i, 1), 1, 1 );
    }
    
    if(std::isnan(exp(log_res))){
        return 0.0;
    }else{
        exp(log_res);   
    }
    
}

/*
 * expected_max_cir:
 * E[S] = \int_0^\infty (1 - pmax_cir(s)) ds
 */
double expected_max_cir(const Tmat<double> &gamma_param)
{
    // Calculate E[S] = \int 1-\prod p(s;\theta_i) ds
    
    // Smart choice of a and b
    int m = gamma_param.rows();
    double avg_shape = gamma_param.col(0).sum() / m;
    double avg_scale = gamma_param.col(1).sum() / m;
    double a = R::qgamma( 0.001, avg_shape, avg_scale, 1, 0);
    double b = R::qgamma( 0.999, avg_shape, avg_scale, 1, 0);
    
    double val = 0.0; //, a=0.0, b=10;
    int n=100;
    double h = (b-a)/n;
    
    // f(a)
    val = 1.0;
    
    // f(b)
    val += 1.0 - pmax_cir(b, gamma_param);
    
    // mid
    for(int j=1; j<=(n/2); j++){
        val += 4.0 * ( 1.0 - pmax_cir( h*(2*j-1), gamma_param ) );
    }
    for(int j=1; j<=(n/2-1); j++){
        val += 2.0 * ( 1.0 - pmax_cir( h*(2*j), gamma_param ) );
    }
    
    // Scale
    val = val * h/3;
    
    return val;
}

/*
 * expected_max_cir_approx:
 * E[S] \sim Q(m/(m+1), avg_shape, avg_scale)
 */
double expected_max_cir_approx(const Tmat<double> &gamma_param)
{
    int m = gamma_param.rows();
    double avg_shape = gamma_param.col(0).sum() / m;
    double avg_scale = gamma_param.col(1).sum() / m;
    
    return R::qgamma( (double)m/(m+1), avg_shape, avg_scale, 1, 0);
    
}

#endif