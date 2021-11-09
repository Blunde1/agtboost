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
    double EPS = 1e-12;
    
    // Find original time of sim: assumption equidistant steps on u\in(0,1)
    double delta_time = 1.0 / ( m+1.0 );
    Tvec<double> u_cirsim = Tvec<double>::LinSpaced(m, delta_time, 1.0-delta_time);
    
    // Transform to CIR time
    Tvec<double> tau = 0.5 * log( (u_cirsim.array()*(1-EPS))/(EPS*(1.0-u_cirsim.array())) );
    
    // Find cir delta
    Tvec<double> tau_delta = tau.tail(m-1) - tau.head(m-1);
    
    // Parameters of CIR
    double a = 2.0;
    double b = 1.0;
    double sigma = 2.0*sqrt(2.0);
    double ncchisq;
    double c = 0.0;
    /*
    double kappa=2.0, sigma = 2.0*sqrt(2.0);
    double a = kappa;
    double b = 2.0 * sigma*sigma / (4.0 * kappa);
    double c = 0;
    double ncchisq;
    */
    
    // Simulate first observation
    // scale = 1/rate = 1/(2a/sigma^2) = sigma^2/(2*a)
    // shape = 2*a*b/sigma^2
    Tvec<double> res(m);
    res[0] = R::rgamma( 0.5, 2.0 );
    
    // Simulate remaining observatins
    for(int i=1; i<m; i++){
        
        c = 2.0 * a / ( sigma*sigma * (1.0 - exp(-a*tau_delta[i-1])) );
        ncchisq =  R::rnchisq( 4.0*a*b/(sigma*sigma), 2.0*c*res[i-1]*exp(-a*tau_delta[i-1]) );
        res[i] = ncchisq/(2.0*c);
        
    }
    
    return res;
    
}

/*
 * cir_sim_mat:
 * Returns 1000 by 1000 cir simulations
 */
Tmat<double> cir_sim_mat(int nsim, int nobs)
{
    //int n=100, m=200;
    Tmat<double> res(nsim, nobs);
    
    for(int i=0; i<nsim; i++){
        res.row(i) = cir_sim_vec(nobs);
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
    double EPS = 1e-12;
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
Tavec<double> rmax_cir(const Tvec<double>& u, const Tmat<double>& cir_sim)
{
    // Simulate maximum of observations on a cir process
    // u: split-points on 0-1
    // cir_sim: matrix of pre-simulated cir-processes on transformed interval

    int nsplits = u.size();
    int simsplits = cir_sim.cols();
    int nsims = cir_sim.rows();
    Tvec<double> max_cir_obs(nsims);
    
    if(nsplits < simsplits){
        double EPS = 1e-12;
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
    
    return max_cir_obs.array();
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

// Empirical cdf p_n(X\leq x)
double pmax_cir(double x, Tvec<double>& obs){
    
    // Returns proportion of values in obs less or equal to x
    int n = obs.size();
    int sum = 0;
    for(int i=0; i<n; i++){
        if( obs[i] <= x ){
            sum++;
        }
    }
    return (double)sum/n;
}

// Composite simpson's rule -- requires n even (grid.size() odd)
double simpson(Tvec<double>& fval, Tvec<double>& grid){
    
    // fval is f(x) on evenly spaced grid
    
    int n = grid.size() - 1;
    double h = (grid[n] - grid[0]) / n;
    double s = 0;
    
    if(n==2){
        s = fval[0] + 4.0*fval[1] + fval[2];
    }else{
        s = fval[0] + fval[n];
        for(int i=1; i<=(n/2); i++) { s += 4.0*fval[2*i-1]; }
        for(int i=1; i<=((n/2)-1); i++) { s += 2.0*fval[2*i]; }
    }
    s = s*h/3.0;
    return s;
    
}

#endif
