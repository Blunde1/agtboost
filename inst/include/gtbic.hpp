// gtbic.hpp

#ifndef __GTBIC_HPP_INCLUDED__
#define __GTBIC_HPP_INCLUDED__

#include "node.hpp"
#include "order_gamma.hpp"

// ---- EXPECTED OPTIMISM ----
double EXM_Optimism(int M, double df, double child_bias){
    double shape = df/2.0;
    double scale = child_bias/2.0;
    double expected_m_gamma = choose_exm_approx(M, shape, scale);
    return expected_m_gamma;
    //return 2*R::qgamma( (double)M / (M+1), df/2, child_bias / 2.0, 1, 0);
}


#endif