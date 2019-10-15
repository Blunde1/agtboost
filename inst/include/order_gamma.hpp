// Order gamma

#ifndef __ORDERGAMMA_HPP_INCLUDED__
#define __ORDERGAMMA_HPP_INCLUDED__


double choose_exm_approx(int M, double shape, double scale){

    
    // Direct quantile
    double quantile_approx = 2*R::qgamma( (double)M / (M+1), shape, scale, 1, 0);
    
    return quantile_approx;
    
    
    // SOME MISTAKE HERE - FIX LATER OR FIGURE OUT EXACT EXM
    /*
    double C, D, B;
    // von Mises asymptotic
    D = R::qgamma( 1.0 - 1/(double)M, shape, scale, 1, 0 );
    C = scale * ( 1.0+shape*(scale - 1.0)/D );
    double von_mises_approx = 2 * (C * (-R::digamma(1)) + D);
    
    // generalized lambert w asymptotic
    B = log(M) - lgamma(shape) + (shape-1.0)*log(shape-1.0);
    D = scale * ( log((double)M)- lgamma(shape) + (shape-1.0)*log(B) + 
        ( pow(shape-1.0,2) * log(B) - pow(shape-1.0,2)*log(shape-1.0) + shape - 1.0  ) / B  );
    C = D * scale * (D+scale*(shape-1.0)) / ( D*D - scale*scale*(shape-1.0)*(shape-2.0) );
    double lambert_w_approx = 2 * (C * (-R::digamma(1)) + D);
    
    
    // Return max
    if(quantile_approx >= von_mises_approx && quantile_approx >= lambert_w_approx){
        return quantile_approx;
    }else if(von_mises_approx >= lambert_w_approx){
        return von_mises_approx;
    }else{
        return lambert_w_approx;
    }
     */
    
    // To-do: implement other approximations
    
}


#endif