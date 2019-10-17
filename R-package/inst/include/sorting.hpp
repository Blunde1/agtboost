// Order gamma

#ifndef __SORTING_HPP_INCLUDED__
#define __SORTING_HPP_INCLUDED__


template <typename T>
Tvec<size_t> sort_indexes(const Tvec<T> &v) {
    
    // Initialize
    Tvec<size_t> idx(v.size());
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    
    // Sort with lambda functionality
    std::sort(idx.data(), idx.data() + idx.size(),
              [&v](int i1, int i2){return v[i1] < v[i2];});
    
    // Return
    return idx;
}


Rcpp::List unbiased_splitting(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int n, int min_obs_in_node=2){
    
    //int n = g.size(), // should be n of full data
    int n_indices = g.size(), n_left=0, n_right=0;
    double split_val=0, split_feature=0, split_score=0, 
        observed_reduction=0, expected_reduction=0,
        w_l=0, w_r=0, bias_left=0, bias_right=0,
        score_left=0, score_right=0;
    double G=0, H=0, G2=0, H2=0, gxh=0;
    bool any_split = false;
    int m=0, i=0;
    
    double score = 0;
    for(i=0; i<n_indices; i++){
        G += g[i]; H+=h[i];
        G2 += g[i]*g[i]; H2 += h[i]*h[i];
        gxh += g[i]*h[i];
    }
    double C = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n_indices);
    
    double Gl, Gl2, Hl, Hl2, gxhl, Gr, Gr2, Hr, Hr2, gxhr, Cl, Cr;
    
    Tvec<double> vm(n_indices);
    Tvec<size_t> idx(n_indices);
    
    // Loop over features
    for(m=0; m<X.cols(); m++){
        
        Gl = 0; Hl=0; Gl2=0; Hl2=0, gxhl=0;
        vm = X.col(m);
        idx = sort_indexes(vm);
        
        // Loop over all possible splits
        for(i=0; i<(n_indices-1); i++){
            
            // Left split
            Gl += g[idx[i]]; Hl+=h[idx[i]];
            Gl2 += g[idx[i]]*g[idx[i]]; Hl2 += h[idx[i]]*h[idx[i]];
            gxhl += g[idx[i]]*h[idx[i]];
            // Bias left
            Cl = (Gl2 - 2.0*gxhl*(Gl/Hl) + Gl*Gl*Hl2/(Hl*Hl)) / (Hl*(i+1));
            
            // Right split
            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            gxhr = gxh - gxhl;
            // Bias right
            Cr = (Gr2 - 2.0*gxhr*(Gr/Hr) + Gr*Gr*Hr2/(Hr*Hr)) / (Hr*(n_indices-(i+1)));
            
            split_score = (Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2.0*n);// + (C-Cl-Cr);
            //(C*n_indices-Cl*(i+1)-Cr*(n_indices-(i+1)))/n;
            
            // score
            // Make sure all values equal to x_i,j has been counted before checking score
            if( score < split_score && vm[idx[i+1]] > vm[idx[i]] ){
                any_split = true;
                score = split_score; //(Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2.0*n) + (C-Cl-Cr);
                expected_reduction = (Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2.0*n) + (C-Cl-Cr);
                observed_reduction = (Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2*n);
                split_val = vm[idx[i]];
                split_feature = m;
                w_l = -Gl/Hl;
                w_r = -Gr/Hr;
                bias_left = Cl;
                bias_right = Cr;
                score_left = -Gl*Gl/(Hl*2.0*n);
                score_right = -Gr*Gr / (Hr*2.0*n);
                n_left = i+1;
                n_right = n_indices-(i+1);
            }
        }
    }
    
    return Rcpp::List::create(
        Named("split_val")  = split_val,
        Named("split_feature")  = split_feature,
        Named("expected_reduction") = expected_reduction,
        Named("observed_reduction") = observed_reduction,
        Named("pred_left") = w_l,
        Named("pred_right") = w_r,
        Named("bias_left") = bias_left,//*n_left/n, // change -- let this be conditional bias
        Named("bias_right") = bias_right,//*n_right/n, // change
        Named("score_left") = score_left, // change
        Named("score_right") = score_right, // change
        Named("n_left") = n_left,
        Named("n_right") = n_right,
        Named("any_split") = any_split
    );
    
}


#endif