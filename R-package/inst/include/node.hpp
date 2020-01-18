// node.hpp

#ifndef __NODE_HPP_INCLUDED__
#define __NODE_HPP_INCLUDED__


#include "cir.hpp"
#include "external_rcpp.hpp"
#include "sorting.hpp"
#include "gtbic.hpp"


// CLASS

class node
{
public:
    int split_feature; // j
    double split_value; // s_j
    double node_prediction; // w_t
    double node_tr_loss; // -G_t^2 / H_t
    double local_optimism; // C(t|q)
    double expected_max_S; // E[S_max]
    double split_point_optimism; // C(\hat{s}) = C(t|q) / 2 * ( E[S_max] - 2)
    double prob_node; // p(q(x)=t)
    int obs_in_node; // |I_t|
    node* left;
    node* right;
    
    node* createLeaf(double node_prediction, double node_tr_loss, double local_optimism, 
                     double prob_node, int obs_in_node); // possibly more as well...
    
    void setLeft(double node_prediction, double node_tr_loss, double local_optimism, double prob_node, int obs_in_node);
    void setRight(double node_prediction, double node_tr_loss, double local_optimism, double prob_node, int obs_in_node);
    node* getLeft();
    node* getRight();
    
    void split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim, node* nptr, int n, 
                    double next_tree_score, bool greedy_complexities, double learning_rate,
                    int depth=0, int maxDepth = 1); // take out from node?
    
    bool split_information(const Tvec<double> &g, const Tvec<double> &h, const Tmat<double> &X,
                           const Tmat<double> &cir_sim, const int n);
    
    double expected_reduction();
    
    void reset_node(); // if no-split, reset j, s_j, E[S] and child nodes
    
    void print_child_branches(const std::string& prefix, const node* nptr, bool isLeft);
    void print_child_branches_2(const std::string& prefix, const node* nptr, bool isLeft);
    
};


// METHODS

node* node::createLeaf(double node_prediction, double node_tr_loss, double local_optimism, 
                       double prob_node, int obs_in_node)
{
    node* n = new node;
    n->node_prediction = node_prediction;
    n->node_tr_loss = node_tr_loss;
    n->local_optimism = local_optimism;
    n->prob_node = prob_node;
    n->obs_in_node = obs_in_node;
    n->left = NULL;
    n->right = NULL;
    
    return n;
}

void node::setLeft(double node_prediction, double node_tr_loss, double local_optimism, double prob_node, int obs_in_node)
{
    this->left = createLeaf(node_prediction, node_tr_loss, local_optimism, prob_node, obs_in_node);
}

void node::setRight(double node_prediction, double node_tr_loss, double local_optimism, double prob_node, int obs_in_node)
{
    this->right = createLeaf(node_prediction, node_tr_loss, local_optimism, prob_node, obs_in_node);
}

node* node::getLeft()
{
    return this->left;
}

node* node::getRight()
{
    return this->right;
}

double node::expected_reduction()
{
    // Calculate expected reduction on node
    node* left = this->left;
    node* right = this->right;
    
    double loss_parent = this->node_tr_loss;
    double loss_l = left->node_tr_loss;
    double loss_r = right->node_tr_loss;
    
    double cond_optimism_parent = this->local_optimism * this->prob_node;
    double cond_optimism_left = left->local_optimism * left->prob_node;
    double cond_optimism_right = right->local_optimism * right->prob_node;
    double s_hat_optimism = this->split_point_optimism;
    
    double res = (loss_parent - loss_l - loss_r) + 
        cond_optimism_parent - ( cond_optimism_left + cond_optimism_right + s_hat_optimism );
    //(cond_optimism_parent - S/2.0*(cond_optimism_left+cond_optimism_right));
    
    return res;
}

void node::reset_node()
{
    
    // Reset node
    this->expected_max_S = 0.0;
    this->split_feature = 0;
    this->split_value = 0.0;
    this->split_point_optimism = 0.0;
    this->left = NULL;
    this->right = NULL;
    
}


// Algorithm 2 in Appendix C
bool node::split_information(const Tvec<double> &g, const Tvec<double> &h, const Tmat<double> &X,
                             const Tmat<double> &cir_sim, const int n)
{
    // 1. Creates left right node
    // 2. Calculations under null hypothesis
    // 3. Loop over features
    // 3.1 Profiles over all possible splits
    // 3.2 Simultaniously builds observations vectors
    // 3.3 Store gamma param estimates
    // 4. Estimate E[S]
    // 5. Estimate local optimism and probabilities
    // 6. Update split information in child nodes
    // 7. Returns false if no split happened, else true
    
    int split_feature =0, n_indices = g.size(), n_left = 0, n_right = 0, n_features = X.cols(), n_sim = cir_sim.rows();
    double split_val=0.0, observed_reduction=0.0, split_score=0.0, w_l=0.0, w_r=0.0, tr_loss_l=0.0, tr_loss_r=0.0;
    
    // Return value
    bool any_split = false;
    
    // Iterators
    int j, i;
    
    // Sorting 
    Tvec<double> vm(n_indices);
    Tvec<size_t> idx(n_indices);
    
    // Local optimism
    double local_opt_l=0.0, local_opt_r=0.0;
    double Gl, Gl2, Hl, Hl2, gxhl, Gr, Gr2, Hr, Hr2, gxhr;    
    double G=0, H=0, G2=0, H2=0, gxh=0;
    
    // Prepare for CIR
    Tvec<double> u_store(n_indices);
    double prob_delta = 1.0/n_indices;
    int feature_counter=0, num_splits;
    Tvec<double> max_cir(n_sim);
    Tmat<double> gamma_param(n_features, 2); // Store estimated shape-scale parameters
    
    // 1. Create child nodes
    node* left = new node;
    node* right = new node;
    
    // 2. Calculations under null hypothesis
    for(i=0; i<n_indices; i++){
        G += g[i]; H+=h[i];
        G2 += g[i]*g[i]; H2 += h[i]*h[i];
        gxh += g[i]*h[i];
    }
    
    // 3. Loop over features
    for(j=0; j<n_features; j++){
        
        // 3.1 Profiles over all possible splits
        Gl = 0.0; Hl=0.0; Gl2=0; Hl2=0, gxhl=0;
        vm = X.col(j);
        idx = sort_indexes(vm);
        
        // 3.2 Simultaniously build observations vectors
        u_store.setZero();
        num_splits = 0;
        
        for(i=0; i<(n_indices-1); i++){
            
            // Left split calculations
            Gl += g[idx[i]]; Hl+=h[idx[i]];
            Gl2 += g[idx[i]]*g[idx[i]]; Hl2 += h[idx[i]]*h[idx[i]];
            gxhl += g[idx[i]]*h[idx[i]];
            
            // Right split calculations
            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            gxhr = gxh - gxhl;
            
            // Is x_i the same as next?
            if(vm[idx[i+1]] > vm[idx[i]]){
                
                // Update observation vector
                u_store[num_splits] = (i+1)*prob_delta;
                num_splits++;
                
                // Check for new maximum reduction
                split_score = (Gl*Gl/Hl + Gr*Gr/Hr - G*G/H)/(2.0*n);
                if(observed_reduction < split_score){
                    
                    any_split = true;
                    observed_reduction = split_score; // update
                    
                    // Populate nodes with information
                    split_feature = j;
                    split_val = vm[idx[i]];
                    w_l = -Gl/Hl;
                    w_r = -Gr/Hr;
                    tr_loss_l = -Gl*Gl / (Hl*2.0*n);
                    tr_loss_r = -Gr*Gr / (Hr*2.0*n);
                    n_left = i+1;
                    n_right = n_indices - (i+1);
                    // Eq. 25 in paper
                    local_opt_l = (Gl2 - 2.0*gxhl*(Gl/Hl) + Gl*Gl*Hl2/(Hl*Hl)) / (Hl*(i+1));
                    local_opt_r = (Gr2 - 2.0*gxhr*(Gr/Hr) + Gr*Gr*Hr2/(Hr*Hr)) / (Hr*(n_indices-(i+1)));
                    
                }
                
            }
            
        }
        
        // 3.3 Store gamma param estimates    
        if(num_splits > 0){
            // At least one split-point
            
            if(num_splits == 1){
                // one-hot encoding
                gamma_param.row(feature_counter) << 1.0, 2.0; // Asymptotic cir
            }else{
                // More than one split
                Tvec<double> u = u_store.head(num_splits);
                max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
                gamma_param.row(feature_counter) = estimate_shape_scale(max_cir);
            }
            
            feature_counter++;
            
        }
        
    }
    
    if(any_split){
        
        // 4. Estimate E[S]
        //this->expected_max_S = std::max(2.0, expected_max_cir(gamma_param.block(0,0,feature_counter, 2)));
        this->expected_max_S = std::max(2.0, expected_max_cir_approx(gamma_param.block(0,0,feature_counter, 2)));
        
        // 5. Update information in parent node -- reset later if no-split
        this->split_feature = split_feature;
        this->split_value = split_val;
        // C(s) = C(w|q)p(q)/2 * (E[S_max]-2)
        this->split_point_optimism = (local_opt_l*n_left + local_opt_r*n_right)/(2*n) * (this->expected_max_S - 2.0);
        
        // 6. Update split information in child nodes
        left = createLeaf(w_l, tr_loss_l, local_opt_l, (double)n_left/n, n_left); // Update createLeaf()
        right = createLeaf(w_r, tr_loss_r, local_opt_r, (double)n_right/n, n_right);
        
        // 7. update childs to left right
        this->left = left;
        this->right = right;
        
    }
    
    return any_split;
    
}

void node::split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim, 
                      node* nptr, int n, 
                      double next_tree_score, bool greedy_complexities, double learning_rate,
                      int depth, int maxDepth)
{
    
    // if flags stop
    if(g.size()<2){
        return;
    }
    
    /*
    // Check depth
    if(depth>=maxDepth){
        return;
    }
    */
    
    //else check split
    // Calculate split information
    bool any_split = nptr->split_information(g, h, X, cir_sim, n);
    
    // Check if a split happened
    if(!any_split){
        return;
    }
    
    // Comment out of working on depth<maxDepth
    double expected_reduction = nptr->expected_reduction();
    
    // Considering additive effects vs interaction effects trade-off?
    if(!greedy_complexities){
        
        // Don't consider trade-off
        // if expected_reduction < 0 then reset node
        // Force at least one split: kind-of approximately considering trade-off by not scaling with learning_rate
        if(expected_reduction < 0 && depth > 0){
            nptr->reset_node();
            return;
        }
        
    }else{
        
        // Consider trade-off
        
        // depth==0: calculate next_tree_score
        if(depth==0){
            // Quadratic approximation
            next_tree_score = std::max(0.0, expected_reduction * (1.0 - learning_rate*(2.0-learning_rate)) );
        }
        
        double expected_reduction_normalized = expected_reduction / (nptr->prob_node);
        
        // Check trade-off
        if(expected_reduction_normalized < next_tree_score && depth > 0){
            nptr->reset_node();
            return;
        }
        
    }
    
    // Tests ok: partition data and split child-nodes
    // Create new g, h, and X partitions
    int n_left = nptr->left->obs_in_node;
    int n_right = nptr->right->obs_in_node;
    Tvec<double> vm = X.col(nptr->split_feature);
    Tvec<size_t> idx = sort_indexes(vm);
    Tvec<double> gl(n_left), hl(n_left), gr(n_right), hr(n_right);
    Tmat<double> xl(n_left, X.cols()), xr(n_right, X.cols());
    
    for(int i=0; i<n_left; i++){
        gl[i] = g[idx[i]];
        hl[i] = h[idx[i]];
        xl.row(i) = X.row(idx[i]); 
    }
    for(int i=n_left; i<(n_left+n_right); i++){
        gr[i-n_left] = g[idx[i]];
        hr[i-n_left] = h[idx[i]];
        xr.row(i-n_left) = X.row(idx[i]); 
    }
    
    // Run recursively on left
    split_node(gl, hl, xl, cir_sim, nptr->left, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
    // Run recursively on right 
    split_node(gr, hr, xr, cir_sim, nptr->right, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
}



#endif