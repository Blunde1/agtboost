// node.hpp

#ifndef __NODE_HPP_INCLUDED__
#define __NODE_HPP_INCLUDED__


#include "cir.hpp"
#include "external_rcpp.hpp"
#include "gumbel.hpp"


// CLASS

class node
{
public:
    
    int split_feature; // j
    int obs_in_node; // |I_t|
    
    double split_value; // s_j
    double node_prediction; // w_t
    double node_tr_loss; // -G_t^2 / (2*H_t)
    double prob_node; // p(q(x)=t)
    double local_optimism; // C(t|q) = E[(g+hw_0)^2]/(n_tE[h])
    double expected_max_S; // E[S_max]
    //double split_point_optimism; // C(\hat{s}) = C(t|q)*p(q(x)=t)*(E[S_max]-1)
    double CRt; // p(q(x)=t) * C(t|q) * E[S_max]
    double p_split_CRt; // p(split(left,right) | q(x)=t) * CRt, p(split(left,right) | q(x)=t) \approx nl/nt for left node
    double g_sum_in_node;
    double h_sum_in_node;
    
    node* left;
    node* right;
    
    //node(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
    //     int obs_in_node, int obs_in_parent, int obs_tot);
    
    void createLeaf(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
                     int obs_in_node, int obs_in_parent, int obs_tot, 
                     double _g_sum_in_node, double _h_sum_in_node);
    
    node* getLeft();
    node* getRight();
    
    void split_node(Tvec<double> &g, Tvec<double> &h, Tvec<int> &ind, Tmat<double> &X, Tmat<double> &cir_sim, node* nptr, int n, 
                    double next_tree_score, bool greedy_complexities, double learning_rate,
                    int depth=0, int maxDepth = 1); // take out from node?
    
    bool split_information(const Tvec<double> &g, const Tvec<double> &h, const Tvec<int> &ind, const Tmat<double> &X,
                           const Tmat<double> &cir_sim, const int n);
    
    double expected_reduction(double learning_rate = 1.0);
    
    void reset_node(); // if no-split, reset j, s_j, E[S] and child nodes
    
    void print_child_branches(const std::string& prefix, const node* nptr, bool isLeft);
    void print_child_branches_2(const std::string& prefix, const node* nptr, bool isLeft);
    
    void serialize(node* nptr, std::ofstream& f);
    bool deSerialize(node *nptr, std::ifstream& f);
};

/*
// Constructor
node::node(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
                       int obs_in_node, int obs_in_parent, int obs_tot)
{
    this->node_prediction = node_prediction;
    this->node_tr_loss = node_tr_loss;
    this->local_optimism = local_optimism;
    this->prob_node = (double)obs_in_node / obs_tot; // prob_node;
    double prob_split_complement = 1.0 - (double)obs_in_node / obs_in_parent; // if left: p(right, not left), oposite for right
    this->p_split_CRt = prob_split_complement * CRt;
    this->obs_in_node = obs_in_node;
    this->left = NULL;
    this->right = NULL;
}
*/

// METHODS

void node::serialize(node* nptr, std::ofstream& f)
{
    // Check for null
    int MARKER = -1;
    if(nptr == NULL)
    {
        f << MARKER << "\n";
        return;
    }
    
    // Else, store information on node
    f << std::fixed << nptr->split_feature << " ";
    f << std::fixed << nptr->obs_in_node << " ";
    f << std::fixed << nptr->split_value << " ";
    f << std::fixed << nptr->node_prediction << " ";
    f << std::fixed << nptr->node_tr_loss << " ";
    f << std::fixed << nptr->prob_node << " ";
    f << std::fixed << nptr->local_optimism << " ";
    f << std::fixed << nptr->expected_max_S << " ";
    f << std::fixed << nptr->CRt << " ";
    f << std::fixed << nptr->p_split_CRt << "\n";

    // Recurrence
    serialize(nptr->left, f);
    serialize(nptr->right, f);
    
}

bool node::deSerialize(node *nptr, std::ifstream& f)
{
    
    int MARKER = -1;

    std::string stemp;
    if(!std::getline(f,stemp)){
        nptr = NULL;
        return false;
    }
    
    // Check stemp for MARKER
    std::istringstream istemp(stemp);
    int val;
    istemp >> val;
    if(val == MARKER){
        nptr = NULL;
        return false;
    }
    
    // Load node
    nptr->split_feature = val;
    istemp >> nptr->obs_in_node >> nptr->split_value >> nptr->node_prediction >>
        nptr->node_tr_loss >> nptr->prob_node >> nptr->local_optimism >>
        nptr->expected_max_S >> nptr->CRt >> nptr->p_split_CRt;
    
    // Node check value
    bool node_success = false;
    
    // Left node
    node* new_left = new node;
    node_success = deSerialize(new_left, f);
    if(node_success)
    {
        nptr->left = new_left;
    }else{
        nptr->left = NULL;
    }
    
    // Right node
    node_success = false;
    node* new_right = new node;
    node_success = deSerialize(new_right, f);
    if(node_success)
    {
        nptr->right = new_right;
    }else{
        nptr->right = NULL;
    }
    
    return true;
}

void node::createLeaf(double node_prediction, double node_tr_loss, double local_optimism, double CRt,
                       int obs_in_node, int obs_in_parent, int obs_tot,
                       double _g_sum_in_node, double _h_sum_in_node)
{
    //node* n = new node;
    this->node_prediction = node_prediction;
    this->node_tr_loss = node_tr_loss;
    this->local_optimism = local_optimism;
    this->prob_node = (double)obs_in_node / obs_tot; // prob_node;
    double prob_split_complement = 1.0 - (double)obs_in_node / obs_in_parent; // if left: p(right, not left), oposite for right
    this->p_split_CRt = prob_split_complement * CRt;
    this->obs_in_node = obs_in_node;
    this->g_sum_in_node = _g_sum_in_node;
    this->h_sum_in_node = _h_sum_in_node;
    this->left = NULL;
    this->right = NULL;
    
    //return n;
}


node* node::getLeft()
{
    return this->left;
}

node* node::getRight()
{
    return this->right;
}

double node::expected_reduction(double learning_rate)
{
    // Calculate expected reduction on node
    node* left = this->left;
    node* right = this->right;
    
    double loss_parent = this->node_tr_loss;
    double loss_l = left->node_tr_loss;
    double loss_r = right->node_tr_loss;
    
    double R = (loss_parent - loss_l - loss_r);
    double CR = left->p_split_CRt + right->p_split_CRt;
    
    return learning_rate*(2.0-learning_rate)*R-learning_rate*CR;
    
}

void node::reset_node()
{
    
    // Reset node
    this->expected_max_S = 0.0;
    this->split_feature = 0;
    this->split_value = 0.0;
    this->p_split_CRt = 0.0;
    this->CRt = 0.0;
    //this->split_point_optimism = 0.0;
    this->left = NULL;
    this->right = NULL;
    
}


// Algorithm 2 in Appendix C
bool node::split_information(const Tvec<double> &g, const Tvec<double> &h, const Tvec<int> &ind, const Tmat<double> &X,
                             const Tmat<double> &cir_sim, const int n)
{
    // 1. Creates left right node
    // 2. Calculations under null hypothesis
    // 3. Loop over features
    // 3.1 Profiles over all possible splits
    // 3.2 Simultaniously builds observations vectors
    // 3.3.1 Build gumbel (or gamma-one-hot) cdf of max cir for feature j
    // 3.3.2 Update joint cdf of max max cir over all features
    // 4. Estimate E[S]
    // 5. Estimate local optimism and probabilities
    // 6. Update split information in child nodes, importantly p_split_CRt
    // 7. Returns false if no split happened, else true
    
    int split_feature =0, n_indices = ind.size(), n_left = 0, n_right = 0, n_features = X.cols(), n_sim = cir_sim.rows();
    double split_val=0.0, observed_reduction=0.0, split_score=0.0, w_l=0.0, w_r=0.0, tr_loss_l=0.0, tr_loss_r=0.0;
    double Gl_final, Gr_final, Hl_final, Hr_final;
    
    // Return value
    bool any_split = false;
    
    // Iterators
    int j, i;
    
    // Sorting 
    //Tvec<double> vm(n_indices);
    Tvec<size_t> idx(n_indices);
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    
    // Local optimism
    double local_opt_l=0.0, local_opt_r=0.0;
    double Gl, Gl2, Hl, Hl2, gxhl, Gr, Gr2, Hr, Hr2, gxhr;    
    double G=0, H=0, G2=0, H2=0, gxh=0;
    
    // Prepare for CIR
    Tvec<double> u_store(n_indices);
    //double prob_delta = 1.0/n;
    double prob_delta = 1.0/n_indices;
    int num_splits;
    Tavec<double> max_cir(n_sim);
    int grid_size = 101; // should be odd
    double grid_end = 1.5*cir_sim.maxCoeff();
    Tvec<double> grid = Tvec<double>::LinSpaced( grid_size, 0.0, grid_end );
    Tavec<double> gum_cdf_grid(grid_size);
    Tavec<double> gum_cdf_mmcir_grid = Tvec<double>::Ones(grid_size);
    Tvec<double> gum_cdf_mmcir_complement(grid_size);
    
    
    // 1. Create child nodes
    node* left = new node;
    node* right = new node;
    
    // 2. Calculations under null hypothesis
    for(i=0; i<n_indices; i++){
        G += g[ind[i]]; H+=h[ind[i]];
        G2 += g[ind[i]]*g[ind[i]]; H2 += h[ind[i]]*h[ind[i]];
        gxh += g[ind[i]]*h[ind[i]];
    }
    
    // 3. Loop over features
    for(j=0; j<n_features; j++){
        
        // 3.1 Profiles over all possible splits
        Gl = 0.0; Hl=0.0; Gl2=0; Hl2=0, gxhl=0;
        //vm = X.col(j);
        std::sort(idx.data(), idx.data() + idx.size(), [&](int a, int b){return X(ind[a],j) < X(ind[b],j);});
        //idx = sort_indexes(vm);
        
        // 3.2 Simultaniously build observations vectors
        u_store.setZero();
        num_splits = 0;
        
        for(i=0; i<(n_indices-1); i++){
            
            // Left split calculations
            Gl += g[ind[idx[i]]]; Hl+=h[ind[idx[i]]];
            Gl2 += g[ind[idx[i]]]*g[ind[idx[i]]]; Hl2 += h[ind[idx[i]]]*h[ind[idx[i]]];
            gxhl += g[ind[idx[i]]]*h[ind[idx[i]]];
            
            // Right split calculations
            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            gxhr = gxh - gxhl;
            
            // Is x_i the same as next?
            if(X(ind[idx[i+1]],j) > X(ind[idx[i]],j)){
            //if(vm[idx[i+1]] > vm[idx[i]]){
                
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
                    split_val = X(ind[idx[i]],j);
                    w_l = -Gl/Hl;
                    w_r = -Gr/Hr;
                    tr_loss_l = -Gl*Gl / (Hl*2.0*n);
                    tr_loss_r = -Gr*Gr / (Hr*2.0*n);
                    n_left = i+1;
                    n_right = n_indices - (i+1);
                    // Eq. 25 in paper
                    local_opt_l = (Gl2 - 2.0*gxhl*(Gl/Hl) + Gl*Gl*Hl2/(Hl*Hl)) / (Hl*(i+1));
                    local_opt_r = (Gr2 - 2.0*gxhr*(Gr/Hr) + Gr*Gr*Hr2/(Hr*Hr)) / (Hr*(n_indices-(i+1)));
                    // Store more information
                    Gl_final = Gl;
                    Gr_final = Gr;
                    Hl_final = Hl;
                    Hr_final = Hr;
                }
                
            }
            
        }
        
        // 3.3 Estimate empirical cdf for feature j
        if(num_splits > 0){
            // At least one split-point
            
            // Get probabilities
            Tvec<double> u = u_store.head(num_splits);
            //Rcpp::Rcout << "u: \n" <<  u << std::endl; // COMMENT REMOVE
            
            // Get observations of max cir on probability observations
            max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
            
            if(num_splits==1){
                
                // Exactly gamma distrbuted: shape 1, scale 2
                
                // Estimate cdf of max cir for feature j
                for(int k=0; k<grid_size; k++){ 
                    gum_cdf_grid[k] = R::pgamma(grid[k], 0.5, 2.0, 1, 0); // lower tail, not log
                }
                
            }else{
                
                // Asymptotically Gumbel
                
                // Estimate Gumbel parameters
                Tvec<double> par_gumbel = par_gumbel_estimates(max_cir);
                
                // Estimate cdf of max cir for feature j
                for(int k=0; k<grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }
            }
            
            // Update empirical cdf for max max cir
            gum_cdf_mmcir_grid *= gum_cdf_grid;
            
        }
        
    }
    
    if(any_split){
        
        // 4. Estimate E[S]
        gum_cdf_mmcir_complement = Tvec<double>::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        this->expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        
        // 5. Update information in parent node -- reset later if no-split
        this->split_feature = split_feature;
        this->split_value = split_val;
        // C(s) = C(w|q)p(q)/2 * (E[S_max]-2)
        this->CRt = (this->prob_node)*(this->local_optimism)*(this->expected_max_S);
        //Rcpp::Rcout << "E[S]: " << this->expected_max_S << "\n" << "CRt: " << this->CRt << std::endl;
        //this->split_point_optimism = (local_opt_l*n_left + local_opt_r*n_right)/(2*n) * (this->expected_max_S - 2.0);
        
        
        // 6. Update split information in child nodes
        left->createLeaf(w_l, tr_loss_l, local_opt_l, this->CRt, 
                         n_left, n_left+n_right, n, Gl_final, Hl_final); // Update createLeaf()
        right->createLeaf(w_r, tr_loss_r, local_opt_r, this->CRt,
                          n_right, n_left+n_right, n, Gr_final, Hr_final);
        //Rcpp::Rcout << "p_left_CRt: " << left->p_split_CRt << "\n" <<  "p_right_CRt:"  << right->p_split_CRt << std::endl;
        
        // 7. update childs to left right
        this->left = left;
        this->right = right;
        
    }
    
    return any_split;
    
}

void node::split_node(Tvec<double> &g, Tvec<double> &h, Tvec<int> &ind, Tmat<double> &X, Tmat<double> &cir_sim, 
                      node* nptr, int n, 
                      double next_tree_score, bool greedy_complexities, double learning_rate,
                      int depth, int maxDepth)
{
    
    // if flags stop
    if(ind.size()<2){
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
    bool any_split = nptr->split_information(g, h, ind, X, cir_sim, n);
    
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
            // Compare with this root loss = next root loss
            // Can perhaps be done a little better...
            next_tree_score = std::max(0.0, nptr->expected_reduction(1.0));
            //next_tree_score = std::max(0.0, expected_reduction * (1.0 - learning_rate*(2.0-learning_rate)) );
        }
        
        double expected_reduction_normalized = nptr->expected_reduction(1.0) / nptr->prob_node;
        //double expected_reduction_normalized = expected_reduction / (nptr->prob_node);
        
        // Check trade-off
        if(expected_reduction_normalized < next_tree_score && depth > 0){
            nptr->reset_node();
            return;
        }
        
    }
    
    // Tests ok: create new left right indices for partition
    int n_left = nptr->left->obs_in_node;
    int n_right = nptr->right->obs_in_node;
    Tvec<int> ind_left(n_left), ind_right(n_right);
    // Any way to get the idx from split_information?...
    Tvec<size_t> idx(ind.size());
    std::iota(idx.data(), idx.data()+idx.size(), 0);
    std::sort(idx.data(), idx.data() + idx.size(), 
              [&](int a, int b){
                  return X(ind[a],nptr->split_feature) < X(ind[b],nptr->split_feature);
                  }
              );
    for(int i=0; i<n_left; i++){
        ind_left[i] = ind[idx[i]];
    }
    for(int i=n_left; i<(n_left+n_right); i++){
        ind_right[i-n_left] = ind[idx[i]];
    }
    
    // Run recursively on left
    split_node(g, h, ind_left, X, cir_sim, nptr->left, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
    // Run recursively on right 
    split_node(g, h, ind_right, X, cir_sim, nptr->right, n, 
               next_tree_score, greedy_complexities, learning_rate, 
               depth+1, maxDepth);
    
    /*
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
     */
    
}



#endif
