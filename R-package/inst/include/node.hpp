// node.hpp

#ifndef __NODE_HPP_INCLUDED__
#define __NODE_HPP_INCLUDED__


#include "external_rcpp.hpp"
#include "sorting.hpp"
#include "gtbic.hpp"


class node
{
public:
    int split_feature;
    int num_features;
    double prob_node; // Probability of being in node
    double split_value;
    double node_prediction;
    double score;
    double bias;
    node* left;
    node* right;
    node* sibling;
    
    node* createLeaf(//int split_feature,
            //double split_value,
            double node_prediction,
            double score,
            double bias,
            double prob_node);
    
    void setLeft(double node_prediction, double score, double bias, double prob_left);
    void setRight(double node_prediction, double score, double bias, double prob_right);
    node* getLeft();
    node* getRight();
    
    void split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, node* nptr, int n, 
                    double prob_parent,
                    double next_tree_score, bool greedy_complexities, double learning_rate,
                    int depth=0, int maxDepth = 1); // take out from node?
};


// --------------- NODE FUNCTIONS -----------

node* node::createLeaf(double node_prediction, double score, double bias, double prob_node)
{
    node* n = new node;
    n->node_prediction = node_prediction;
    n->score = score; 
    n->bias = bias;
    n->prob_node = prob_node;
    n->left = NULL;
    n->right = NULL;
    n->sibling = NULL;
    
    return n;
}

void node::setLeft(double node_prediction, double score, double bias, double prob_left){
    this->left = createLeaf(node_prediction, score, bias, prob_left);
}

void node::setRight(double node_prediction, double score, double bias, double prob_right){
    this->right = createLeaf(node_prediction, score, bias, prob_right);
}
node* node::getLeft(){
    return this->left;
}
node* node::getRight(){
    return this->right;
}

void node::split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, node* nptr, int n, 
                      double prob_parent, 
                      double next_tree_score, bool greedy_complexities, double learning_rate,
                      int depth, int maxDepth)
{
    
    // if flags stop
    if(g.size()<2){
        return;
    }else{
        //else check split
        
        // Calculate split information
        Rcpp::List split_information = unbiased_splitting(g, h, X, n);
        
        // Extract information
        double split_val = split_information["split_val"];
        int split_feature = split_information["split_feature"];
        double expected_reduction = split_information["expected_reduction"];
        double pred_left = split_information["pred_left"];
        double pred_right = split_information["pred_right"];
        double cond_bias_left = split_information["bias_left"];
        double cond_bias_right = split_information["bias_right"];
        double score_left = split_information["score_left"];
        double score_right = split_information["score_right"];
        int n_left = split_information["n_left"];
        int n_right = split_information["n_right"];
        
        // Calculate probs
        int n_parent = n_left + n_right;
        double prob_left = prob_parent * n_left/n_parent;
        double prob_right = prob_parent * n_right/n_parent;
        double bias_left = cond_bias_left * prob_left;
        double bias_right = cond_bias_right * prob_right;
        int m = X.cols();
        
        // This is wrong
        //expected_reduction = nptr->score - (score_left + score_right) + 
        //    nptr->bias - (2*std::max(1.0, log(m+1e-4)) + 1.0)*(bias_left + bias_right);
        
        // Create a sibling node pointer
        // optimism = 2 * ( m-th order stats Gamma(3, children) + Gamma(2.5, parent+parent_sibling) - Gamma(3, parent+parent_sibling))
        double previous_parent, current_parent, current_child, sum_optimism, observed_reduction;
        
        observed_reduction = nptr->score - (score_left + score_right);
        
        
        if(nptr->sibling != NULL){
            
            // Parent not root
            double parent_sibling_bias = nptr->bias + nptr->sibling->bias;
            
            if(nptr->sibling->left == NULL){
                
                // parent-siblig is leaf
                previous_parent = EXM_Optimism(m, 6.0, parent_sibling_bias);
                current_parent = EXM_Optimism(m, 5.0, parent_sibling_bias);
                
            }else{
                
                // Parent sibling is not leaf
                previous_parent = EXM_Optimism(m, 5.0, parent_sibling_bias);
                current_parent = EXM_Optimism(m, 4.0, parent_sibling_bias);
                
            }
            
        }else{
            
            // Parent is root
            previous_parent = 0.0; //nptr->bias;
            current_parent = 0.0;
        }
        
        current_child = EXM_Optimism(m, 6.0, bias_left + bias_right);
        sum_optimism = current_child + current_parent - previous_parent;
        
        expected_reduction = observed_reduction - sum_optimism;
        
        /*
         * Compare complexities - start
         * Calculate score expected_red / P(q(x)=t)
         * Must be larger than Root score * (1 - d(2-d)) d=learning rate
         * Save root score and use iteratively
         * Does not affect first split, as || depth==0 condition
         */
        double expected_reduction_normalized = 0.0;
        if(greedy_complexities){
            // Greedy-add-complexities
            expected_reduction_normalized = expected_reduction / prob_parent;
            
            if(nptr->sibling == NULL){
                // Calculate for deeper comparisons
                // Approximate asymptotic calculation
                next_tree_score = std::max(0.0, expected_reduction * (1.0 - learning_rate*(2.0-learning_rate)) );
            }
        }else{
            expected_reduction_normalized = expected_reduction; 
        }
        
        // For development
        //std::cout << expected_reduction << " " << expected_reduction_normalized << " " << next_tree_score << std::endl;
        
        /*
         * Compare complexities calculations - end
         */
        
        // If positive reduction
        // That possibly beats next tree
        if(expected_reduction_normalized >= next_tree_score || depth == 0){
            // next_tree_score set initially as 0.0
            //if(expected_reduction > 1e-9){
            // check depth
            //if(depth<maxDepth){ 
            //&& expected_reduction > 1e-14){    
            // Udpate node with split_value and feature
            nptr->split_value = split_val;
            nptr->split_feature=split_feature;
            nptr->num_features=m;
            
            // Create left leaf node
            nptr->left = createLeaf(pred_left, score_left, bias_left, prob_left);
            
            // Create right leaf node
            nptr->right = createLeaf(pred_right, score_right, bias_right, prob_right);
            
            // Set siblings -- perhaps do this somewhere else
            nptr->left->sibling = nptr->right;
            nptr->right->sibling = nptr->left;
            
            // Create new g, h, and X partitions
            Tvec<double> vm = X.col(split_feature);
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
            split_node(gl, hl, xl, nptr->left, n, prob_left, 
                       next_tree_score, greedy_complexities, learning_rate, 
                       depth+1, maxDepth);
            
            // Run recursively on right 
            split_node(gr, hr, xr, nptr->right, n, prob_right, 
                       next_tree_score, greedy_complexities, learning_rate, 
                       depth+1, maxDepth);
        }
    }
    
}


#endif