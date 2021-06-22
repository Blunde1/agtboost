// tree.hpp

#ifndef __TREE_HPP_INCLUDED__
#define __TREE_HPP_INCLUDED__


#include "node.hpp"
#include "external_rcpp.hpp"


class GBTREE
{
    
    //private:
public:
    
    node* root;
    GBTREE* next_tree;
    
    GBTREE();

    node* getRoot();
    void train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim,
               Tvec<double> &ensemble_influence,
               bool greedy_complexities, double learning_rate, int maxDepth=1);
    double predict_obs(Tvec<double> &x);
    Tvec<double> predict_data(Tmat<double> &X);
    double getTreeScore();
    double getConditionalOptimism();
    double getFeatureMapOptimism();
    double getTreeOptimism(); // sum of the conditional and feature map optimism
    int getNumLeaves();
    void print_tree(int type);
    void serialize(GBTREE* tptr, std::ofstream& f);
    bool deSerialize(GBTREE* tptr,  std::ifstream& f, int& lineNum);
    void importance(Tvec<double> &importance_vector, double learning_rate);
    
    double selfInfluence(double g, double h, Tvec<double> &x);
    Tvec<double> predict_data_cvn(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X);
    Tvec<double> residualInfluence(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X);
    Tvec<double> residualScore(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X);
};


// METHODS

void GBTREE::serialize(GBTREE *tptr, std::ofstream& f)
{
    
    int MARKER = -1;
    // If current tree is NULL, store marker
    if(tptr == NULL)
    {
        f << MARKER;
        return;
    }
    
    // Else, store current tree, recur on next tree
    tptr->root->serialize(tptr->root, f);
    serialize(tptr->next_tree, f);
}

bool GBTREE::deSerialize(GBTREE *tptr, std::ifstream& f, int& lineNum)
{
    
    int MARKER = -1;
    
    // Start at beginning
    f.seekg(0, std::ios::beg);
    
    // Run until line lineNum is found
    std::string stemp;
    for(int i=0; i<= lineNum; i++)
    {
        if(!std::getline(f,stemp)){
            tptr = NULL;
            return false;
        }
    }
    
    // Check stemp for MARKER
    std::istringstream istemp(stemp);
    int val;
    istemp >> val;
    if(val == MARKER){ 
        tptr = NULL;
        return false;
    }

    // If not MARKER, deserialize root node (unincremented lineNum)
    tptr->root = new node;
    tptr->root->deSerialize(tptr->root, f, lineNum); // lineNum passed by reference and incremented
    GBTREE* new_tree = new GBTREE;
    bool new_tree_exist = deSerialize(new_tree, f, lineNum);
    if(new_tree_exist)
    {
        tptr->next_tree = new_tree;
    }else{
        tptr->next_tree = NULL;
    }
    
    return true;
}

    
GBTREE::GBTREE(){
    this->root = NULL;
    this->next_tree = NULL;
}

node* GBTREE::getRoot(){
    return this->root;
}


void GBTREE::train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, Tmat<double> &cir_sim,
                   Tvec<double> &ensemble_influence,
                   bool greedy_complexities, double learning_rate, int maxDepth)
{
    // Check if root exists 
    // Else create root
    int n = g.size();
    
    if(root == NULL){
        // Calculate information
        double G=0, H=0, G2=0, H2=0, gxh=0;
        for(int i=0; i<n; i++){
            G += g[i]; H+=h[i];
            G2 += g[i]*g[i]; H2 += h[i]*h[i];
            gxh += g[i]*h[i];
        }
        double local_optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
        
        node* root_ptr = new node;
        root_ptr->createLeaf(-G/H, -G*G/(2*H*n), local_optimism, local_optimism, n, n, n, H/n);
        root = root_ptr;
        //root = root->createLeaf(-G/H, -G*G/(2*H*n), local_optimism, local_optimism, n, n, n);
    }
    
    // Root-node indices
    Tvec<int> ind(n);
    std::iota(ind.data(), ind.data()+ind.size(), 0);
    
    root->split_node(g, h, ind, X, cir_sim, root, n, 0.0, greedy_complexities, learning_rate, ensemble_influence, 0, maxDepth);
    
}

double GBTREE::selfInfluence(double g, double h, Tvec<double> &x){
    // LIC/TIC * Grad loss_i / average hessian in Node
    // LIC: sum path-node-reductions * p(q_t|x_i)
    // TIC: local-optimism * p(q_t|x_i)
    node* current = this->root;
    double profile_bias = 0.0;
    double influence = 0.0;
    double TIC = 0.0;
    double LIC = 0.0;
    double w_t = 0.0;
    double hess = 0.0;
    int leaf_obs = 0;
    if(current == NULL){
        return 0;
    }
    while(current != NULL){
        // get val from splitting
        if(current->left==NULL && current->right ==NULL){
            // Get contribution at leaf?
            //profile_bias = profile_bias * current->prob_node 
            TIC = current->prob_node * current->local_optimism;
            LIC = LIC * current->prob_node;
            w_t = current->node_prediction;
            hess = current->hess;
            leaf_obs = current->obs_in_node;
            break;
        }
        else{
            profile_bias += current->expected_max_S;
            LIC += current->CRt / (current->prob_node);
            // Follow path in tree
            if(x[current->split_feature]<=current->split_value){
                current = current->left;    
            }else{
                current = current->right;    
            }
        }
    }
    influence = -profile_bias * (g + h*w_t) / hess / leaf_obs;
    //influence = -std::max(LIC/TIC, 1.0) * (g + h*w_t) / hess / leaf_obs;
    //influence = -(g + h*w_t) / (hess * leaf_obs);
    if(std::isnan(influence)){
        Rcpp::Rcout << 
            "TIC: " << TIC << " - LIC: " << LIC << " - w_t: " << w_t << " - hess: " << hess << std::endl;
    }
    return influence;
}

Tvec<double> GBTREE::predict_data_cvn(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X){
    // pred - 1/n In(y_i, pred_i)
    int n = X.rows();
    int m = X.cols();
    Tvec<double> x(m);
    Tvec<double> pred = predict_data(X);
    for(int i=0; i<n; i++){
        x = X.row(i);
        pred[i] -= selfInfluence(g[i], h[i], x); // Should be n_t?
    }
    return pred;
}

Tvec<double> GBTREE::residualInfluence(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X){
    int n = X.rows();
    int m = X.cols();
    Tvec<double> x(m), residual_influence(n);
    for(int i=0; i<n; i++){
        x = X.row(i);
        residual_influence[i] = selfInfluence(g[i], h[i], x); // Should be n_t?
    }
    return residual_influence;
}

Tvec<double> GBTREE::residualScore(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X){
    int n = X.rows();
    Tvec<double> tree_pred = predict_data(X);
    Tvec<double> score(n);
    for(int i=0; i<n; i++){
        score[i] = g[i]+h[i]*tree_pred[i];
    }
    return score;
}

double GBTREE::predict_obs(Tvec<double> &x){
    
    node* current = this->root;
    
    if(current == NULL){
        return 0;
    }
    
    
    while(current != NULL){
        if(current->left == NULL && current ->right == NULL){
            return current->node_prediction;
        }
        else{
            
            if(x[current->split_feature] <= current->split_value){
                current = current->left;
            }else{
                current = current->right;
            }
        }
    }
    return 0;
}

Tvec<double> GBTREE::predict_data(Tmat<double> &X){
    
    int n = X.rows();
    int m = X.cols();
    Tvec<double> res(n), x(m);
    
    for(int i=0; i<n; i++){
        x = X.row(i);
        res[i] = predict_obs(x);
    }
    return res;
    
}

double GBTREE::getTreeScore(){
    // Recurse tree and sum leaf scores
    double treeScore = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) {
            //std::cout <<  current->node_prediction << std::endl; 
            treeScore += current->node_tr_loss;
            current = current->right;
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return treeScore;
}

double GBTREE::getConditionalOptimism(){
    // Recurse tree and sum conditional optimism in leaves
    double conditional_opt_leaves = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return conditional_opt_leaves;
}


double GBTREE::getTreeOptimism(){
    
    // Recurse tree and sum p_split_CRt in leaf-nodes
    double tree_optimism = 0.0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                tree_optimism += current->CRt * std::max(std::min(current->avg_informtion_left, 1.0), 0.0); // current->split_point_optimism;
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
    return tree_optimism;
    
}

void GBTREE::importance(Tvec<double> &importance_vector, double learning_rate){
    
    // Recurse tree and sum importance (reduction in generalization loss)
    int importance_feature = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //conditional_opt_leaves += current->local_optimism * current->prob_node;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                importance_feature = current->split_feature;
                importance_vector[importance_feature] += current->expected_reduction(learning_rate);
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
    return;
    
}

int GBTREE::getNumLeaves(){
    int numLeaves = 0;
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            numLeaves += 1;
            current = current->right; 
        } 
        else { 
            
            /* Find the inorder predecessor of current */
            pre = current->left; 
            while (pre->right != NULL && pre->right != current) 
                pre = pre->right; 
            
            /* Make current as right child of its inorder 
             predecessor */
            if (pre->right == NULL) { 
                pre->right = current; 
                current = current->left; 
            } 
            
            /* Revert the changes made in if part to restore 
             the original tree i.e., fix the right child 
             of predecssor */
            else { 
                pre->right = NULL; 
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
    return numLeaves;
}



#endif