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

// ----------- LOSS --------------
double loss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type){
    int n = y.size();
    double res = 0;
    
    if(loss_type=="mse"){
        // MSE
        for(int i=0; i<n; i++){
            res += pow(y[i]-pred[i],2);
        }
        
    }else if(loss_type=="logloss"){
        for(int i=0; i<n; i++){
            res += y[i]*log(1.0+exp(-pred[i])) + (1.0-y[i])*log(1.0 + exp(pred[i]));
        }
    }
    
    return res/n;
    
}
Tvec<double> dloss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type){
    
    int n = y.size();
    Tvec<double> g(n);
    
    if(loss_type == "mse"){
        // MSE
        for(int i=0; i<n; i++){
            g[i] = -2*(y[i]-pred[i]);
        }
    }else if(loss_type == "logloss"){
        // LOGLOSS
        for(int i=0; i<n; i++){
            g[i] = ( exp(pred[i]) * (1.0-y[i]) - y[i] ) / ( 1.0 + exp(pred[i]) );
        }
    }
    
    return g;
}
Tvec<double> ddloss(Tvec<double> &y, Tvec<double> &pred, std::string loss_type="mse"){
    int n = y.size();
    Tvec<double> h(n);
    
    if( loss_type == "mse" ){
        for(int i=0; i<n; i++){
            h[i] = 2.0;
        }
    }else if(loss_type == "logloss"){
        // LOGLOSS
        for(int i=0; i<n; i++){
            h[i] = exp(pred[i]) / ( (exp(pred[i])+1.0)*(exp(pred[i])+1.0) ) ;
        }
    }
    
    return h;    
}

// ------------ CLASSES ---------------

class node
{
public:
    int split_feature;
    int num_features;
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
            double bias);
    
    void setLeft(double node_prediction, double score, double bias);
    void setRight(double node_prediction, double score, double bias);
    node* getLeft();
    node* getRight();
    
    void split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, node* nptr, int n, 
                    double prob_parent,
                    int depth=0, int maxDepth = 1); // take out from node?
};

class GBTREE
{
    
    //private:
public:
    
    node* root;
    GBTREE* next_tree;
    
    GBTREE();
    
    node* getRoot();
    void train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int maxDepth=1);
    double predict_obs(Tvec<double> &x);
    Tvec<double> predict_data(Tmat<double> &X);
    double getTreeScore();
    double getTreeBias();
    double getTreeBiasFull();
    double getTreeBiasFullEXM();
    int getNumLeaves();
    
};

//' @export ENSEMBLE
class ENSEMBLE
{
public:
    double initialPred;
    double learning_rate;
    double initial_score;
    GBTREE* first_tree;
    Rcpp::List param;
    
    // constructors
    ENSEMBLE();
    ENSEMBLE(double learning_rate_);
    
    // Functions
    void set_param(Rcpp::List par_list);
    Rcpp::List get_param();
    double initial_prediction(Tvec<double> &y, std::string loss_function);
    void train(Tvec<double> &y, Tmat<double> &X);
    Tvec<double> predict(Tmat<double> &X);
    Tvec<double> predict2(Tmat<double> &X, int num_trees);
    double get_ensemble_bias(int num_trees);
    int get_num_trees();
};

// ---- EXPECTED OPTIMISM ----
double EXM_Optimism(int M, double df, double child_bias){
    return 2*R::qgamma( (double)M / (M+1), df/2, child_bias / 2.0, 1, 0);
}


// ------------------- SORTING AND SPLITTING --------------------

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
        Named("n_right") = n_right
    );
    
}

// --------------- NODE FUNCTIONS -----------

node* node::createLeaf(double node_prediction, double score, double bias)
{
    node* n = new node;
    n->node_prediction = node_prediction;
    n->score = score; 
    n->bias = bias;
    n->left = NULL;
    n->right = NULL;
    n->sibling = NULL;
    
    return n;
}

void node::setLeft(double node_prediction, double score, double bias){
    this->left = createLeaf(node_prediction, score, bias);
}

void node::setRight(double node_prediction, double score, double bias){
    this->right = createLeaf(node_prediction, score, bias);
}
node* node::getLeft(){
    return this->left;
}
node* node::getRight(){
    return this->right;
}

void node::split_node(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, node* nptr, int n, 
                      double prob_parent,
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
        
        // If positive reduction
        if(expected_reduction > 1e-9 || depth == 0){
            //if(expected_reduction > 1e-9){
            // check depth
            //if(depth<maxDepth){ 
            //&& expected_reduction > 1e-14){    
            // Udpate node with split_value and feature
            nptr->split_value = split_val;
            nptr->split_feature=split_feature;
            nptr->num_features=m;
            
            // Create left leaf node
            nptr->left = createLeaf(pred_left, score_left, bias_left);
            
            // Create right leaf node
            nptr->right = createLeaf(pred_right, score_right, bias_right);
            
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
            split_node(gl, hl, xl, nptr->left, n, prob_left, depth+1, maxDepth);
            
            // Run recursively on right 
            split_node(gr, hr, xr, nptr->right, n, prob_right, depth+1, maxDepth);
        }
    }
    
}

// --------------- TREE FUNCTIONS -------
GBTREE::GBTREE(){
    this->root = NULL;
    this->next_tree = NULL;
}

node* GBTREE::getRoot(){
    return this->root;
}


void GBTREE::train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int maxDepth)
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
        double C = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
        root = root->createLeaf(-G/H, -G*G/(2*H), C);
        
    }
    
    root->split_node(g, h, X, root, n, 1.0,  0, maxDepth);
    
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
    Tvec<double> res(n), x(n);
    
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
            treeScore += current->score;
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

double GBTREE::getTreeBias(){
    // Recurse tree and sum leaf bias
    double treeBias = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            treeBias += current->bias;
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
            
            return treeBias;
}

double GBTREE::getTreeBiasFull(){
    // Recurse tree and sum leaf bias
    double treeBias = 0;
    
    node* current = this->root;
    treeBias = - 2*(current->bias);
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            treeBias += 2*(current->bias);
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
                treeBias += 2*(current->bias);
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return treeBias;
}

double GBTREE::getTreeBiasFullEXM(){
    // Recurse tree and sum leaf bias
    // Work on bias in child nodes, neglect leaves
    double treeBias = 0, child_bias=0;
    int M;
    node* current = this->root;
    //treeBias = - 2*(current->bias);
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            // LEAF :: DO NOT SUM
            //std::cout <<  current->node_prediction << std::endl; 
            //treeBias += 2*(current->bias);
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
                child_bias = (current->left->bias) + (current->right->bias);
                M = current->num_features;
                
                if (current->left->left == NULL && current->right->left == NULL) { 
                    // Check if both child nodes are leaves -- enough to check if left equals NULL (always binary split)
                    treeBias += 2*R::qgamma( (double)M / (M+1), 3.0, child_bias / 2.0, 1, 0);
                } 
                else if(current->left->left == NULL || current->right->left == NULL){
                    // Check if only one child node is leaf
                    treeBias += 2*R::qgamma( (double)M/(M+1), 2.5, child_bias / 2.0, 1, 0);
                }
                else{
                    // No child is leaf
                    treeBias += 2 * R::qgamma( (double)M / (M+1), 2.0, child_bias / 2.0, 1, 0);
                    
                }
                
                // POSSIBLY, IF CHILDS ARE LEAVES, THEN DO FULL OR SOMETHING.... TRY OUT
                // - BIAS?
                // 2/3EXM VS EXM IN LEAF?
                //treeBias += 2*(current->bias);
                current = current->right; 
            } /* End of if condition pre->right == NULL */
        } /* End of if condition current->left == NULL*/
    } /* End of while */
            
            return treeBias;
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

// ---------------- ENSEMBLE ----------------
ENSEMBLE::ENSEMBLE(){
    this->learning_rate=0.01;
    this->param = Rcpp::List::create(
        Named("learning_rate")  = 0.01,
        Named("loss_function")  = "mse",
        Named("nrounds") = 5000
    );
}

ENSEMBLE::ENSEMBLE(double learning_rate_){
    this->first_tree = NULL;
    this->learning_rate = learning_rate_;
    this->param = Rcpp::List::create(
        Named("learning_rate") = learning_rate_,
        Named("loss_function") = "mse",
        Named("nrounds") = 5000
    );
}

void ENSEMBLE::set_param(Rcpp::List par_list){
    this->param = par_list;
    this->learning_rate = par_list["learning_rate"];
}
Rcpp::List ENSEMBLE::get_param(){
    return this->param;
}

double ENSEMBLE::initial_prediction(Tvec<double> &y, std::string loss_function){
    
    double pred;
    int n = y.size();
    
    if(loss_function=="mse"){
        pred = y.sum() / n;
    }else if(loss_function=="logloss"){
        double pred_g_transform = y.sum()/n; // naive probability
        pred = log(pred_g_transform) - log(1 - pred_g_transform);
    }
    
    return pred;
}


void ENSEMBLE::train(Tvec<double> &y, Tmat<double> &X){
    // Set init -- mean
    int MAXITER = param["nrounds"];
    int NUM_BINTREE_CONSECUTIVE = 0;
    double MAX_NUM_BINTREE_CONSECUTIVE = 2 / learning_rate; // Logic: if learning_rate=1 and bintree, then should be no more splits
    int NUM_LEAVES;
    int n = y.size();
    //int m = X.size();
    double EPS = -1E-12;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> pred(n), g(n), h(n);
    
    // MSE -- FIX FOR OTHER LOSS FUNCTIONS
    this->initialPred = this->initial_prediction(y, param["loss_function"]); //y.sum()/n;
    pred.setConstant(this->initialPred);
    this->initial_score = loss(y, pred, param["loss_function"]); //(y - pred).squaredNorm() / n;
    
    // First tree
    g = dloss(y, pred, param["loss_function"]);
    h = ddloss(y, pred, param["loss_function"]);
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (this->first_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeBiasFullEXM();
    //( std::max(1.0, log(m+1e-4)) * current_tree->getTreeBiasFull() + current_tree->getTreeBias());// obs_loss + bias -- POSSIBLY SCALED
    
    // COUT
    // std::cout<< "learning rate: " << learning_rate << "\n" <<
    //     "initial prediction: " << (this->initialPred) << "\n" <<
    //         "iteration: " << 1 << "\n" << 
    //             "expected_loss: " << expected_loss << "\n" << std::endl;
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, param["loss_function"]);
        h = ddloss(y, pred, param["loss_function"]);
        new_tree->train(g, h, X);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeBiasFullEXM();
        //(new_tree->getTreeBiasFull() * std::max(1.0, log(m+1e-4)) + new_tree->getTreeBias());// obs_loss + bias -- POSSIBLY SCALED
        
        // // CHECKING IF BINARY TREE
        // NUM_LEAVES = new_tree->getNumLeaves();
        // if(NUM_LEAVES < 3){
        //     NUM_BINTREE_CONSECUTIVE++;
        // }else{
        //     NUM_BINTREE_CONSECUTIVE = 0;
        // }
        
        // std::cout << "iteration " << 
        //     i << "\n" << 
        //     "Num leaves: " << new_tree->getNumLeaves() << "\n" <<
        //         "expected_loss: " << expected_loss << "\n" << std::endl;
        
        
        if(expected_loss < EPS){ // && NUM_BINTREE_CONSECUTIVE < MAX_NUM_BINTREE_CONSECUTIVE){
            //if(new_tree->getNumLeaves() == 1){
            current_tree->next_tree = new_tree;
            pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
            current_tree = new_tree;
        }else{
            break;
        }
    }
}

Tvec<double> ENSEMBLE::predict(Tmat<double> &X){
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    while(current != NULL){
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        current = current->next_tree;
    }
    return pred;
}

Tvec<double> ENSEMBLE::predict2(Tmat<double> &X, int num_trees){
    int n = X.rows();
    int tree_num = 1;
    
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    
    
    if(num_trees < 1){
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    
    return pred;
}

double ENSEMBLE::get_ensemble_bias(int num_trees){
    
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    double total_optimism = 0.0;
    double learning_rate = this->learning_rate;
    GBTREE* current = this->first_tree;
    
    if(num_trees<1){
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeBiasFullEXM();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeBiasFullEXM();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    //std::cout<< (this->initial_score) << std::endl;
    return (this->initial_score) + total_observed_reduction * (-2)*learning_rate*(learning_rate/2 - 1) + 
        learning_rate * total_optimism;
    
}

int ENSEMBLE::get_num_trees(){
    int num_trees = 0;
    GBTREE* current = this->first_tree;
    
    while(current != NULL){
        num_trees++;
        current = current->next_tree;
    }
    
    return num_trees;
}

// Expose the classes
RCPP_MODULE(MyModule) {
    using namespace Rcpp;
    
    class_<ENSEMBLE>("ENSEMBLE")
        .default_constructor("Default constructor")
        .constructor<double>()
        .field("initialPred", &ENSEMBLE::initialPred)
        .method("set_param", &ENSEMBLE::set_param)
        .method("get_param", &ENSEMBLE::get_param)
        .method("train", &ENSEMBLE::train)
        .method("predict", &ENSEMBLE::predict)
        .method("predict2", &ENSEMBLE::predict2)
        .method("get_ensemble_bias", &ENSEMBLE::get_ensemble_bias)
        .method("get_num_trees", &ENSEMBLE::get_num_trees)
    ;
}