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
    double split_induced_variable;
    double percentage_moved;
    node* left;
    node* right;
    
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
    
    void print_child_branches(const std::string& prefix, const node* nptr, bool isLeft);
};

class GBTREE
{
    
    //private:
public:
    
    node* root;
    int maxDepth;
    
    GBTREE();
    GBTREE(int maxDepth_);
    
    node* getRoot();
    //void train_private(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, node* nPtr);
    void train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int maxDepth=1);
    double predict_obs(Tvec<double> &x);
    Tvec<double> predict_data(Tmat<double> &X);
    double getTreeScore();
    double getTreeBias();
    double getTreeBiasFull();
    double getTreeBiasFull2();
    double getTreeBiasFullEXM();
    int getNumLeaves();
    void print_tree();
    
};

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


// [[Rcpp::export]]
Rcpp::List unbiased_splitting(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int n, int min_obs_in_node=2){
    
    //int n = g.size(), // should be n of full data
    int n_indices = g.size(), n_left=0, n_right=0, m_cols = X.cols();
    double split_val=0, split_feature=0, split_score=0, 
        observed_reduction=0, expected_reduction=0,
        w_l=0, w_r=0, bias_left=0, bias_right=0,
        score_left=0, score_right=0;
    double G=0, H=0, G2=0, H2=0, gxh=0;
    bool any_split = false;
    Tvec<double> num_groups(m_cols); // Store the number of unique groups in column X[,j]
    Tvec<double> percentage_moved(m_cols); // Store how much percentage of datapoints are "moved" when checking splits
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
        
        // How many unique? This plays a role -- count 
        // Create a counter -- but later somewhere else
        // Can also be used to check if any splits happened --> if num_groups_m > 1 --> split
        int num_groups_m = 1; 
        double percentage_moved_m = 0.0; // value?
        double prob_right_prev = 0.0; // default max val -> no contrib from first split
        
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
            
            // Is x_i the same as before?
            if(vm[idx[i+1]] > vm[idx[i]]){
                
                // Update uniqueness counter
                num_groups_m++;
                
                // Update probability mass moved
                if(num_groups_m>2){
                    percentage_moved_m = percentage_moved_m + prob_right_prev - (double)(n_indices-(i+1))/n_indices; 
                }
                prob_right_prev = (double)(n_indices-(i+1))/n_indices; // Update
                
                // Check if score better than previous best score
                if(score < split_score){
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
        num_groups[m] = num_groups_m; // just update the counter directly above... later
        percentage_moved[m] = percentage_moved_m; // Update directly as above?
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
        Named("any_split") = any_split,
        Named("num_groups") = num_groups,
        Named("percentage_moved") = percentage_moved
    );
    
}

// --------------- NODE FUNCTIONS -----------

//GBTREE::node* GBTREE::createLeaf(
node* node::createLeaf(
        //int split_feature, 
        //double split_value, 
        double node_prediction,
        double score,
        double bias)
{
    node* n = new node;
    n->node_prediction = node_prediction;
    n->score = score; 
    n->bias = bias;
    n->left = NULL;
    n->right = NULL;
    
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
    if(g.size()<5){
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
        bool any_split = split_information["any_split"];
        Tvec<double> num_groups = split_information["num_groups"];
        Tvec<double> percentage_moved_both = split_information["percentage_moved"];
        //std::cout << percentage_moved_both << std::endl;
        
        // Check if a split happened
        if(!any_split){
            return;
        }
        
        // Calculate probs
        int n_parent = n_left + n_right;
        double w = -(g.sum()/h.sum());
        double testVar = (g + w*h).dot(g + w*h) * n_parent / (h.sum()*h.sum());
        /*
         std::cout << (g + w*h).dot(g + w*h) << std::endl;
         std::cout << w << std::endl;
         std::cout << testVar << std::endl;
         //double penalty_left = exp();
         //double penalty_right = 0;
         */
        double prob_left = prob_parent * n_left/n_parent;// * exp(-testVar*(depth+1)/(n_parent) * log(n_parent) );
        double prob_right = prob_parent * n_right/n_parent;// * exp(-testVar*(depth+1)/(n_parent) * log(n_parent) );
        double bias_left = cond_bias_left * prob_left;//score_left*(n*prob_left/n_left - 1.0) + cond_bias_left * prob_left;
        double bias_right = cond_bias_right * prob_right;//score_right * (n*prob_right/n_right - 1.0) + cond_bias_right * prob_right;
        
        // If positive reduction
        //if(expected_reduction > 1e-14){
        // check depth
        if(depth<maxDepth){ 
            //&& expected_reduction > 1e-14){    
            // Udpate node with split_value and feature
            nptr->split_value = split_val;
            nptr->split_feature=split_feature;
            nptr->num_features=X.cols();
            
            // Create left leaf node
            nptr->left = createLeaf(pred_left, score_left, bias_left);
            nptr->left->percentage_moved = percentage_moved_both[split_feature];
            // WARNING -- THIS ONLY WORKS IN 1 -DIMENSION!
            nptr->left->split_induced_variable=(num_groups[split_feature] - 1.0) / n_parent; // Ratio of possible splits to observations
            
            
            // Create right leaf node
            nptr->right = createLeaf(pred_right, score_right, bias_right);
            nptr->right->percentage_moved = percentage_moved_both[split_feature];
            // WARNING -- THIS ONLY WORKS IN 1 -DIMENSION!
            nptr->right->split_induced_variable=(num_groups[split_feature] - 1.0) / n_parent; // Ratio of possible splits to observations
            
            
            //std::cout << "inside vector: " << percentage_moved_both[split_feature];
            //std::cout << "inside vector: " << nptr->right->percentage_moved;
            
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
}
GBTREE::GBTREE(int maxDepth_){
    this->root = NULL;
    this->maxDepth = maxDepth_;
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

double GBTREE::getTreeBiasFull2(){
    // Recurse tree and sum evrything but leaf bias
    double treeBias = 0;
    
    node* current = this->root;
    node* pre;
    
    if(current == NULL){
        return 0;
    }
    
    while (current != NULL) { 
        
        if (current->left == NULL) { 
            //std::cout <<  current->node_prediction << std::endl; 
            //treeBias += 2*(current->bias);
            
            // Good name for fraction: split_induced_variable
            //std::cout << "percentage: " << current->percentage_moved << " , bias: " << current->bias << std::endl;
            //treeBias += 2 * (current->percentage_moved) * (current->bias);
            treeBias += 2 * (current->split_induced_variable) * (current->bias);
            
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

void node::print_child_branches(const std::string& prefix, const node* nptr, bool isLeft){
    
    if(nptr != NULL)
    {
        
        std::cout << prefix;
        
        std::cout << (isLeft ? "├──" : "└──" );
        
        // print the value of the node
        // if leaf, print prediction, else print split info.
        if(nptr->left == NULL){
            // is leaf: node prediction
            std::cout << nptr->node_prediction << std::endl;
        }else{
            // not leaf: split information
            std::cout << "(" << nptr->split_feature << ", " << nptr->split_value << ")" << std::endl;
        }
        
        // enter the next tree level - left and right branch
        print_child_branches( prefix + (isLeft ? "|   " : "    "), nptr->left, true);
        print_child_branches( prefix + (isLeft ? "|   " : "    "), nptr->right, false);
        
    }
    
    
}

void GBTREE::print_tree(){
    
    // Horizontal printing of the tree
    // Prints ( col_num , split_val ) for nodes not leaves
    // Prints node_prediction for all leaves
    
    root->print_child_branches("", root, false);
    
    
    
}

// ---------------- EXPOSING CLASSES TO R ----------
RCPP_EXPOSED_CLASS(GBTREE)
    
    RCPP_MODULE(gbtree_module){
        using namespace Rcpp;
        class_<GBTREE>("GBTREE")
            .constructor()
            .constructor<int>()
            .field("maxDepth", &GBTREE::maxDepth)
            .method("getRoot", &GBTREE::getRoot)
            .method("train", &GBTREE::train)
            .method("predict_obs", &GBTREE::predict_obs)
            .method("predict_data", &GBTREE::predict_data)
            .method("getTreeScore", &GBTREE::getTreeScore)
            .method("getTreeBiasFull", &GBTREE::getTreeBiasFull)
            .method("getTreeBiasFull2", &GBTREE::getTreeBiasFull2)
            .method("getTreeBiasFullEXM", &GBTREE::getTreeBiasFullEXM)
            .method("getTreeBias", &GBTREE::getTreeBias)
            .method("getNumLeaves", &GBTREE::getNumLeaves)
            .method("print_tree", &GBTREE::print_tree)
        ;
        // class_<node>("node")
        //     .constructor()
        //     .field("split_feature", &node::split_feature)
        //     .field("split_value", &node::split_value)
        //     .field("node_prediction", &node::node_prediction)
        //     .method("getLeft", &node::getLeft)
        //     .method("getRight", &node::getRight)
        // ;
    }


