// tree.hpp

#ifndef __TREE_HPP_INCLUDED__
#define __TREE_HPP_INCLUDED__

#include "gtbic.hpp"
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
    void train(Tvec<double> &g, Tvec<double> &h, Tmat<double> &X, int maxDepth=1);
    double predict_obs(Tvec<double> &x);
    Tvec<double> predict_data(Tmat<double> &X);
    double getTreeScore();
    double getTreeBias();
    double getTreeBiasFull();
    double getTreeBiasFullEXM();
    int getNumLeaves();
    
};

// --------------- TREE FUNCTIONS -------
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
        root = root->createLeaf(-G/H, -G*G/(2*H), C, 1.0);
        
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


#endif