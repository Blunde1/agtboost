// ensemble.hpp

#ifndef __ENSEMBLE_HPP_INCLUDED__
#define __ENSEMBLE_HPP_INCLUDED__


#include "tree.hpp"


// -- TRY WITHOUT EXPORT //' @export ENSEMBLE
class ENSEMBLE
{
public:
    int nrounds;
    double initialPred;
    double learning_rate;
    double initial_score;
    double extra_param; // Needed for certain distributions s.a. negative binomial, typically a dispersion param
    std::string loss_function;
    GBTREE* first_tree;
    //Rcpp::List param;
    
    // Constructors
    ENSEMBLE();
    
    ENSEMBLE(double learning_rate_);
    
    // Getters and setters
    void set_param(int nrounds_, double learning_rate_, double extra_param_, std::string loss_function_);
    
    int get_nrounds();
    
    double get_learning_rate();
    
    double get_extra_param();
    
    std::string get_loss_function();
    
    // Loss-related functions
    double loss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &w);
    
    Tvec<double> dloss(Tvec<double> &y, Tvec<double> &pred);
    
    Tvec<double> ddloss(Tvec<double> &y, Tvec<double> &pred);
    
    double link_function(double pred_observed);
    
    double inverse_link_function(double pred);
    
    double initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w);
    
    // Training and prediction
    void train(
            Tvec<double> &y, 
            Tmat<double> &X, 
            int verbose, 
            bool greedy_complexities,
            bool force_continued_learning, 
            Tvec<double> &w, 
            Tvec<double> &offset
        );
    
    void train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w);
    
    Tvec<double> predict(Tmat<double> &X, Tvec<double> &offset);
    
    Tvec<double> predict2(Tmat<double> &X, int num_trees);
    
    // Checks on trained model
    double estimate_training_loss(int num_trees);
    
    double estimate_optimism(int num_trees);
    
    double estimate_generalization_loss(int num_trees);
    
    int get_num_trees();
    
    Tvec<double> get_num_leaves();
    
    void serialize(ENSEMBLE *eptr, std::ofstream& f);
    
    void deSerialize(ENSEMBLE *eptr, std::ifstream& f);
    
    void save_model(std::string filepath);
    
    void load_model(std::string filepath);
    
    Tvec<double> importance(int ncols);
    
    Tvec<double> convergence(Tvec<double> &y, Tmat<double> &X);
    
    Tvec<int> get_tree_depths();
    
    double get_max_node_optimism();
    
    double get_min_hessian_weights();
};


#endif
