// ensemble.hpp

#ifndef __ENSEMBLE_HPP_INCLUDED__
#define __ENSEMBLE_HPP_INCLUDED__


#include "tree.hpp"


// -- TRY WITHOUT EXPORT //' @export ENSEMBLE
class ENSEMBLE
{
public:
    double initialPred;
    double learning_rate;
    double initial_score;
    double extra_param; // Needed for certain distributions s.a. negative binomial, typically a dispersion param
    GBTREE* first_tree;
    Rcpp::List param;
    
    // constructors
    ENSEMBLE();
    ENSEMBLE(double learning_rate_);
    
    // Functions
    void set_param(Rcpp::List par_list);
    Rcpp::List get_param();
    double initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w);
    void train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities,
               bool force_continued_learning, Tvec<double> &w);
    void train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w);
    Tvec<double> predict(Tmat<double> &X);
    Tvec<double> predict2(Tmat<double> &X, int num_trees);
    double estimate_generalization_loss(int num_trees);
    int get_num_trees();
    Tvec<double> get_num_leaves();
    double get_extra_param();
};



#endif