
/*
 * gbtorch: Adaptive and automatic gradient boosting computations.
 * Berent Lunde
 * 07.09.2019
 */

#include "gbtorch.hpp"



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
    this->extra_param = par_list["extra_param"];
}
Rcpp::List ENSEMBLE::get_param(){
    return this->param;
}

double ENSEMBLE::initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w){
    
    double pred=0;
    int n = y.size();
    double pred_g_transform = (y*w).sum()/n; // Only initialize once, transform given link
    
    if(loss_function=="mse"){
        pred = pred_g_transform;
    }else if(loss_function=="logloss"){
        //double pred_g_transform = (y*w).sum()/n; // naive probability
        pred = log(pred_g_transform) - log(1 - pred_g_transform);
    }else if(loss_function=="poisson"){
        //double pred_g_transform = (y*w).sum()/n; // naive intensity
        pred = log(pred_g_transform);
    }else if(loss_function=="gamma::neginv"){
        //double pred_g_transform = (y*w).sum()/n;
        pred = - 1.0 / pred_g_transform;
    }else if(loss_function=="gamma::log"){
        pred = log(pred_g_transform);
    }else if(loss_function=="negbinom"){
        pred = log(pred_g_transform);
    }else if(loss_function=="poisson::zip"){
        pred = poisson_zip_start(pred_g_transform);
    }else if(loss_function=="zero_inflation"){
        // (mean(pois_pred)-mean(y))/mean(pois_pred)
        pred = zero_inflation_start(y, this);
    }
    
    return pred;
}


void ENSEMBLE::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, 
                     bool force_continued_learning, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = param["nrounds"];
    int n = y.size(); 
    //int m = X.cols();
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> pred(n), g(n), h(n);
    
    // MSE -- FIX FOR OTHER LOSS FUNCTIONS
    this->initialPred = this->initial_prediction(y, param["loss_function"], w); //y.sum()/n;
    pred.setConstant(this->initialPred);
    this->initial_score = loss(y, pred, param["loss_function"], w, this); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    // PARAMETERS FOR CIR CONTROL: Choose nsim and nobs by user
    // Default to nsim=100 nobs=100
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, param["loss_function"], this) * w;
    h = ddloss(y, pred, param["loss_function"], this) * w;
    //Rcpp::Rcout << g.array()/h.array() << std::endl;
    

    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
        //learning_rate_set * current_tree->getFeatureMapOptimism();
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
            "it: " << 1 << 
            "  |  n-leaves: " << current_tree->getNumLeaves() <<
            "  |  tr loss: " << loss(y, pred, param["loss_function"], w, this) <<
            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
             std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, param["loss_function"], this) * w;
        h = ddloss(y, pred, param["loss_function"], this) * w;
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
            //1.0*learning_rate_set * new_tree->getFeatureMapOptimism();

        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred = pred + learning_rate * (new_tree->predict_data(X));
            
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                        "  |  n-leaves: " << new_tree->getNumLeaves() << 
                        "  |  tr loss: " << loss(y, pred, param["loss_function"], w, this) <<
                        "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                        std::endl;
                
            }
        }
        
        // Stopping criteria 
        // Check for continued learning
        if(!force_continued_learning){
            
            // No forced learning
            // Check criterion
            if(expected_loss > EPS){
                break;
            }
            
        }
        
        // Passed criterion or force passed: Update ensemble
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
        
        /*
        if(expected_loss < EPS){ // && NUM_BINTREE_CONSECUTIVE < MAX_NUM_BINTREE_CONSECUTIVE){
            current_tree->next_tree = new_tree;
            current_tree = new_tree;
        }else{
            break;
        }
         */
    }
}

void ENSEMBLE::train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = param["nrounds"];
    int n = y.size(); 
    //int m = X.cols();
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> g(n), h(n);
    
    // Initial prediction
    g = dloss(y, pred, param["loss_function"], this)*w;
    h = ddloss(y, pred, param["loss_function"], this)*w;
    this->initialPred = - g.sum() / h.sum();
    pred = pred.array() + this->initialPred;
    this->initial_score = loss(y, pred, param["loss_function"], w, this); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, param["loss_function"], this)*w;
    h = ddloss(y, pred, param["loss_function"], this)*w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
    
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
                "it: " << 1 << 
                    "  |  n-leaves: " << current_tree->getNumLeaves() <<
                        "  |  tr loss: " << loss(y, pred, param["loss_function"], w, this) <<
                            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
                                std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, param["loss_function"], this)*w;
        h = ddloss(y, pred, param["loss_function"], this)*w;
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
        
        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred = pred + learning_rate * (current_tree->predict_data(X));
        
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                            "  |  n-leaves: " << current_tree->getNumLeaves() << 
                                "  |  tr loss: " << loss(y, pred, param["loss_function"], w, this) <<
                                    "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                                        std::endl;
                
            }
        }
        
        
        if(expected_loss < EPS){ // && NUM_BINTREE_CONSECUTIVE < MAX_NUM_BINTREE_CONSECUTIVE){
            current_tree->next_tree = new_tree;
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

double ENSEMBLE::estimate_generalization_loss(int num_trees){
    
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    double total_optimism = 0.0;
    double learning_rate = this->learning_rate;
    GBTREE* current = this->first_tree;
    
    if(num_trees<1){
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeOptimism();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeOptimism();
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

Tvec<double> ENSEMBLE::get_num_leaves(){
    int num_trees = this->get_num_trees();
    Tvec<double> num_leaves(num_trees);
    GBTREE* current = this->first_tree;
    for(int i=0; i<num_trees; i++){
        num_leaves[i] = current->getNumLeaves();
        current = current->next_tree;
    }
    return num_leaves;
}

double ENSEMBLE::get_extra_param(){
    return this->extra_param;
}


// ------ GBT_ZI_MIX ------------

void GBT_ZI_MIX::set_param(Rcpp::List par_list){
    this->param = par_list;
    this->learning_rate = par_list["learning_rate"];
    this->extra_param = par_list["extra_param"];
}
Rcpp::List GBT_ZI_MIX::get_param(){
    return this->param;
}

GBT_ZI_MIX::GBT_ZI_MIX(){
    this->count_conditional = NULL;
    this->zero_inflation = NULL;
}

ENSEMBLE* GBT_ZI_MIX::get_count_conditional(){
    return this->count_conditional;
}

ENSEMBLE* GBT_ZI_MIX::get_zero_inflation(){
    return this->zero_inflation;
}

double GBT_ZI_MIX::get_overdispersion(){
    return this->extra_param;
}

void GBT_ZI_MIX::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities)
{
    // 1.0 train zero-inflated count-model
    
    this->count_conditional = new ENSEMBLE;
    this->count_conditional->set_param(
            Rcpp::List::create(
                Named("learning_rate") = param["learning_rate"],
                Named("loss_function") = "poisson::zip",
                Named("nrounds") = param["nrounds"],
                Named("extra_param") = param["extra_param"]
                )
            );
    
    // Build new design matrix and vector: Needs to be done until Eigen 3.4 is in RcppEigen...
    // which are non-zero?
    int n = y.size();
    Tavec<int> ind_nz_tmp(n);
    int counter = 0;
    for(int i=0; i<n; i++){
        if(y[i] > 0){
            ind_nz_tmp[counter] = i;
            counter++;
        }
    }
    Tavec<int> ind_nz = ind_nz_tmp.head(counter);
    
    // Before Eigen 3.4 solution...
    Tmat<double> Xnz(ind_nz.size(), X.cols());
    Tvec<double> ynz(ind_nz.size());
    for(int i=0; i<counter; i++){
        ynz[i] = y[ind_nz[i]];
        Xnz.row(i) = X.row(ind_nz[i]);
    }
    
    Tvec<double> weights = Tvec<double>::Ones(counter); // This is unnecessary -- CLEANUP! --> fix ENSEMBLE->train()
    this->count_conditional->train(ynz, Xnz, verbose, greedy_complexities, false, weights);
    //this->count_conditional->train(y(ind_nz), X(ind_nz,Eigen::all), verbose, greedy_complexities, false, weights);
    
    // 2.0 Calculate training weights
    Tvec<double> lprob_weights(n);
    Tvec<double> pred_conditional = this->count_conditional->predict(X);
    double log_factorial;
    for(int i=0; i<n; i++)
    {
        log_factorial = 0;
        for(int j=0; j<y[i]; j++){ // also works when y=0-->log_factorial=0, R would have failed...
            log_factorial += log(j+1.0);
        }
        lprob_weights[i] = y[i]*pred_conditional[i] - exp(pred_conditional[i]) - log_factorial;
    }
    
    // 3.0 train mixture probability
    this->zero_inflation = new ENSEMBLE;
    this->zero_inflation->set_param(
            Rcpp::List::create(
                Named("learning_rate") = param["learning_rate"],
                Named("loss_function") = "zero_inflation",
                Named("nrounds") = param["nrounds"],
                Named("extra_param") = param["extra_param"],
                Named("preds_cond_count") = exp(pred_conditional.array()),
                Named("log_prob_weights") = lprob_weights
            )
    );
    this->zero_inflation->train(y, X, verbose, greedy_complexities, false, weights);
    
}

Tvec<double> GBT_ZI_MIX::predict(Tmat<double> &X)
{
    // Predict from conditional count and zero inflation
    // Mean prediction is (1-pi)*Poisson_preds
    Tvec<double> pred_l_lambda = this->count_conditional->predict(X);
    Tvec<double> pred_logit_prob = this->zero_inflation->predict(X);
    Tavec<double> n_prob = 1.0 - 1.0/(1.0+exp(-pred_logit_prob.array()));
    Tavec<double> lambda = exp(pred_l_lambda.array());
    return ( n_prob*lambda ).matrix();
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
        .method("train_from_preds", &ENSEMBLE::train_from_preds)
        .method("predict", &ENSEMBLE::predict)
        .method("predict2", &ENSEMBLE::predict2)
        .method("estimate_generalization_loss", &ENSEMBLE::estimate_generalization_loss)
        .method("get_num_trees", &ENSEMBLE::get_num_trees)
        .method("get_num_leaves", &ENSEMBLE::get_num_leaves)
        .method("get_extra_param", &ENSEMBLE::get_extra_param)
    ;
    
    class_<GBT_ZI_MIX>("GBT_ZI_MIX")
        .default_constructor("Default constructor")
        .method("set_param", &GBT_ZI_MIX::set_param)
        .method("get_param", &GBT_ZI_MIX::get_param)
        .method("train", &GBT_ZI_MIX::train)
        .method("predict", &GBT_ZI_MIX::predict)
        .method("get_overdispersion", &GBT_ZI_MIX::get_overdispersion)
    ;
}