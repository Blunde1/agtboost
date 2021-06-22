/*
 * agtboost: Adaptive and automatic gradient tree boosting computations.
 * Berent Lunde
 * 07.09.2019
 */

#include "agtboost.hpp"


// ---------------- ENSEMBLE ----------------
ENSEMBLE::ENSEMBLE(){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->learning_rate=0.01;
    this->extra_param = 0.0;
    this->loss_function = "mse";
}

ENSEMBLE::ENSEMBLE(double learning_rate_){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->learning_rate=learning_rate_;
    this->extra_param = 0.0;
    this->loss_function = "mse";
}

void ENSEMBLE::set_param(int nrounds_, double learning_rate_, double extra_param_, std::string loss_function_)
{
    this->nrounds = nrounds_;
    this->learning_rate = learning_rate_;
    this->extra_param = extra_param_;
    this->loss_function = loss_function_;
}

int ENSEMBLE::get_nrounds(){
    return this->nrounds;
}

double ENSEMBLE::get_learning_rate(){
    return this->learning_rate;
}

double ENSEMBLE::get_extra_param(){
    return this->extra_param;
}

std::string ENSEMBLE::get_loss_function(){
    return this->loss_function;
}

void ENSEMBLE::serialize(ENSEMBLE *eptr, std::ofstream& f)
{
    // If current ENSEMBLE is NULL, return
    if(eptr == NULL)
    {
        //Rcpp::Rcout << "Trying to save NULL pointer" << std::endl;
        return;
    }
    
    f << std::fixed << eptr->nrounds << "\n";
    f << std::fixed << eptr->learning_rate << "\n";
    f << std::fixed << eptr->extra_param << "\n";
    f << std::fixed << eptr->initialPred << "\n";
    f << std::fixed << eptr->initial_score << "\n";
    f << eptr->loss_function << "\n";
    
    eptr->first_tree->serialize(eptr->first_tree, f);
    f.close();
    
}

void ENSEMBLE::deSerialize(ENSEMBLE *eptr, std::ifstream& f)
{
    
    // Check stream
    std::streampos oldpos = f.tellg();
    int val;
    int MARKER = -1;
    if( !(f >> val) || val==MARKER ){
        return;   
    }
    f.seekg(oldpos);
    
    // Read from stream
    f >> eptr->nrounds >> eptr->learning_rate >> eptr->extra_param >>
        eptr->initialPred >> eptr->initial_score >> eptr->loss_function;
    
    // Start recurrence
    int lineNum = 6;
    eptr->first_tree = new GBTREE;
    eptr->first_tree->deSerialize(eptr->first_tree, f, lineNum);
    
}

void ENSEMBLE::save_model(std::string filepath)
{
    std::ofstream f;
    f.open(filepath.c_str());
    this->serialize(this, f);
    f.close();
}
void ENSEMBLE::load_model(std::string filepath)
{
    std::ifstream f;
    f.open(filepath.c_str());
    this->deSerialize(this, f);
    f.close();
}

 
double ENSEMBLE::initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w){
    
    double pred=0;
    double pred_g_transform = y.sum()/w.sum(); // should be optim given weights...
    
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
    }
    
    return pred;
}

void ensemble_influence_update(Tvec<double> &ensemble_influence, Tvec<double> &residual_influence, Tvec<double>& ensemble_influence_fraction, double learning_rate){
    int n = ensemble_influence.size();
    for(int i=0; i<n; i++){
        ensemble_influence[i] += learning_rate*ensemble_influence_fraction[i]*residual_influence[i];    
    }
}

void ensembleInfluenceFractionUpdate(Tvec<double> &ensemble_influence_fraction, Tvec<double> &residual_influence, 
                                     Tvec<double> &g, Tvec<double> &h, Tvec<double> &tree_pred, double learning_rate){
    int n = ensemble_influence_fraction.size();
    double residual;
    for(int i=0; i<n; i++){
        residual = -g[i]/h[i] - tree_pred[i];
        if(residual!=0){
            // Bound: relative self-influence is bounded by 1
            ensemble_influence_fraction[i] *= std::max(1.0-learning_rate, residual / (residual + learning_rate*residual_influence[i]));
        }
        //Rcpp::Rcout<< "R: " << residual << " | Infl: " << residual_influence[i] << " | Scaling: " << (residual + learning_rate*residual_influence[i]) / residual << std::endl;
    }
}

Tvec<double> ensembleInfluenceFraction(Tvec<double> &y, Tvec<double> &ensemble_influence, double y0){
    // Write as update function
    int n = y.size();
    double residual_original;
    Tvec<double> ensemble_influence_fraction(n);
    for(int i=0; i<n; i++){
        //Rcpp::Rcout<< "Y: " << y[i] << " - R: "<< y[i] - y0 << " - Infl: " << ensemble_influence[i] << std::endl;
        residual_original = y[i] - y0;
        ensemble_influence_fraction[i] = (residual_original + ensemble_influence[i]) / residual_original; 
    }
    return ensemble_influence_fraction;
}

// double gbtree_criterion(Tvec<double> &ensemble_influence, ){
//     // score * influence
//     double gbtree_optimism = 0.0;
//     return gbtree_optimism;
// }


void ENSEMBLE::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, 
                     bool force_continued_learning, Tvec<double> &w, bool influence_adjustment){
    // Set init -- mean
    influence_adjustment = false;
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> pred(n), pred_tree(n), g(n), h(n), ensemble_influence(n), ensemble_influence_fraction(n), residual_influence(n);
    //Tvec<double> pred_cvn(n)
    //pred_cvn.setZero();
    ensemble_influence.setZero();
    ensemble_influence_fraction.setOnes();
    
    // Initial unscaled prediction, f^{(0)}
    this->initialPred = this->initial_prediction(y, loss_function, w); //y.sum()/n;
    pred.setConstant(this->initialPred);
    this->initial_score = loss(y, pred, loss_function, w, this); //(y - pred).squaredNorm() / n;
    //pred_cvn.setConstant(this->initialPred);
    
    // Prepare cir matrix
    // PARAMETERS FOR CIR CONTROL: Choose nsim and nobs by user
    // Default to nsim=100 nobs=100
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = dloss(y, pred, loss_function, this) * w;
    h = ddloss(y, pred, loss_function, this) * w;

    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, ensemble_influence_fraction, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred_tree = current_tree->predict_data(X);
    pred = pred + learning_rate * pred_tree; // POSSIBLY SCALED
    //pred_cvn = pred_cvn + learning_rate * (current_tree->predict_data_cvn(g, h, X));
    //pred_trees = pred_trees + learning_rate * (current_tree->predict_data(X));
    residual_influence = current_tree->residualInfluence(g,h,X);
    ensemble_influence_update(ensemble_influence, residual_influence, ensemble_influence_fraction, learning_rate);
    ensembleInfluenceFractionUpdate(ensemble_influence_fraction, residual_influence, g, h, pred_tree, learning_rate);
    //ensemble_influence_fraction = ensembleInfluenceFraction(y, ensemble_influence, this->initialPred);
    
    // Print
    //Rcpp::Rcout << "pred: " << pred << std::endl;
    //Rcpp::Rcout << "cvn-pred: " << pred_cvn << std::endl;
    
    //pred_trees = pred_trees + learning_rate * (current_tree->predict_data(X));
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
        //learning_rate_set * current_tree->getFeatureMapOptimism();
    if(verbose>0){
        Rcpp::Rcout  <<
            std::setprecision(4) <<
            "it: " << 1 << 
            "  |  n-leaves: " << current_tree->getNumLeaves() <<
            "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
            "  |  ensemble-influence: " << ensemble_influence_fraction.sum()/n << 
            "  |  residual-influence: " << (current_tree->residualInfluence(g,h,X)).array().abs().sum()/n <<
             std::endl;
    }
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        if(influence_adjustment){
            //g = dloss(y, pred_cvn, loss_function, this) * w;
            //h = ddloss(y, pred_cvn, loss_function, this) * w;
        }else{
            g = dloss(y, pred, loss_function, this) * w;
            h = ddloss(y, pred, loss_function, this) * w;
        }
        
        // Check perfect fit
        if(((g.array())/h.array()).matrix().maxCoeff() < EPS){
            // Every perfect step is below tresh
            break;
        }
        
        
        new_tree->train(g, h, X, cir_sim, ensemble_influence_fraction, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
            //1.0*learning_rate_set * new_tree->getFeatureMapOptimism();

        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred_tree = new_tree->predict_data(X);
        pred = pred + learning_rate * pred_tree;
        //pred_cvn = pred_cvn + learning_rate* (new_tree->predict_data_cvn(g, h, X));
        //pred_trees = pred_trees + learning_rate * (new_tree->predict_data(X));
            
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                Rcpp::Rcout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                        "  |  n-leaves: " << new_tree->getNumLeaves() << 
                        "  |  tr loss: " << loss(y, pred, loss_function, w, this) <<
                        "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                        "  |  ensemble-influence: " << ensemble_influence_fraction.array().abs().sum()/n << 
                        "  |  residual-influence: " << (new_tree->residualInfluence(g,h,X)).array().abs().sum()/n <<
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
        
        // Update influence
        residual_influence = new_tree->residualInfluence(g,h,X);
        ensemble_influence_update(ensemble_influence, residual_influence, ensemble_influence_fraction, learning_rate);
        ensembleInfluenceFractionUpdate(ensemble_influence_fraction, residual_influence, g, h, pred_tree, learning_rate);
        //ensemble_influence_fraction = ensembleInfluenceFraction(y, ensemble_influence, this->initialPred);
    }
}

Tvec<double> ENSEMBLE::importance(int ncols)
{
    // Vector with importance
    Tvec<double> importance_vector(ncols);
    importance_vector.setZero();
    
    // Go through each tree to fill importance vector
    GBTREE* current = this->first_tree;
    while(current != NULL)
    {
        current->importance(importance_vector, this->learning_rate);
        current = current->next_tree;
    }
    
    // Scale and return percentwise
    Tvec<double> importance_vec_percent = importance_vector.array()/importance_vector.sum();
    
    return importance_vec_percent;
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

Tvec<double> ENSEMBLE::convergence(Tvec<double> &y, Tmat<double> &X){
    
    // Number of trees
    int K = this->get_num_trees();
    Tvec<double> loss_val(K+1);
    loss_val.setZero();
    
    // Prepare prediction vector
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    
    // Unit weights
    Tvec<double> w(n);
    w.setOnes();
    
    // After each update (tree), compute loss
    loss_val[0] = loss(y, pred, this->loss_function, w, this);
    
    GBTREE* current = this->first_tree;
    for(int k=1; k<(K+1); k++)
    {
        // Update predictions with k'th tree
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        
        // Compute loss
        loss_val[k] = loss(y, pred, this->loss_function, w, this);
        
        // Update to next tree
        current = current->next_tree;
        
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
    }
    
    return loss_val;
}


// Expose the classes
RCPP_MODULE(aGTBModule) {
    using namespace Rcpp;
    
    class_<ENSEMBLE>("ENSEMBLE")
        .default_constructor("Default constructor")
        .constructor<double>()
        .field("initialPred", &ENSEMBLE::initialPred)
        .method("set_param", &ENSEMBLE::set_param)
        .method("get_nrounds", &ENSEMBLE::get_nrounds)
        .method("get_learning_rate", &ENSEMBLE::get_learning_rate)
        .method("get_extra_param", &ENSEMBLE::get_extra_param)
        .method("get_loss_function", &ENSEMBLE::get_loss_function)
        .method("train", &ENSEMBLE::train)
        //.method("train_from_preds", &ENSEMBLE::train_from_preds)
        .method("predict", &ENSEMBLE::predict)
        .method("predict2", &ENSEMBLE::predict2)
        .method("estimate_generalization_loss", &ENSEMBLE::estimate_generalization_loss)
        .method("get_num_trees", &ENSEMBLE::get_num_trees)
        .method("get_num_leaves", &ENSEMBLE::get_num_leaves)
        .method("save_model", &ENSEMBLE::save_model)
        .method("load_model", &ENSEMBLE::load_model)
        .method("importance", &ENSEMBLE::importance)
        .method("convergence", &ENSEMBLE::convergence)
    ;
}