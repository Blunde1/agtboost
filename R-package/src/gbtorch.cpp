
/*
 * gbtorch: Adaptive and automatic gradient boosting computations.
 * Berent Lunde
 * 07.09.2019
 */

#include "gbtorch.hpp"



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
    double initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w);
    void train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w);
    void train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w);
    Tvec<double> predict(Tmat<double> &X);
    Tvec<double> predict2(Tmat<double> &X, int num_trees);
    double estimate_generalization_loss(int num_trees);
    int get_num_trees();
};


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
    }
    
    return pred;
}


void ENSEMBLE::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = param["nrounds"];
    int n = y.size(); 
    //int m = X.size();
    double EPS = -1E-12;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> pred(n), g(n), h(n);
    
    // MSE -- FIX FOR OTHER LOSS FUNCTIONS
    this->initialPred = this->initial_prediction(y, param["loss_function"], w); //y.sum()/n;
    pred.setConstant(this->initialPred);
    this->initial_score = loss(y, pred, param["loss_function"], w); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat();
    
    // First tree
    g = dloss(y, pred, param["loss_function"]) * w;
    h = ddloss(y, pred, param["loss_function"]) * w;
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
            "  |  tr loss: " << loss(y, pred, param["loss_function"], w) <<
            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
             std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, param["loss_function"]) * w;
        h = ddloss(y, pred, param["loss_function"]) * w;
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
                        "  |  tr loss: " << loss(y, pred, param["loss_function"], w) <<
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

void ENSEMBLE::train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = param["nrounds"];
    int n = y.size(); 
    //int m = X.size();
    double EPS = -1E-12;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> g(n), h(n);
    
    // Initial prediction
    g = dloss(y, pred, param["loss_function"])*w;
    h = ddloss(y, pred, param["loss_function"])*w;
    this->initialPred = - g.sum() / h.sum();
    pred = pred.array() + this->initialPred;
    this->initial_score = loss(y, pred, param["loss_function"], w); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat();
    
    // First tree
    g = dloss(y, pred, param["loss_function"])*w;
    h = ddloss(y, pred, param["loss_function"])*w;
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
                        "  |  tr loss: " << loss(y, pred, param["loss_function"], w) <<
                            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
                                std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = dloss(y, pred, param["loss_function"])*w;
        h = ddloss(y, pred, param["loss_function"])*w;
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
                                "  |  tr loss: " << loss(y, pred, param["loss_function"], w) <<
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
    ;
}