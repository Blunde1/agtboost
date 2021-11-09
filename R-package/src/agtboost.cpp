
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
        eptr->initialPred >> eptr->initial_score >> eptr->loss_function >> std::ws;

    eptr->first_tree = new GBTREE;
    eptr->first_tree->deSerialize(eptr->first_tree, f);
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


void verbose_output(int verbose, int iteration, int nleaves, double tr_loss, double gen_loss){
    // Print output-information to user
    if(verbose>0){
        if(iteration % verbose == 0){
            Rcpp::Rcout  <<
                std::setprecision(4) <<
                    "it: " << iteration << 
                        "  |  n-leaves: " << nleaves << 
                            "  |  tr loss: " << tr_loss <<
                                "  |  gen loss: " << gen_loss << 
                                    std::endl;
        }
    }
}


// Loss functions defined in Ensemble class
double ENSEMBLE::loss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &w){
    return loss_functions::loss(y, pred, loss_function, w, extra_param);
}


Tvec<double> ENSEMBLE::dloss(Tvec<double> &y, Tvec<double> &pred){
    return loss_functions::dloss(y, pred, loss_function, extra_param);
}


Tvec<double> ENSEMBLE::ddloss(Tvec<double> &y, Tvec<double> &pred){
    return loss_functions::ddloss(y, pred, loss_function, extra_param);
}


double ENSEMBLE::link_function(double pred_observed){
    return loss_functions::link_function(pred_observed, loss_function);
}


double ENSEMBLE::inverse_link_function(double pred){
    return loss_functions::inverse_link_function(pred, loss_function);
}

                
void ENSEMBLE::train(
        Tvec<double> &y, 
        Tmat<double> &X, 
        int verbose, 
        bool greedy_complexities, 
        bool force_continued_learning, // Default: False
        Tvec<double> &w, Tvec<double> &offset // Defaults to a zero-vector
    ){
    using namespace std::placeholders;
    
    // Set initials and declare variables
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double ensemble_training_loss;
    double ensemble_approx_training_loss;
    double ensemble_optimism;
    Tvec<double> pred(n), g(n), h(n);
    Tmat<double> cir_sim = cir_sim_mat(100, 100); // nsim=100, nobs=100
    
    // Initial constant prediction: arg min l(y,constant)
    this->initialPred = learn_initial_prediction(
        y, 
        offset, 
        std::bind(&ENSEMBLE::dloss, this, _1, _2),
        std::bind(&ENSEMBLE::ddloss, this, _1, _2),
        std::bind(&ENSEMBLE::link_function, this, _1),
        std::bind(&ENSEMBLE::inverse_link_function, this, _1),
        verbose
        );
    pred.setConstant(this->initialPred);
    pred += offset;
    this->initial_score = loss_functions::loss(y, pred, loss_function, w, extra_param);
    
    // First tree
    g = dloss(y, pred) * w;
    h = ddloss(y, pred) * w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = tree_expected_test_reduction(current_tree, learning_rate);
    verbose_output(
        verbose,
        1,
        current_tree->getNumLeaves(),
        loss(y, pred, w),
        this->estimate_generalization_loss(1)
    );
    
    // Consecutive trees
    for(int i=2; i<(MAXITER+1); i++){
        // check for user-interruption at every iteration
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        // Calculate gradients
        g = dloss(y, pred) * w;
        h = ddloss(y, pred) * w;
        // Check for perfect fit
        if(((g.array())/h.array()).matrix().maxCoeff() < 1e-12){
            // Every perfect step is below tresh
            break;
        }
        // Train a new tree
        GBTREE* new_tree = new GBTREE();
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
        // Update ensemble-predictions
        pred = pred + learning_rate * (new_tree->predict_data(X));
        // Calculate expected generalization loss for tree
        expected_loss = tree_expected_test_reduction(new_tree, learning_rate);
        // Update ensemble training loss and ensemble optimism for iteration k-1
        ensemble_training_loss = loss_functions::loss(y, pred, loss_function, w, extra_param);
        ensemble_approx_training_loss = this->estimate_training_loss(i-1) + 
            new_tree->getTreeScore() * (-2)*learning_rate*(learning_rate/2 - 1);
        ensemble_optimism = this->estimate_optimism(i-1) + 
            learning_rate * new_tree->getTreeOptimism();
        // Optionally output information to user
        verbose_output(
            verbose,
            i,
            new_tree->getNumLeaves(),
            ensemble_training_loss,
            ensemble_training_loss + ensemble_optimism + expected_loss
        );
        // Stopping criteria
        if(!force_continued_learning){
            // Check criterion
            if(expected_loss > EPS){
                break;
            }
            
        }
        // Passed criterion or force passed: Update ensemble
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
        // Check for non-linearity
        if(std::abs(ensemble_training_loss-ensemble_approx_training_loss)>1E-5){
            Rcpp::stop("Error: Loss-function deviating from gradient boosting approximation. Try smaller learning_rate.");
        }
    }
}

void ENSEMBLE::train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> g(n), h(n);
    
    // Initial prediction
    g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
    h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
    this->initialPred = - g.sum() / h.sum();
    pred = pred.array() + this->initialPred;
    this->initial_score = loss_functions::loss(y, pred, loss_function, w, extra_param); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat(100, 100);
    
    // First tree
    g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
    h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
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
                        "  |  tr loss: " << loss_functions::loss(y, pred, loss_function, w, extra_param) <<
                            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
                                std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
        // check for interrupt every iterations
        if (i % 1 == 0)
            Rcpp::checkUserInterrupt();
        
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
        h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
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
                                "  |  tr loss: " << loss_functions::loss(y, pred, loss_function, w, extra_param) <<
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

Tvec<double> ENSEMBLE::predict(Tmat<double> &X, Tvec<double> &offset){
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    pred += offset;
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


double ENSEMBLE::estimate_optimism(int num_trees){
    // Return optimism approximated from 2'nd order GB loss-approximation
    // And assuming no-influence / influence adjustment
    double optimism = 0.0;
    int tree_num = 1;
    GBTREE* current = this->first_tree;
    if(num_trees<1){
        while(current != NULL){
            optimism += current->getTreeOptimism();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            optimism += current->getTreeOptimism();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    optimism = learning_rate * optimism;
    return optimism;
    
}


double ENSEMBLE::estimate_training_loss(int num_trees){
    // Return training loss approximated from 2'nd order GB loss-approximation
    double training_loss = 0.0;
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    GBTREE* current = this->first_tree;
    if(num_trees<1){
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    training_loss = 
        this->initial_score + 
        total_observed_reduction * 
        (-2)*learning_rate*(learning_rate/2 - 1);
    return training_loss;
}


double ENSEMBLE::estimate_generalization_loss(int num_trees){
    
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    double total_optimism = 0.0;
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
    loss_val[0] = loss_functions::loss(y, pred, this->loss_function, w, extra_param);
    
    GBTREE* current = this->first_tree;
    for(int k=1; k<(K+1); k++)
    {
        // Update predictions with k'th tree
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        
        // Compute loss
        loss_val[k] = loss_functions::loss(y, pred, this->loss_function, w, extra_param);
        
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

Tvec<int> ENSEMBLE::get_tree_depths(){
    // Return vector of ints with individual tree-depths
    int number_of_trees = this->get_num_trees();
    Tvec<int> tree_depths(number_of_trees);
    GBTREE* current = this->first_tree;
    
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get tree depth
        tree_depths[i] = current->get_tree_depth();
        // Update to next tree
        current = current->next_tree;
    }
    return tree_depths;
}

double ENSEMBLE::get_max_node_optimism(){
    // Return the minimum loss-reduction in ensemble
    double max_node_optimism = 0.0;
    double tree_max_node_optimism;
    int number_of_trees = this->get_num_trees();
    GBTREE* current = this->first_tree;
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get minimum loss reduction in tree
        tree_max_node_optimism = current->get_tree_max_optimism();
        if(tree_max_node_optimism > max_node_optimism){
            max_node_optimism = tree_max_node_optimism;
        }
        // Update to next tree
        current = current->next_tree;
    }
    return max_node_optimism;
}

double ENSEMBLE::get_min_hessian_weights(){
    double min_hess_weight = R_PosInf;
    double tree_min_hess_weight;
    int number_of_trees = this->get_num_trees();
    GBTREE* current = this->first_tree;
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get minimum loss reduction in tree
        tree_min_hess_weight = current->get_tree_min_hess_sum();
        if(tree_min_hess_weight < min_hess_weight){
            min_hess_weight = tree_min_hess_weight;
        }
        // Update to next tree
        current = current->next_tree;
    }
    return min_hess_weight;
}

// --- GBT_COUNT_AUTO ----
void GBT_COUNT_AUTO::set_param(Rcpp::List par_list){
    this->param = par_list;
    this->learning_rate = par_list["learning_rate"];
    this->extra_param = par_list["extra_param"]; // Starting value
}
Rcpp::List GBT_COUNT_AUTO::get_param(){
    return this->param;
}
GBT_COUNT_AUTO::GBT_COUNT_AUTO(){
    this->count_mod = NULL;
}
ENSEMBLE* GBT_COUNT_AUTO::get_count_mod(){
    return this->count_mod;
}

double GBT_COUNT_AUTO::get_overdispersion(){
    return this->count_mod->get_extra_param();
}

std::string GBT_COUNT_AUTO::get_model_name(){
    std::string count_loss = this->count_mod->get_loss_function();
    if(count_loss == "poisson"){
        return "poisson";
    }else if(count_loss == "negbinom"){
        return "negbinom";
    }else{
        return "unknown";
    }
}

void GBT_COUNT_AUTO::train(Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities)
{
    /*
     * 1. Train Poisson model
     * 2. Learn overdispersion
     * 2.1 If overdispersion large, return Poisson model
     * 3. Train negbinom model
     * 4. Compare relative AIC of models
     * 5. Return and set count model as model with best AIC
     */
    
    
    
    
    // Variables
    double log_factorial;
    double MAX_DISPERSION = 1e9;
    int n =y.size();
    
    // --- 1.0 Poisson ---
    ENSEMBLE* mod_pois = new ENSEMBLE;
    mod_pois->set_param(param["nrounds"], param["learning_rate"], param["extra_param"], "poisson");
    /*
    mod_pois->set_param(
            Rcpp::List::create(
                Named("learning_rate") = param["learning_rate"],
                                              Named("loss_function") = "poisson",
                                              Named("nrounds") = param["nrounds"],
                                                                      Named("extra_param") = param["extra_param"]
            )
    );
    */
    // Training
    Tvec<double> weights = Tvec<double>::Ones(n); // This is unnecessary -- CLEANUP! --> fix ENSEMBLE->train()
    Tvec<double> offset = Tvec<double>::Zero(n);
    mod_pois->train(y, X, verbose, greedy_complexities, false, weights, offset);

    // ---- 2.0 Learn overdispersion ----
    // Predictions on ynz
    Tvec<double> y_pred_pois = mod_pois->predict(X, offset); // log intensity  
    double dispersion = learn_dispersion(y, y_pred_pois);
    
    // ---- 2.1 Check dispersion -----
    if(dispersion<MAX_DISPERSION)
    {
        // --- 3.1 Train negbinom ----
        ENSEMBLE* mod_nbinom = new ENSEMBLE;
        mod_nbinom->set_param(param["nrounds"], param["learning_rate"], dispersion, "negbinom");
        /*
        mod_nbinom->set_param(
                Rcpp::List::create(
                    Named("learning_rate") = param["learning_rate"],
                                                  Named("loss_function") = "negbinom",
                                                  Named("nrounds") = param["nrounds"],
                                                                          Named("extra_param") = dispersion
                )
        );
         */
        mod_nbinom->train(y, X, verbose, greedy_complexities, false, weights, offset);
        
        // ---- 4. Compare relative AIC of models ----
        Tvec<double> y_pred_nbinom = mod_nbinom->predict(X, offset); // log mean
        dispersion = learn_dispersion(y, y_pred_nbinom, dispersion);
        mod_nbinom->extra_param = dispersion;
        
        // Needs to compare on full likelihood!
        double nll_pois=0.0, nll_nbinom=0.0;
        for(int i=0; i<y.size(); i++)
        {
            // poisson
            log_factorial = 0;
            for(int j=0; j<y[i]; j++){ // also works when y=0-->log_factorial=0, R would have failed...
                log_factorial += log(j+1.0);
            }
            nll_pois -= y[i]*y_pred_pois[i] - exp(y_pred_pois[i]) - log_factorial;
            
            // negative binomial
            nll_nbinom += y[i]*log(dispersion) - y[i]*y_pred_nbinom[i] + 
                (y[i]+dispersion)*log(1.0+exp(y_pred_nbinom[i])/dispersion) - 
                R::lgammafn(y[i]+dispersion) + R::lgammafn(y[i]+1.0) + R::lgammafn(dispersion);
        }
        
        double poisson_aic = nll_pois / y.size();
        double nbinom_aic = (nll_nbinom + 1.0) / y.size();
        
        Rcpp::Rcout << "Relative AIC Poisson: " << poisson_aic << "\n" << 
            "Relative AIC nbinom: " << nbinom_aic << std::endl;
        if(poisson_aic <= nbinom_aic){
            Rcpp::Rcout << "Choosing Poisson model " << std::endl;
            this->count_mod = mod_pois;
        }else{
            Rcpp::Rcout << "Choosing nbinom model " << std::endl;
            this->count_mod = mod_nbinom;
        }
        
    }else{
        // Return with Poisson
        Rcpp::Rcout << "Dispersion too high: Choosing Poisson model " << std::endl;
        this->count_mod = mod_pois;
    }
    
}

Tvec<double> GBT_COUNT_AUTO::predict(Tmat<double> &X)
{
    int n = X.rows();
    Tvec<double> offset = Tvec<double>::Zero(n);
    return this->count_mod->predict(X, offset);
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
        .method("train_from_preds", &ENSEMBLE::train_from_preds)
        .method("predict", &ENSEMBLE::predict)
        .method("predict2", &ENSEMBLE::predict2)
        .method("estimate_generalization_loss", &ENSEMBLE::estimate_generalization_loss)
        .method("get_num_trees", &ENSEMBLE::get_num_trees)
        .method("get_num_leaves", &ENSEMBLE::get_num_leaves)
        .method("save_model", &ENSEMBLE::save_model)
        .method("load_model", &ENSEMBLE::load_model)
        .method("importance", &ENSEMBLE::importance)
        .method("convergence", &ENSEMBLE::convergence)
        // get for complexity methods
        .method("get_tree_depths", &ENSEMBLE::get_tree_depths)
        .method("get_max_node_optimism", &ENSEMBLE::get_max_node_optimism)
        .method("get_min_hessian_weights", &ENSEMBLE::get_min_hessian_weights)
    ;
    
    class_<GBT_COUNT_AUTO>("GBT_COUNT_AUTO")
        .default_constructor("Default constructor")
        .method("set_param", &GBT_COUNT_AUTO::set_param)
        .method("get_param", &GBT_COUNT_AUTO::get_param)
        .method("train", &GBT_COUNT_AUTO::train)
        .method("predict", &GBT_COUNT_AUTO::predict)
        .method("get_overdispersion", &GBT_COUNT_AUTO::get_overdispersion)
        .method("get_model_name", &GBT_COUNT_AUTO::get_model_name)
    ;
}
