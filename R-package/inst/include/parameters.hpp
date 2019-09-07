// Parameters

#ifndef __PARAMETERS_HPP_INCLUDED__
#define __PARAMETERS_HPP_INCLUDED__

class PARAMETERS
{
public:
    // Fields
    double learning_rate;
    std::string loss_function; // Valid: mse, logloss
    
    // Constructor
    PARAMETERS(); // default
    PARAMETERS(double learn_rte);
    PARAMETERS(std::string loss_fn);
    PARAMETERS(double learn_rte, std::string loss_fn);
    
    // Methods
    
};

PARAMETERS::PARAMETERS(){
    // Set to default vaules
    learning_rate = 0.01;
    loss_function = "mse";
}
PARAMETERS::PARAMETERS(double learn_rte){
    learning_rate = learn_rte;
    loss_function = "mse"; // DEFAULT
}
PARAMETERS::PARAMETERS(std::string loss_fn){
    learning_rate = 0.01; // Default
    loss_function = loss_fn;
}
PARAMETERS::PARAMETERS(double learn_rte, std::string loss_fn){
    learning_rate = learn_rte;
    loss_function = loss_fn;
}

#endif