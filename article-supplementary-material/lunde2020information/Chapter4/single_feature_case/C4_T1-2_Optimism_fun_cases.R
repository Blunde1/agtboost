library(RcppEigen)
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)
library(xtable)
library(latex2exp)
library(stringr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
Rcpp::sourceCpp("../tree_dim.cpp")

# Table 1 or Table 2
T1 <- TRUE
if(!T1) T2 <- TRUE

# loss functions
loss <- function(y,y.hat) sum((y-y.hat)^2) / length(y)
dloss <- function(y, y.hat) -2*sum(y-y.hat)
ddloss <- function(y, y.hat) 2*length(y)

rbbridge <- function(t_vec){
    # simulate a brownian bridge at places t in t_vec
    m <- length(t_vec)
    W <- numeric(m+1)
    W[1] <- rnorm(1, 0, sd=sqrt(t_vec[1]))
    if(m>1){
        for(i in 2:m){
            W[i] <- W[i-1] + rnorm(1, 0, sqrt(t_vec[i]-t_vec[i-1]))
        }
        
    }
    W[m+1] <- W[m] + rnorm(1, 0, sqrt(1-t_vec[m])) # at t=T=1
    B <- W[1:m] - t_vec*W[m+1] # B(t) = W(t) - t/T W(T)
    return(B)
}

emax_bb <- function(t){
    
    # t should be Uniform on 0-1, preferably sorted
    B=1000 # replicates
    mx <- numeric(B) # storing max on bridge
    
    for( i in 1:B ){
        # Simulate left
        b <- rbbridge(t)
        x <- b^2/(t*(1-t))
        mx[i] <- max(x)
    }
    
    return(mean(mx))
}

feature_to_unif <- function(x){
    
    # returns sorted x->u, minus last
    # x: numeric vector
    n <- length(x)
    x_unique <- unique(x)
    v <- length(x_unique)
    u <- numeric(v)
    
    for(i in 1:v){
        u[i] <- sum(x <= x_unique[i]) / n
    }
    
    return(sort(u)[-v])
}

# returns list of (x,y) where x has nsplits possible splits and y is of type type
fun <- function(n, type, nsplits){
    
    y <- numeric(n)
    
    n_group <- floor(n/(nsplits+1))
    xvalues <- 0:nsplits
    x <- as.vector(sapply(xvalues, function(x)rep(x,n_group))) / nsplits
    x <- c(x, rep(last(x),n-length(x)))
    
    if(type==1){
        y <- rnorm(n,0,1)    
    }else if(type==2){
        y <- rnorm(n,0,5)
    }else if(type==3){
        y <- rnorm(n, round(x), 1)
    }else if(type==4){
        y <- rnorm(n,round(x),5)
    }else if(type==5){
        y <- rnorm(n, x, 1)
    }else if(type==6){
        y <- rnorm(n,x,5)
    }
    
    res <- list(x=x, y=y)
    return(res)
}

# Simulate loss reduction type 1
set.seed(123)
B <- 1000 # Number of simulations
m <- 1000 # Number of simulations of test in each simulation
if(T1)
{
    n <- 100
    nsplits <- c(1, 9, n-1)    
}else if(T2)
{
    n <- 1000
    nsplits <- c(1, 9, 99, n-1)
    
}
type <- 1:6
typef <- c( "$y\\sim N(0,1)$", "$y\\sim N(0,5^2)$",
            "$y\\sim N(\\nint{x},1)$", "$y\\sim N(\\nint{x},5^2)$",
            "$y\\sim N(x,1)$", "$y\\sim N(x,5^2)$")
lossred_tr <- lossred_tr_adjusted <- lossred_te <- lossred_cv <- lossred_cv100 <- numeric(B)
lossred_te_tmp <- numeric(m) # number of test 
lossred_cv_tmp <- numeric(folds) # 10-fold cv
lossred_cv100_tmp <- numeric(100) # 100-fold cv
train <- adjusted <- test <- cv <- cv100 <- matrix(nrow=length(nsplits), ncol=length(type))
traindf <- adjusteddf <- testdf <- cvdf <- cv100df <- data.frame(matrix(NA, nrow = length(type)*length(nsplits), ncol = 4))

for(i in 1:length(type)){
    cat("\n type: ", type[i], ": ")
    for(j in 1:length(nsplits)){
        cat(nsplits[j], ", ")
        for(b in 1:B){
            
            # generate data
            data <- fun(n, type[i], nsplits[j])
            x <- as.matrix(data$x)
            y <- as.vector(data$y)
            
            # compute derivatives
            y0.hat <- mean(y)
            g <- sapply(y, dloss, y.hat=y0.hat)
            h <- sapply(y, ddloss, y.hat=y0.hat)
            ch0 <- mean((g+h*y0.hat)^2) / mean(h)/n
            
            # Train gbtree
            tree <- new(GBTREE)
            tree$train(g, h, x, 1)
            
            # calculate loss
            pred_tr <- y0.hat + tree$predict_data(x)
            lossred_tr[b] <- loss(y, y0.hat) - loss(y, pred_tr) 
            
            # Calculate optimism
            #tree_opt_leaf <- tree$getTreeBias()
            t <- feature_to_unif(x)
            max_cir <- emax_bb(t)
            CRt <- ch0 * max_cir
            #ch1 <- tree_opt_leaf/2 * max_double_sq_bb
            
            # adjusted lossred
            lossred_tr_adjusted[b] <- lossred_tr[b] - CRt
            
            # Test reduction in loss
            for(k in 1:m){
                testdata <- fun(n, type[i], nsplits[j])
                xte <- as.matrix(testdata$x)
                yte <- as.matrix(testdata$y)
                
                pred_te <- y0.hat + tree$predict_data(xte)
                lossred_te_tmp[k] <- loss(yte, y0.hat) - loss(yte, pred_te)
                
            }
            lossred_te[b] <- mean(lossred_te_tmp)
            
            # cv 10-fold
            # Generate indices of holdout observations
            folds <- 10
            holdout <- split(sample(1:n), 1:folds)
            
            # Train CV-gbtrees
            for(k in 1:folds){
                
                n_holdout <- length(holdout[[k]])
                
                # train
                # h0
                y0.hat.cv <- mean(y[-holdout[[k]]])
                gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
                hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
                
                # h1
                tree_k <- new(GBTREE)
                tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
                
                # predict holdout
                pred_holdout <- y0.hat.cv+tree_k$predict_data(x[holdout[[k]],,drop=F])
                
                # cv loss reduction
                lossred_cv_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], pred_holdout)
                
            }
            
            # Calculate CV-implied optimism
            lossred_cv[b] <- mean(lossred_cv_tmp)
            
            # 100-fold cv
            folds <- 100
            holdout <- split(sample(1:n), 1:folds)
            
            # Train CV-gbtrees 100-fold
            for(k in 1:folds){
                
                n_holdout <- length(holdout[[k]])
                
                # train
                # h0
                y0.hat.cv <- mean(y[-holdout[[k]]])
                gcv <- sapply(y, dloss, y.hat=y0.hat.cv)
                hcv <- sapply(y, ddloss, y.hat=y0.hat.cv)
                
                # h1
                tree_k <- new(GBTREE)
                tree_k$train( gcv[-holdout[[k]]], hcv[-holdout[[k]]], x[-holdout[[k]],,drop=F], 1 )
                
                # predict holdout
                pred_holdout <- y0.hat.cv+tree_k$predict_data(x[holdout[[k]],,drop=F])
                
                # cv loss reduction
                lossred_cv100_tmp[k] <- loss(y[holdout[[k]]], y0.hat.cv) - loss(y[holdout[[k]]], pred_holdout)
                
            }
            
            # Calculate CV-implied optimism
            lossred_cv100[b] <- mean(lossred_cv100_tmp)
            
        }
        
        #lossred_tr <- lossred_tr * n
        #lossred_tr_adjusted <- lossred_tr_adjusted * n
        #lossred_te <- lossred_te * n
        #lossred_cv <- lossred_cv * n # CHECK
        
        traindf[(length(nsplits))*(i-1)+j, ] <- c(paste0( format(mean(lossred_tr), digits = 3), " (",
                                                          format(mean(lossred_tr>0), digits=3), ")"),
                                                  "train",
                                                  typef[i], 
                                                  nsplits[j])
        adjusteddf[(length(nsplits)*(i-1)+j), ] <- c(paste0( format(mean(lossred_tr_adjusted), digits = 3), " (",
                                                             format(mean(lossred_tr_adjusted>0), digits=3), ")"),
                                                     "adjusted",
                                                     typef[i], 
                                                     nsplits[j])
        testdf[(length(nsplits)*(i-1)+j), ] <- c(paste0( format(mean(lossred_te), digits = 3), " (",
                                                         format(mean(lossred_te>0), digits=3), ")"),
                                                 "test1000",
                                                 typef[i], 
                                                 nsplits[j])
        cvdf[(length(nsplits)*(i-1)+j), ] <- c(paste0( format(mean(lossred_cv), digits = 3), " (",
                                                       format(mean(lossred_cv>0), digits=3), ")"),
                                               "10-fold cv",
                                               typef[i], 
                                               nsplits[j])
        cv100df[(length(nsplits)*(i-1)+j), ] <- c(paste0( format(mean(lossred_cv100), digits = 3), " (",
                                                          format(mean(lossred_cv100>0), digits=3), ")"),
                                                  "100-fold cv",
                                                  typef[i], 
                                                  nsplits[j])
        
    }
}

# present results in latex table
df <- rbind(traindf, adjusteddf, testdf, cvdf, cv100df)
names(df) <- c("values", "ctype", "ftype", "nsplits")
df$ctype <- factor(df$ctype)
df <- plyr::rename(df, replace = c("nsplits"="v"))
df$ftype <- as.vector(sapply(typef, function(x)rep(x, length(nsplits))))
df2 <- df
ctypes <- c("$\\mathcal{R}$", "$\\tilde{\\mathcal{R}}^0$", "$\\mathcal{R}^0$", "10-fold CV", "100-fold CV")
df$ctype <- as.vector(sapply(ctypes, function(x)rep(x, nrow(df)/5)))

if(n==100){
    save(df, file="results/loss_reduction_sim_n100_cv100_sd5.RData")
}else if(n==1000){
    save(df, file="results/loss_reduction_sim_n1000_cv100_sd5.RData")
}

textable <- function(dflong, multiply_by_n=F){
    
    # extract prob and values
    values <- prob <- character(nrow(dflong))
    for(i in 1:nrow(dflong)){
        values[i] <- strsplit(dflong$values[i], " ")[[1]][1]
        prob[i] <- strsplit(dflong$values[i], " ")[[1]][2]
    }
    if(multiply_by_n){
        values <- str_trim( as.numeric(values) * (max(as.numeric(dflong$v))+1) )
    }
    prob <- gsub("[\\(\\)]", "", regmatches(prob, gregexpr("\\(.*?\\)", prob)))
    
    # create two subtables
    dfval <- dfprob <- dflong
    dfval$values <- values
    dfprob$values <- prob
    
    # create wide
    df_wide <- spread(dflong, ftype, values)
    dfval_wide <- spread(dfval, ftype, values)
    dfprob_wide <- spread(dfprob, ftype, values)
    
    # sort
    col_map <- c(2,1, 5,6, 3,4, 7,8)
    df_wide2 <- df_wide[with(df_wide, order(v, ctype)), col_map]
    dfval_wide2 <- dfval_wide[with(dfval_wide, order(v, ctype)), col_map]
    dfprob_wide2 <- dfprob_wide[with(dfprob_wide, order(v, ctype)), col_map]
    
    # extract colnames
    colnames_main <- colnames(df_wide2)
    colnames_sub <- rep(c("$\\mathcal{R}$", "$p(\\mathcal{R}^+)$"),
                        ncol(df_wide2)-2)
    
    # print table
    nrows <- nrow(df_wide2)
    ncols <- length(colnames_sub)+2
    
    # # first part of table
    textable <- paste0( "\\begin{table}[ht] \n", 
                        "{\\small \n",
                        "\\centering \n",
                        "\\begin{tabular}{",paste0(rep("l",ncols-1),collapse = ""),"} \n",
                        sep="")
    
    textable <- paste0(textable, "\\hline \n ")
    
    # colnames
    textable <- paste0(textable, "DGP", " & ")
    for(j in 3:(length(colnames_main)-1)){
        textable <- paste0(textable, "\\multicolumn{2}{c}{", colnames_main[j],"} & ")
    }
    textable <- paste0(textable, "\\multicolumn{2}{c}{", last(colnames_main),"} \\\\ \n")
    
    # underlines
    for(j in 1:(length(colnames_main)-2)){
        textable <- paste0(textable, " \\cmidrule(lr){", 2*j,"-",2*j+1, "} ")
    }
    textable <- paste0(textable, " \n")
    
    # subcolnames
    textable <- paste0(textable, " & ", 
                       paste0( rep(c("$E$", "&", "$P$", "&"),
                                   (ncol(df_wide2)-3) ), collapse= " "))
    textable <- paste0(textable, "$E$", " & ", "$P$", " \\\\ \n")
    
    
    # underline
    textable <- paste0(textable, "\\hline \n ")
    
    # print a+1= nsplits for every new type
    ntypes <- length(unique(dflong$ctype))
    
    # # meat
    for(i in 1:nrows){
        for(j in 2:(ncols)){
            
            if( j==2 && ((i-1) %% ntypes == 0) ){
                # print a+1 = nsplits
                if(i > 1){
                    textable <- paste0(textable, "\\hline \n")
                }
                textable <- paste0(textable, 
                                   " \\multicolumn{", ncols-1,"}{c}{$a+1=",as.numeric(df_wide2$v[i])+1,"$}",
                                   " \\\\ \\hline \n")
                
                #\hline 
                #\multicolumn{13}{c}{$a+1=10$} \\\hline
            }
            
            if(j<3){
                textable <- paste0(textable, df_wide2[i,j])
            }else if( j%%2 == 0 ){
                # prob
                textable <- paste0(textable, dfprob_wide2[i,ceiling((j+2)/2)])
            }else{
                # val
                textable <- paste0(textable, dfval_wide2[i,ceiling((j+2)/2)])
            }
            
            # prob
            
            if(j<ncols){
                textable <- paste0(textable, " & ")
            }else{
                textable <- paste0(textable, " \\\\ \n")
            }
            
        }
    }
    
    # underline
    textable <- paste0(textable, "\\hline \n ")
    
    # # last part of table
    textable <- paste0(textable, 
                       "\\end{tabular} \n",
                       "} \n",
                       "\\caption{amazing table} \n",
                       "\\label{tab:amazingtab} \n",
                       "\\end{table} \n")
    
    return(textable)
    
    
}

# n=100
if(T1)
{
    load("results/loss_reduction_sim_n100_cv100_sd5.RData")   
}else if(T1)
{
    load("results/loss_reduction_sim_n1000_cv100_sd5.RData")   
}


dftmp <- plyr::rename(df, replace = c("nsplits"="v"))
tex_table <- textable(dftmp, multiply_by_n=TRUE)
cat(tex_table)




