library(tidyverse)
library(latex2exp)
library(gridExtra)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

dmse <- function(y, y.hat) -2*sum(y-y.hat)
ddmse <- function(y, y.hat) 2*length(y)
loss_gb_sum <- function(g,h,w) sum(g*w + 0.5*h*w^2)

set.seed(12345)
B <- 100000
n <- 1000
y <- rnorm(n)
x <- sort(runif(n))
w1 <- mean(y)
tr_loss_1 <- mean((y-w1)^2)
tr_loss <- numeric(n-1)
te_loss <- numeric(n-1)
tr_R <- numeric(n-1)
te_R <- numeric(n-1)
# g, h
g <- sapply(y, dmse, y.hat=w1)
h <- sapply(y, ddmse, y.hat=w1)

pb <- txtProgressBar(min=0,max=n-1, style=3)
for(i in 1:(n-1))
{
    
    # ind
    Il <- (x <= x[i])
    
    # w_l, w_r
    Gl <- sum(g[Il])
    Gr <- sum(g[!Il])
    Hl <- sum(h[Il])
    Hr <- sum(h[!Il])
    wl <- -Gl/Hl
    wr <- -Gr/Hr
    
    # train loss
    tr_loss[i] <- -1/n * 0.5 * ( Gl*Gl/Hl + Gr*Gr/Hr )
    
    # Reduction
    tr_R[i] <- -1/n*0.5*(Gl+Gr)^2/(Hl+Hr) - tr_loss[i]
    
    # monte-carlo test loss
    te_loss_b <- numeric(B)
    te_R_b <- numeric(B)
    for(b in 1:B)
    {
        # test data
        yte <- rnorm(n)
        xte <- sort(runif(n))
        
        # derivatives
        #gte <- sapply(yte, dmse, y.hat=w1)
        #hte <- sapply(yte, ddmse, y.hat=w1)
        
        # evaluate mse
        Ilte <- (xte <= x[i])
        #score_l <- loss_gb_sum(gte[Ilte], hte[Ilte], wl)
        #score_r <- loss_gb_sum(gte[!Ilte], hte[!Ilte], wr)
        
        score_l <- sum((yte[Ilte] - (w1 + wl))^2)
        score_r <- sum((yte[!Ilte] - (w1 + wr))^2)
        te_loss_b[b] <- (score_l + score_r)/n
        
        # Reduction
        #te_R_b[b] <- -0.5/n*(sum(gte)^2)/sum(hte) - te_loss_b[b]
        te_R_b[b] <- mean((yte-w1)^2) - te_loss_b[b]
    }
    te_loss[i] <- mean(te_loss_b)
    te_R[i] <- mean(te_R_b)
    setTxtProgressBar(pb,i)
}
close(pb)

plot(tr_loss_1 + tr_loss, type="l", ylim=range(c(tr_loss_1 + tr_loss,te_loss)))
points(te_loss, type="l")

plot(tr_R, type="l", ylim=range(c(tr_R, te_R)))
points(te_R, type="l")

# optimism
delta <- 1/n
u <- delta * (1:(n-1))
plot(u,te_loss - tr_loss, type="l")
plot(u, tr_R - te_R, type="l")
eps <- 1e-9
tau <- 0.5*log(u*(1-eps)/(eps*(1-u)))
plot(tau, n/4*(te_loss - (tr_loss_1 + tr_loss)), type="l")
mean(n/4*(te_loss - (tr_loss_1 + tr_loss)))
plot(tau, n*(tr_R - te_R), type="l")
plot(tau, n/mean((g+h*0)^2)*(tr_R - te_R), type="l")
mean(n/mean(g^2)*(tr_R-te_R))

# df for plotting
df <- data.frame(u=u, tau=tau, 
                 tr_l = tr_loss_1+tr_loss,
                 te_l = te_loss,
                 tr_R = tr_R,
                 te_R = te_R,
                 l0 = 1,
                 C=4,
                 n=n)

if(F)
{
    save(df, file="results/split_to_cir_mc.RData")
    load("results/split_to_cir_mc.RData")
}


p4 <- df %>%
    ggplot() + 
    geom_line(aes(u, tr_R), colour="black", alpha=0.7, size=0.4, linetype = "solid") + 
    geom_line(aes(u, te_R), colour="#56B4E9", alpha=0.8, size=0.4, linetype="solid") +
    geom_hline(yintercept = 0, linetype="longdash", colour="#009E73", size=1) + 
    #geom_text(data=data.frame( x = 0.4, y= 0), map=aes(x=x, y=y), label = "Asymptotic value", vjust=-1) + 
    ylab("Loss reduction") + 
    xlab(TeX('$ u = \\frac{i}{n}')) + 
    #xlab(TeX('$ u = p(x_j \\leq s ) $')) + 
    ggtitle("Loss reduction profiling") +
    theme_minimal()
p4


p5 <- df %>%
    ggplot() + 
    geom_line(aes(u, n/C*(tr_R-te_R)), colour="black", alpha=0.7, size=0.4, linetype="solid") + 
    ylab("L.h. side of (53)") + 
    #ylab(TeX('$ \\frac{n}{\\hat{C}_{stump}} (R-E_0\\[R^0\\]) $')) + 
    xlab(TeX('$ u = \\frac{i}{n}')) + 
    ggtitle("Loss reduction optimism") +
    theme_minimal()
p5


p6 <- df %>%
    ggplot() +
    geom_line(aes(tau, n/C*(tr_R-te_R)), colour="black", alpha=0.7, size=0.4, linetype="solid") + 
    ylab(TeX('$ S(\\tau) $')) + 
    xlab(TeX('$ \\tau = \\frac{1}{2}\\log \\frac{u(1-\\epsilon)}{\\epsilon (1-u)} $')) + 
    #ylab(TeX('$\\alpha  x^\\alpha$, where $\\alpha \\in 1\\ldots 5$')) +
    ggtitle("Time transform to CIR") +
    #ylim(0,max(0.5*s)) +
    theme_minimal()
p6

grid.arrange(p4,p5,p6, ncol=3)


if(F)
{
    #setwd("~/Projects/Github repositories/gbtree_information/figures")
    #pdf("split_reduction_seq_to_cir.pdf", width=8, height=3.5, paper="special")
    grid.arrange(p4,p5,p6, ncol=3)
    #dev.off()
}
