# Appendix figures
library(ggplot2)

# Simulation
dmse <- function(y, y.hat) -2*sum(y-y.hat)
ddmse <- function(y, y.hat) 2*length(y)
nsim <- 10000 # number of simulations
n <- 100 # number of obs at each simulation
u <- 0.3
psil <- psir <- wli <- wri <- numeric(nsim)
for(i in 1:nsim)
{
    y <- rnorm(n, 3, 1)
    x <- runif(n, 0,1)
    Il <- x<=u
    y0 <- 2 # some initial value != mean(y), to make it difficult
    g <- sapply(y, dmse, y.hat=y0)
    h <- sapply(y, ddmse, y.hat=y0)
    wt <- -sum(g)/sum(h)
    gt <- g+h*wt
    ht <- h
    w0 <- 0 # under H0
    wli[i] <- -sum(gt[Il]) / sum(ht[Il])
    wri[i] <- -sum(gt[!Il]) / sum(ht[!Il])
    psil[i] <- sum(gt[Il]+ht[Il]*w0) / n
    psir[i] <- sum(gt[!Il]+ht[!Il]*w0) / n
}

# Score plots
p1 <- ggplot(data=data.frame(psi_l=psil), aes(x=psi_l)) + 
    geom_histogram(aes(y=..density..), colour="black", fill="cornflowerblue") + 
    stat_function(fun=dnorm, args=list(mean=0, sd=sqrt(4*u*(1-u)/n)), colour="orange", size=1)+
    xlab("Left node value") +
    #ggtitle("Score in left node") + 
    theme_bw()
p2 <- ggplot(data=data.frame(psi_r=psir), aes(x=psi_r)) + 
    geom_histogram(aes(y=..density..), colour="black", fill="cornflowerblue") + 
    stat_function(fun=dnorm, args=list(mean=0, sd=sqrt(4*u*(1-u)/n)), colour="orange", size=1)+
    xlab("Right node value") +
    #ggtitle("Score in right node") +
    theme_bw()
p12 <- ggplot(data=data.frame(psi_l=psil, psi_r=psir), aes(x=psi_l, y=psi_r)) + 
    geom_point() +
    xlab("Left node value") + 
    ylab("Right node value") + 
    theme_bw()
gridExtra::grid.arrange(p1,p2, p12,
                        ncol=3, 
                        top=grid::textGrob("Asymptotic normality of modified score in left and right node",
                                           gp=grid::gpar(fontsize=14,font=1)))

# Estimator plots
p3 <- ggplot(data=data.frame(wl=wli), aes(x=wl)) + 
    geom_histogram(aes(y=..density..), colour="black", fill="cornflowerblue") + 
    stat_function(fun=dnorm, args=list(mean=0, sd=sqrt(4*(1-u)/n/(u*2^2))), colour="orange", size=1)+
    xlab("Left estimator") +
    theme_bw()
p4 <- ggplot(data=data.frame(wr=wri), aes(x=wr)) + 
    geom_histogram(aes(y=..density..), colour="black", fill="cornflowerblue") + 
    stat_function(fun=dnorm, args=list(mean=0, sd=sqrt(4*u/n/((1-u)*2^2))), colour="orange", size=1)+
    xlab("Right estimator") +
    theme_bw()
dependence_line <- function(x, u)
{
    # x = wl, output wr
    multiplier <- -u/(1-u)
    multiplier*x
}
p34 <- ggplot(data=data.frame(wl=wli, wr=wri), aes(x=wl, y=wr)) + 
    geom_point() + 
    xlab("Left estimator") + 
    ylab("Right estimator") + 
    stat_function(fun=dependence_line, args = list(u=u), colour="orange") + 
    theme_bw()
gridExtra::grid.arrange(p3,p4, p34,
                        ncol=3, 
                        top=grid::textGrob("Asymptotic normality of modified estimator in left and right node",
                                           gp=grid::gpar(fontsize=14,font=1)))

# Quadratics plot
Ct <- 4 / 2/n
quad_terms_scaled <- ( 2*u*wli^2 + 2*(1-u)*wri^2 ) / Ct
p5 <- ggplot(data=data.frame(quad_terms_scaled=quad_terms_scaled), aes(x=quad_terms_scaled)) + 
    geom_histogram(aes(y = ..density..), colour = "black", fill = "cornflowerblue", 
                   breaks=seq(0,10,length.out=30)) +
    stat_function(fun=dchisq, args=list(df=1), colour="orange",size=1) + 
    ggtitle("Comparison of the summed quadratic terms to the Chi-square") +
    xlab("Value") + 
    xlim(0,7.5) + 
    #ylim(0,1.41) +
    theme_bw()
p5

# Saving plots
#pdf("appendix_score_dist.pdf", width=8, height=4, paper="special")
gridExtra::grid.arrange(p1,p2, p12,
                        ncol=3, 
                        top=grid::textGrob("Asymptotic normality of modified score in left and right node",
                                           gp=grid::gpar(fontsize=14,font=1)))
#dev.off()

#pdf("appendix_estimator_dist.pdf", width=8, height=4, paper="special")
gridExtra::grid.arrange(p3,p4, p34,
                        ncol=3, 
                        top=grid::textGrob("Asymptotic normality of modified estimator in left and right node",
                                           gp=grid::gpar(fontsize=14,font=1)))
#dev.off()

#pdf("appendix_quadratics_dist.pdf", width=6, height=5, paper="special")
p5
#dev.off()
