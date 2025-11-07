###TOPICS

# done	HW	Topic
# x	1	Response vs explanatory
# x		Nominal vs ordinal
# x		Bernoulli
# x		Binomial distribution
# x		Negative binomial distribution (and in R)
# x		Using binomial in R
# x		Building binomial based on Bernoulli
# x		Normal approximation to Binomial
# x		Using 0/1 for two categories
# x		Sigma = sqrt(p(1-p))
# x		Link between binomial mean and variance
# x		Binomial likelihood function
# x		Score is the derivative of a log likelihood
# x		Solving a likelihood for the MLE
# x		HT for binomial data
# x		CI for binomial data
# x		Z=(x-mu)/sqrt(npi(1-pi))
# x		Z=(p-pi)/sqrt(pi(1-pi)/n)
# x		prop.test(x,n,p,alternative)
# x		library(binom)
# x		binom::binom.confint(x,n,method=c("asymptotic","wilson","agresti-coull")
#                         x		difference between Wald, Agresti, and Wilson
#                         x		Wald (asymptotic): p+-Zsqrt(p(1-p)/n)
#                         x		Wilson (Score) (phat+z^2/(2*n)+-z*sqrt(phat*(1-phat)/n+z^2/(4*n^2)))/(1+z^2/n)
#                         x		Agresti (Add 2 fails/successes): n*=n+z^2, p*<-1/n* (x+z^2/2), p*+z*sqrt(p* (1-p*)/n*)
#                         x		Jeffrey (Bayes with Beta(.5,.5))) uses (x + 0.5)/(n + 1) 
# x		Jeffrey uses HPD interval Beta(x+a,n-x+b)
# x		Wald uses phat in denominator, score uses pi
# x		Goodness of fit tests H0: muij=muhatij
# x		Chisq=(Obs-Exp)^2/Exp
# x		Binomial exact p-values vs mid p-values
# x		Mid p-values as a more conservative estimate
# x		Exactci library masks function binom.exact 
# x		sensitivity and specificity
# x		Poisson distribution
# x		Using poisson in R
# x		Method of moments
# x		likelihood for poisson
# x		Link between poisson mean and variance
# x		Zero inflated poisson
# x		Wald Poisson CI: ybar +- Z sqrt(Ybar/n)
# x		Score Poisson CI: ybar+Z^2/2n+Z/sqrt(n) sqrt(ybar+Z^2/4n)
# x		GOF for poisson: Z={s^2/xbar)-1}sqrt((n-1)/2)
# x	2	Bayes Rule
# x		Conditional Probabilities
# x		Difference in proportions
# x		Relative Risk
# x		Case/Control can only do relative risk in one direction
# x		pi1/pi2
# x		Odds
# x		wins/loses
# x		Odds ratio
# x		Odds Ratio: [p1/(1-p1)]/[p2/(1-p2)]
# x		Difference in Odds
# x		Calculating probability from odds
# x		CI for odds ratio
# x		Chi-squared Goodness of fit
# -		LR statistic G=2sumnlog(n/mu)
# x		Maximizing likelihood in R
# -		uniroot command to solve in R
# x		Likelihood of multiple random events
# x		Wald interval for diff in props
# x		Newcombe interval for diff in props
# x		Agresti-Caffo interval for diff in props
# x	3	Chisquared test of independence
# x		E=R*C/T
# x		std res = (O-E)/SQRT(E*(1-R/T)*(1-C/T))
# x		chisq.test(data)$stdres
# x		Chisq =SUM (O-E)^2/E
# x		mosaicplot(data)
# x		linear trend test
# x		R code for linear trend test
# x		Fisher's Exact Test
# x		fisher.test(data,alternative="less")
# x		Mid pvalues 
# x		epitools::ormidp.test(x1,x2,x3,x4,or=1)
# x		epitools::or.midp(data=c(x1,x2,x3,x4))
# x	4	Simpson's Paradox
# x		oddsratio(data)
# x		Conditional odds ratios
# x		Marginal odds ratios
# x		Breslow Day test of equal odds
# x		Tests Homogeneous association
# x		DescTools::BreslowDayTest(data)
# x		Cochran-Mantel-Haenszel test of odds all equal 1
# x		Tests conditional indpendence
# x		mantelhaen.test(data,correct=FALSE)
# x		Array command to read in data
# x		array(data,dim=c(row,cols,depth))
# x		Odds ratios across a subset of data
# x		oddsratio(data[,,1])
# x		Apply command for 3 way matrices
# x		fitting linear probability models
# x		glm(cbind(y,n)~x,family=binomial(link="identity"))
# x		glm(y~x,family=binomial(link="identity"))
# x		logistic regression
# x		GLM code for logistic regression
# x		Logistic regression with 0/1
# x		Logistic regression with y/n
# x		glm(y~x,family=binomial(link="logit"))
# x		glm(y/n~x,family=binomial(link="logit"),weight=n)
# x		glm(y~x,family=binomial(link="probit"))
# x		logit into probabilities
# x		predict.glm(fit,data.frame(x),type="response")
# x	5	Poisson Regression
# x		glm(y~x,family=poisson(link=log))
# x		when x is 0/1 beta is the difference
# x		Describes a rate of occurance
# x		dummy variables
# x		log difference is going to be quotient
# x		As x increases by 1 y increases by exp(beta) as a percent
# x		Wald test for poisson regression
# x		Likelihood ratio test for poisson regression
# x		anova(fit) for individual items
# x		anova(fit1,fit2) for nested models
# x		car::Anova
# x		confint(fit)
# x		predict.glm(fit,data.frame(x),type="response")
# x		Wald CI for poisson theta: y/n+-Zsqrt(y/n*(1-y/n)/n)
# x		comparing two models against each other
# x		lmtest:lrtest
# x		ANOVA for testing categorical terms in a model
# x		library(MASS);glm.nb(y~x,data=data)
# x		Zero inflated models
# x		pscl::zeroinfl(y~x|zero terms,data=data,dist=c("poisson","negbin")
#x		AIC(fit)
#x		MASS::glm.nb(y~x+offset(z))
#x		Negative binomial allows larger variance than poisson
#                   x		dispersion parameter theta measures how much more variance
#                   x		var=mu+mu^2/theta   
#                   x		In R theta very large if poisson
#                   x		CI for 1/theta would want zero in the interval
#                   x		fit$theta and fit$SE.theta from glm.nb
#                   x		Poisson regression with all categorical predictors
#                   x		Log-linear modeling
#                   x		offset to describe rate per numerical count
#                   x	6	rate of change for p in logistic regression
#                   x		sensitivity and specificity
#                   x		predict.glm(fit,data.frame(x),type="response",se.fit=TRUE)
#                   x		confint(fitlogistic) for likelihood CI
#                   x		confint.default(fitlogistic) for wald CI
#                   x		d/dx = pi*(1-pi)B1
#                   x		Marginal effect of B1
#                   x		E(dpi/dx)
#                   x		1/n*sum(pi(1-pi)B1)
#                   x		mfx::logitmfx(fit,atmean=FALSE,data=data.frame(x,y))
#                   x		logitmfx(x,atmean=FALSE,data=data) use the data predictions, not xhat
#                   x	7	pROC::roc(data$Y~predict(fit,type="response")
#                                 x		ROC means Recieving Operator Characteristic
#                                 x		pROC::roc(data$Y~predict(fit,type="response",plot=TRUE))
#                                 x		fitted(fit)
#                                 x		Classfication vs confusion table
#                                 x		table(data$y,fitted(fit)>0.5)
#                                 x		roc(data$y~predict(fit,type="response"),plot=TRUE,smooth=TRUE)
#                                 x		coords(myroc,x=.8)
#                                 x		coords(myroc,x="best")
#                                 x		coords(myroc,x=.8,input="sensitivity")
#                                 x		Looking at residuals as binary table 0/1 for each x value
#                                 x		Cooks D for looking at residuals
#                                 x		VIF(fit) to find highly correlated data
#                                 x		Combining x values to keep the degrees of freedom reasonable
#                                 x		Hosmer Lemeshow test
#                                 x		g=10 is a common rule
#                                 x		generalhoslem::logitgof(data$y,fitted(fit))
#                                 x	8	Penalized Firth logistic regression
#                                 x		logistf::logistf(y~x,data,family=binomial)
#                                 x		multinomial version of logistic modeling
#                                 x		VGAM::vglm(y~x1+x2+x3,family=multinomial(refLevel="y1"),data=data)
#                                 VGAM::vglm(cbind(y1,y2,y3)~x1+x2+x3,family=multinomial(refLevel="y1"),data=data)
#                                 x		t(coef(fit,matrix=TRUE)) See all the prediction equations
#                                 x		plotting yhat from predict statement
#                                 VGAM::vglm(cbind(y1,y2,y3)~x1+x2+x3,family=cumulative(parallel=T),data=data)
#                                 x		anova(fit,fit2,type="I")
#                                 x		deviance(fit)
#                                 x	9	Simulating logistic data
#                                 x		Sample size calculations
#Log-Linear e^B0 percent change
#summary(fit)$theta
#summary(fit)$SE.theta

# R CODE


# Method	Code
# test p wald	by hand (phat-p)/sqrt(phat*(1-phat)/n)
# test p Score	prop.test(x,n,alternative="two.sided",correct=FALSE)
# test p fishers	binom.test(x,n,alternative="two.sided",correct=FALSE)
# CI p Wald	binom::binom.confint(x,n,method="asymptotic")
# CI p Score	binom::binom.confint(x,n,method="wilson")
# CI p AC	binom::binom.confint(x,n,method="agresti-coull")
# CI p Fishers	binom::binom.confint(x,n,method="exact")
# CI p Jeffrey	binom::binom.confint(x,n,method="bayes")
# Midp CI	exactci::binom.exact(x,n,p, alternative = "less", midp = TRUE)
# CI poisson Wald	by hand ybar +- Z sqrt(Ybar/n)
# CI poisson Score	by hand  ybar+Z^2/2n+Z/sqrt(n) sqrt(ybar+Z^2/4n)
# CI poisson GOF	by hand Z={(s^2/xbar)-1}sqrt((n-1)/2)
# Relative Risk	by hand p1/p2
# RR CI	eiptools::riskratio.wald(datamatrix)
# Odds Ratio wald CI	epitools::oddsratio(data,method="wald")$measure
# Test p1-p2 exact	epitools::oddsratio(data,method="fisher")$p.value
# Odds Ratio Mid p	epitools::oddsratio(data)
# Test odds ratio=1	epitools::oddsratio(data)
# CI Odds Ratio	epitools::oddsratio(data)
# CI p1-p2 Wald	DescTools::BinomDiffCI(x1,n1,x2,n2,method="wald")
# CI p1-p2 Score	DescTools::BinomDiffCI(x1,n1,x2,n2,method="score")
# CI p1-p2 Newcomb	By hand
# CI p1-p2 AgrestiCaffo	DescTools::BinomDiffCI(x1,n1,x2,n2,method="ac")
# Breslow Day	DescTools::BreslowDayTest(data)
# Mantel Heanszel	mantelhaen.test(data,correct=FALSE)
# X2 GOF	chisq.test(data,p=prop)
# X2 Test indpendence	chisq.test(data)
# Logistic 0/1	glm(y~x,family=binomial)
# Logistic columns	glm(cbind(y1,y2)~x,family=binomial)
# Logistic counts	glm(y/n~x,weights=n,family=binomial)
# Logistic Linear	fit<-glm(y~x,family=binomial(link="identity"))
# Plot effects	plot(yhat~xhat,type="l")
# Logistic prediction	predict(fit,newdata=data.frame(x=10),type="response")
# Logistic predictions	predict(fit,newdata=data,type="response")
# Logistic confusion table	table(data$y,preds)
# LR test nested	anova(fit1,fit2)
# LR test nested pvalue	lmtest::lrtest(fit1,fit2)
# LT test	by hand Chisquare(residual1-residual2,df1-df2)
# CI for beta Wald	confint.default(fit)
# CI for beta L Profile	confint(fit)
# Probit 0/1	glm(y~x,family=binomial(link="probit"))
# Poisson counts	glm(y~x,data=data,family=poisson)
# Poisson by amount	glm(y~x+offset(log(n)),data=data,family=poisson)
# Poisson by amount	glm(y~x,offset=log(n),data=data,family=poisson)
# Log-Linear	glm(y~x,data=data,family=poisson)
# Negative Binomial	MASS::glm.nb(y~x,data=data)
# NB by amount	MASS::glm.nb(y~x+offset(log(n)),data=data)
# Zero-Inflated	pscl::zeroinfl(y~xpoisson|xzero,data=data)
# Zero-Inflated NB	pscl::zeroinfl(y~xpoisson|xzero,data=data,dist="negbin")
# AIC	AIC(fit)
# BIC	BIC(fit)
# ROC curve	pROC::roc(data$Y~predict.glm(fit,type="response"),plot=TRUE)
# AUC	pROC::roc(data$Y~predict.glm(fit,type="response"))$auc
# Logistic Marginal Effect	mfx::logitmfx(fit,atmean=FALSE,data=data.frame(x,y))
# Logistic confusion table	table(data$y,as.numeric(fitted(fit)>0.642))
# Cooks Distance	cooks.distance(fit)
# General Hoslem	generalhoslem::logitgof(data$y,fitted(fit),g=10)
# Firth Penalized Logistic	logistf::logistf(y~x,family=binomial)
# Multinomial columns	vglm(cbind(y1,y2,y3)~x,family=multinomial(refLevel="y1"),data=data)
# Mulitnomial single var	vglm(y~x,family=multinomial(refLevel="y1"),data=data)
# Multinomial counts	vglm(y~x,weight=n,family=multinomial(refLevel="y1"),data=data)
# Sort multi pred eqns	t(coef(fit,matrix=TRUE))
# SimLog create x	x<-runif(n,0,10)
# SimLog create model	model<-B0+B1*x
# SimLog create probs	probs<-exp(model)/(1+exp(model))
# SimLog create y	y<-rbinom(n,1,probs)
# SimLog test p	summary(fit)$coefficients[-1]<0.05
# SimLog for loop	for(loop in 1:10000){
#   LDA analysis	MASS:lda(y~x1+x2,data=data)
#   LDA predictions	predict(fit,newdata=data.frame(x=10),type="response")
#   LDA pred table	table(data$y,predict(fit,type="response")$class)
#   LDA graph prin comp	plot(fit,dimen=2,col=as.numeric(factor(data$y)))
#   Num Clusters elbow	factoextra::fviz_nbclust(sdata,kmeans,method="wss")
#   Cluster Analysis	kmeans(sdata,3,nstart=25)
#   Graph Clusters	factoextra::fviz_cluster(fit,sdata)
#   Decision Tree	tidymodels::fit(decision_tree(mode="classification"),y~x,data=data)
#   Plot decision tree	rpart.plot::rpart.plot(fit$fit,type=4)
#   Decision tree conf table	table(data$buy,predict(fit,new_data=data)[[1]])
#   VIP	vip::vip(fit)
#   Prediction	predict(fit,new_data=data.frame(x1=9))
#   XGBoost RF binary	xgboost::xgboost(data=covariates,label=y,nrounds=4,objective="binary:logistic")
#   XGB RF multinomial	xgboost::xgboost(data=covariates,label=y,nround=3,objective="multi:softmax",num_class=4)
#   XGRF VIP	vip::vip(fit)
#   XGRF VIP	DiagrammeR::xgb.plot.importance(xgb.importance(names(y), model = fit))
#   XGRF prediction	predict(fit,as.matrix(data.frame(x1=2,x2=3))
#                           XGRF confusion	table(numy,predict(model,newdata=covariates))
#  Turn of scientific notation	sigs<-0
#False Positive rate = FP/(FP + TN)
#False negative rate = FN/(FN + TP)
#Sensitivity         = TP/(TP + FN)
#Specificity         = TN/(TN + FP)
#Rate of change      = B1 p(1 - p)
#Wald Prediction Interval : pi <- predict.glm(fit, data.frame(age = 30, gender = 1, z1 = 1, z2 = 1, z3 = 0),type = "response",se.fit = TRUE), pi$fit + c(-1,1)*1.96*pi$se.fit
#To include only effect X:Y
#pi = e^(b0+b1x...)/( 1 + e^(b0+b1x ...))