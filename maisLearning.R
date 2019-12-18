library(MASS)
library(leaps)
library(kernlab)
library(caret)
library(tidyverse)
library(corrplot)
library(naivebayes)
library(randomForest)
library(car)
library(glmnet)
library(mclust)
library(upclass)
library(klaR)
library(mda)
library(fda)
library(corrplot)
library(FactoMineR)
library(pls)
library(splines)
library(rpart)
#---------------------------------------------
data <- read.csv(file="mais_train.csv", header = TRUE, sep = ",", row.names = 1)

head(data)
nbdata <- nrow(data)
#Les données concernent le rendement du maïs en France, dans les différents département sur plusieurs années. 
#L'objectif est de prédire le rendement à partir de données climatiques. Il y a 2300 individus et 58 variables : 
#57 prédicteurs.On choisit dès à présent de supprimer la colonne X contenant un identifiant unique n'apportant 
#pas d'informations particulières. on a p < n

boxplot(data$yield_anomaly)  
nrow(data)
createDataPartition <- function(data, nb) {
  smp_size <- floor(0.80 * nb)
  train_ind <- sample(seq_len(nb), size = smp_size)
  train <- data[train_ind,]
  test <- data[-train_ind,]
  train
  returnList <- list("train" = train, "test" = test, "smp_ind" = train_ind)
  return(returnList)
}

trainAndTest <- createDataPartition(data, nbdata)
train <- trainAndTest$train
test <- trainAndTest$test
y.train <- train[,3]
ntrain <- length(train$yield_anomaly)
ntest <- length(test$yield_anomaly)

modlin=lm(yield_anomaly~., data)
# Residus
res=residuals(modlin)
# Histograme et QQ plot 
par(mfrow=c(1,3))
hist(residuals(modlin))
qqnorm(res)
qqline(res, col = 2)
# Residus
plot(modlin$fitted.values,res, asp=1)


#PCA to explore predictors
pca <- PCA(data[, -2])
plot(pca$eig[,2], type='l', ylab='cumulative percentage of variance', 
     xlab="components")
abline(h=90, col="red", lwd=2)
abline(h=95, col="blue", lwd=2)
#30 Comps



#Les hypothèses du modèle lineaire ne semblent pas etre valable car les residus 
#ne sont pas gaussiens (test de Shapiro-Wilk). Cependant dans le cas qu'un échantillon de grande taille, 
#ce modèle reste robuste et peux convenir. C'est pour cela que nous le testons par la suite.
par(mfrow=c(1,1))
corrplot::corrplot(cor(data), type="upper", tl.col="black", tl.srt=45)



data <- scale(data)

trainAndTest <- createDataPartition(data, nbdata)
  
train.data <- trainAndTest$train
test.data <- trainAndTest$test
idx_train <- trainAndTest$smp_ind

fit.lm <- lm(yield_anomaly~., data=data.frame(train.data))
summary(fit.lm)

pred.lm  <- predict(fit.lm, newdata=data.frame(test.data[,-2]))


mean((test.data[,-2] - pred.lm)^2)
#MSE = 1,277

#REGULARIZATION
## RIDGE
cv.out <- cv.glmnet(train.data[,-2], train.data[,2], alpha=0)
plot(cv.out)

fit.ridge <- glmnet(train.data[,-2], train.data[,2], lambda = cv.out$lambda.min, alpha=0)
ridge.pred <- predict(fit.ridge, s= cv.out$lambda.min, newx = test.data[,-2])

mean((test.data[,2]-ridge.pred)^2)
#0,728

## LASSO
cv.out <- cv.glmnet(train.data[,-2], train.data[,2], alpha=1)
plot(cv.out)

fit.ridge <- glmnet(train.data[,-2], train.data[,2], lambda = cv.out$lambda.min, alpha=1)
ridge.pred <- predict(fit.ridge, s= cv.out$lambda.min, newx = test.data[,-2])

mean((test.data[,2]-ridge.pred)^2)
#0,709

## ELASTIC NET
ALPHA <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
nAlpha <- length(ALPHA)
group <- rep((1:10), (nbdata/10)+1)[1:nbdata]
err <- rep(0,nAlpha)


for(a in (1:nAlpha)) {
  CV <- rep(0,10)
  for(k in (1:10)) {
    idx.train <- which(group==k)
    val <- data[idx.train,]
    train <- data[-idx.train,]
    
    fit.elas <- glmnet(train[,-2], train[,2], lambda = cv.out$lambda.min, alpha=a)
    cv.out <- cv.glmnet(train[,-2], train[,2], alpha=a)
    elast.pred <- predict(fit.elas, s=cv.out$lambda.min, newx = val[,-2])
    CV[k] <- mean((val[,2]-elast.pred)^2)
  }
  err[a] = mean(CV)
}
err

##PCR
pcr.fit <- pcr(yield_anomaly~., data=data.frame(train.data), scale=TRUE, validation="CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type="MSEP", legendpos="topright")


#TREE
fit.treeb <- rpart(yield_anomaly~., data=data.frame(data), subset=idx_train, method="anova")
plot(fit.treeb)
text(fit.treeb, pretty=0, cex = 0.8)

pred.treeb <- predict(fit.treeb, newdata=data.frame(data[-idx_train,-2]))
mean((data[-idx_train, 2] - pred.treeb)^2)
#0.924
fit.treeb <- rpart(yield_anomaly~., data=data.frame(data), subset=idx_train, method="anova", 
                   control=rpart.control(xval=10, minbucket=10, cp=0.00))
plot(fit.treeb)
text(fit.treeb, pretty=0, cex = 0.8)

pred.treeb <- predict(fit.treeb, newdata=data.frame(data[-idx_train,-2]))
mean((data[-idx_train, 2] - pred.treeb)^2)
#0.98

fit.rf <- randomForest(yield_anomaly~., data=data.frame(data))
plot(fit.treeb)

which.min(fit.treeb$mse)
## [1] 500

# RMSE of this optimal random forest
sqrt(fit.treeb$mse[which.min(fit.treeb$mse)])
#0.74

pred.treeb <- predict(fit.treeb, newdata=data.frame(data[-idx_train,-2]))
mean((data[-idx_train, 2] - pred.treeb)^2)
#



#SVM
svmfit <- ksvm(yield_anomaly~., data= data.frame(train.data), scaled=TRUE, type="eps-svr", 
               kerne="polydot", C=10, epsilon=0.1, kpar=list(degree=2))
yhat = predict(svmfit, newdata=test.data[,-2])

yhat
MSE = mean((test.data[,2] - yhat)^2)
sqrt(MSE)

#KPCA
kpc <- kpca(train.data[,-2], kernel="rbfdot", kpar=list(sigma=0.3))
kpc


