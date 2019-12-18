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
data <- scale(data)
data <- data.frame(data)

#Les données concernent le rendement du maïs en France, dans les différents département sur plusieurs années. 
#L'objectif est de prédire le rendement à partir de données climatiques. Il y a 2300 individus et 58 variables : 
#57 prédicteurs.On choisit dès à présent de supprimer la colonne X contenant un identifiant unique n'apportant 
#pas d'informations particulières. on a p < n

boxplot(data[,2])  
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
ntrain <- length(train[,2])
ntest <- length(test[,2])

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
pca <- PCA(data[,-2])
par(mfrow=c(1,1))
plot(pca$eig[,3], type='l', ylab='cumulative percentage of variance', 
     xlab="components")
abline(h=90, col="red", lwd=2)
abline(h=95, col="blue", lwd=2)
#30 Comps

#Les hypothèses du modèle lineaire ne semblent pas etre valable car les residus 
#ne sont pas gaussiens (test de Shapiro-Wilk). Cependant dans le cas qu'un échantillon de grande taille, 
#ce modèle reste robuste et peux convenir. C'est pour cela que nous le testons par la suite.
corrplot::corrplot(cor(data), type="upper", tl.col="black", tl.srt=45)
trainAndTest <- createDataPartition(data, nbdata)
train.data <- trainAndTest$train
test.data <- trainAndTest$test
idx_train <- trainAndTest$smp_ind

fit.lm <- lm(yield_anomaly~., data=train.data)
summary(fit.lm)

pred.lm  <- predict(fit.lm, newdata=test.data[,-2])


mean((test.data[,2] - pred.lm)^2)
#MSE = 0.8005 - 0.907

#REGULARIZATION
#Need of a matrix not dataframe here
## RIDGE
cv.out <- cv.glmnet(as.matrix(train.data[,-2]), as.matrix(train.data[,2]), alpha=0)
plot(cv.out)

fit.ridge <- glmnet(as.matrix(train.data[,-2]), as.matrix(train.data[,2]), lambda = cv.out$lambda.min, alpha=0)
ridge.pred <- predict(fit.ridge, s= cv.out$lambda.min, newx = as.matrix(test.data[,-2]))

mean((test.data[,2]-ridge.pred)^2)
#0,728

## LASSO
cv.out <- cv.glmnet(as.matrix(train.data[,-2]), as.matrix(train.data[,2]), alpha=1)
plot(cv.out)

fit.lasso <- glmnet(as.matrix(train.data[,-2]), as.matrix(train.data[,2]), lambda = cv.out$lambda.min, alpha=1)
lasso.pred <- predict(fit.lasso, s= cv.out$lambda.min, newx = as.matrix(test.data[,-2]))

mean((test.data[,2]-ridge.pred)^2)
#0,72 same

## ELASTIC NET
ALPHA <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
nAlpha <- length(ALPHA)
group <- rep((1:10), (nbdata/10)+1)[1:nbdata]
err <- rep(0,nAlpha)
for(a in (1:nAlpha)) {
  CV <- rep(0,10)
  for(k in (1:10)) {
    idx.train <- which(group==k)
    val <- as.matrix(data[idx.train,])
    train <- as.matrix(data[-idx.train,])
    
    cv.out <- cv.glmnet(train[,-2], train[,2], alpha=a)
    
    fit.elas <- glmnet(train[,-2], train[,2], lambda = cv.out$lambda.min, alpha=a)
    elast.pred <- predict(fit.elas, s=cv.out$lambda.min, newx = val[,-2])
    CV[k] <- mean((val[,2]-elast.pred)^2)
  }
  err[a] = mean(CV)
}
err

cv.out <- cv.glmnet(as.matrix(train.data[,-2]),as.matrix(train.data[,2]), alpha=0.3)
fit.elas <- glmnet(as.matrix(train.data[,-2]), as.matrix(train.data[,2]), lambda = cv.out$lambda.min, alpha=0.3)
elast.pred <- predict(fit.elas, s=cv.out$lambda.min, newx = as.matrix(test.data[,-2]))
mean((as.matrix(test.data[,2])-elast.pred)^2)
#0.68


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

fit.rf <- randomForest(yield_anomaly~., data=data, subset = idx_train, mtry=7)
plot(fit.rf)


which.min(fit.rf$mse)
## [1] 496

# RMSE of this optimal random forest
fit.rf$mse[which.min(fit.rf$mse)]
#0.7472 and MSE = 0.5884

pred.rf <- predict(fit.rf, newdata=data.frame(test.data[,-2]))
mean((test.data[, 2] - pred.rf)^2)
#0.62

#SVR
svmfit <- ksvm(yield_anomaly~., data= data.frame(train.data), scaled=TRUE, type="eps-svr", 
               kerne="rbfdot", C=10, kpar=list(sigma=0.1))
yhat = predict(svmfit, newdata=test.data[,-2])
MSE = mean((test.data[,2] - yhat)^2)
MSE
#0.556




#Model selection
## Reg subset

regsubset1 <- regsubsets(yield_anomaly~., data=data, method = "forward", nvmax = 57)
summary(regsubset1)
plot(summary(regsubset1)$bic,xlab='Number of variables',ylab='BIC', type='l', main="forward")
which.min(summary(regsubset1)$bic)
#summary(regsubset1)$which[which.min(summary(regsubset1)$bic),-1]
names(data)[summary(regsubset1)$which[which.min(summary(regsubset1)$bic),-1]]
regsubset2 <- regsubsets(yield_anomaly~., data=data, method = "backward", nvmax = 57)
#summary(regsubset2)
plot(summary(regsubset2)$bic,xlab='Number of variables',ylab='BIC', type='l', main="backward")
which.min(summary(regsubset2)$bic)
#summary(regsubset2)$which[which.min(summary(regsubset2)$bic),-1]
names(data)[summary(regsubset2)$which[which.min(summary(regsubset2)$bic),-1]]

best_models <- c(
  yield_anomaly~.,
  yield_anomaly~IRR+ETP_2+ETP_3+ETP_6+ETP_7+PR_4+PR_5+PR_6+PR_7+PR_8+PR_9+RV_2+RV_6+SeqPR_1+SeqPR_2+SeqPR_4+SeqPR_9+Tn_1+Tn_3+Tn_4+Tn_5+Tn_8+Tx_2+Tx_3+Tx_7,
  yield_anomaly~ETP_2+ETP_3+ETP_6+ETP_7+ETP_8+PR_4+PR_5+PR_6+PR_7+PR_9+RV_2+RV_6+RV_7+RV_8+SeqPR_1+SeqPR_2+SeqPR_4+SeqPR_9+Tn_5+Tn_7+Tx_2+Tx_3+Tx_4+Tx_7
)

set.seed(5)
index <- sample(nrow(data), 0.8*nrow(data))
train <- data[index,]
test <- data[-index,]

for (m in (1:length(best_models))) {
  CC<-c(0.01,0.1,1,10,100,1000)
  N<-length(CC)
  err<-rep(0,N)
  for(i in 1:N) {
    err[i]<-cross(ksvm(best_models[[m]], data = data.frame(train), kernel = "rbfdot", epsilon=0.1, C=CC[i], cross=5))
  }
  plot(CC,err,type="b",log="x",xlab="C",ylab="CV error")
  svmfit <- ksvm(best_models[[m]], data = as.data.frame(train) ,kernel="rbfdot", C=CC[which.min(err)], epsilon=0.1)
  predict <- predict(svmfit, test)
  error <- predict - test[["yield_anomaly"]]
  MSE <- mean(error^2)
  print(MSE)
}






#KPCA
kpc <- kpca(as.matrix(train.data[,-2]), kernel="rbfdot", kpar=list(sigma=0.3))
kpc
eig(kpc)





#--------------------------------------------------------------------------------------



data <- read.csv("mais_train.csv")
names(data)
head(data)
dim(data)
set.seed(5)
boxplot(data$yield_anomaly)  
data
set.seed(5)
index <- sample(nrow(data), 0.8*nrow(data))
train <- data[index,]
test <- data[-index,]
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







regsubset1 <- regsubsets(yield_anomaly~., data=data, method = "forward", nvmax = 57)
#summary(regsubset1)
plot(summary(regsubset1)$bic,xlab='Number of variables',ylab='BIC', type='l', main="forward")
which.min(summary(regsubset1)$bic)
#summary(regsubset1)$which[which.min(summary(regsubset1)$bic),-1]
names(data)[summary(regsubset1)$which[which.min(summary(regsubset1)$bic),-1]]
regsubset2 <- regsubsets(yield_anomaly~., data=data, method = "backward", nvmax = 57)
#summary(regsubset2)
plot(summary(regsubset2)$bic,xlab='Number of variables',ylab='BIC', type='l', main="backward")
which.min(summary(regsubset2)$bic)
#summary(regsubset2)$which[which.min(summary(regsubset2)$bic),-1]
names(data)[summary(regsubset2)$which[which.min(summary(regsubset2)$bic),-1]]
models <- c(
  yield_anomaly~.,
  yield_anomaly~IRR+ETP_2+ETP_3+ETP_6+ETP_7+PR_4+PR_5+PR_6+PR_7+PR_8+PR_9+RV_2+RV_6+SeqPR_1+SeqPR_2+SeqPR_4+SeqPR_9+Tn_1+Tn_3+Tn_4+Tn_5+Tn_8+Tx_2+Tx_3+Tx_7,
  yield_anomaly~ETP_2+ETP_3+ETP_6+ETP_7+ETP_8+PR_4+PR_5+PR_6+PR_7+PR_9+RV_2+RV_6+RV_7+RV_8+SeqPR_1+SeqPR_2+SeqPR_4+SeqPR_9+Tn_5+Tn_7+Tx_2+Tx_3+Tx_4+Tx_7
)

library("")

for (m in (1:length(models))) {
  x.train <- model.matrix(models[[m]], train)[,-1]
  x.test <- model.matrix(models[[m]], test)[,-1]
  cv.ridge <- cv.glmnet(x.train, y.train, alpha = 0.9)
  plot(cv.ridge)
  model.ridge <- glmnet(x.train, y.train, alpha = 0.9, lambda=cv.ridge$lambda.min)
  pred.test <- predict(model.ridge,s=cv.ridge$lambda.min,newx=x.test)
  
  print(data.frame(
    model = m,
    lambda_min = cv.ridge$lambda.min,
    mse_test = mse(pred.test, test$yield_anomaly)
  ))
}
































