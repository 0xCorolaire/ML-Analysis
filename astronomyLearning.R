library(MASS)
library(leaps)
library(kernlab)
library(caret)
library(e1071)
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
#---------------------------------------------
data <- read.csv(file="astronomy_train.csv", header = TRUE, sep = ",")

head(data)

indSc <- sapply(data, is.numeric)
data[indSc] <- lapply(data[indSc], scale)
nbdata <- nrow(data)

createDataPartition <- function(data, nb) {
  smp_size <- floor(0.70 * nb)
  train_ind <- sample(seq_len(nb), size = smp_size)
  train <- data[train_ind,]
  test <- data[-train_ind,]
  train
  returnList <- list("train" = train, "test" = test, "smp_ind" = train_ind)
  return(returnList)
}


accuracy = function(confusion_matrix){
  sum(diag(confusion_matrix)/(sum(rowSums(confusion_matrix)))) * 100
}

#type of data we will use
sapply(data, class)
#type of classes
levels(data$class)
#distrib
percentage <- prop.table(table(data$class)) * 100
cbind(freq=table(data$class), percentage=percentage)
# => lot of galaxy (avg 50% and few QSO 9%)

uselessName <- c("rerun", "objid", "specobjid")
#dropping useless column
data <- data[,!names(data) %in% uselessName]

data$class <- as.factor(data$class)

outputData <- data[,11]
predictorData <- data[,-11]

par(mfrow=c(2,8))
for(i in 1:16) {
  boxplot(predictorData[,i], main=names(predictorData)[i])
}
par(mfrow=c(1,1))
plot(outputData)
featurePlot(x=predictorData, y=outputData, plot="box")

#first pass through different algorithm
#KNN, LDA, QDA, LOGIC REG, RANDOM FOREST, BAGGING SVM

###A LDA
#LDA s'utilise principalement quand on considere que les classes ont une matrice de covariance identique.
#on aura sans doute pas un bon taux d'erreur
trainAndTest <- createDataPartition(data, nbdata)
train.data <- trainAndTest$train
test.data <- trainAndTest$test
indic <- trainAndTest$smp_ind
fit.lda <- lda(class ~ ., train.data)
pred.lda <- predict(fit.lda, newdata=test.data[,-11])

conf.lda <- table(test.data$class, pred.lda$class)

err.lda <- 100-accuracy(conf.lda)
err.lda #8.8%

###B Naivesb
fit.naive <- naive_bayes(class~.,data=train.data)
pred.naive <- predict(fit.naive,newdata=test.data[,-11])

conf.naive <- table(test.data$class, pred.naive)

err.naive <- 100-accuracy(conf.naive)
err.naive #3.5%

### C randomForst and Bagging
bagg.fit = randomForest(class ~.,data=data, subset = indic,mtry=ncol(data) - 1)
rf.fit = randomForest(class ~.,data=data, subset = indic,mtry=4)
yhat1 = predict(bagg.fit,newdata=data[-indic,],type="response")
yhat2 = predict(rf.fit,newdata=data[-indic,],type="response")

table(test.data$class,yhat1)
table(test.data$class,yhat2)
1 - mean(test.data$class==yhat1)
1 - mean(test.data$class==yhat2)
#error of 1% very good

### D SVM
fit.svm <- ksvm(class ~., kernel="vanilladot", C=1, data=train.data)
pred.svm <- predict(fit.svm, newdata=test.data[,-11], type="response")
mean(test.data$class!=pred.svm)
#6.3% err for rdf
#1.9% for vanilladot

### E GMM
Xgmm <- as.matrix(data[,-11])
cl <- as.matrix(data[,11])
Cltrain = cl[indic,]
Xtrain = Xgmm[indic,]

Xtest <- Xgmm[-indic,]
Cltest <- cl[-indic,]

models <- c('EII','VII','VEI','VVV','EEE')
fitup <- upclassify(Xtrain, Cltrain, Xtest, modelscope = models)
fitup$Best$modelName #VVV
gmm.fit <- Mclust(data)
plot(gmm.fit)

mod2 <- MclustDA(Xtrain, Cltrain)
summary(mod2, newdata = Xtest, newclass=Cltest)
mod3 <- MclustDA(Xtrain, Cltrain, modelType = "EDDA")
summary(mod3)


cv <- cvMclustDA(mod2, nfold = 10)
str(cv)
unlist(cv[3:4])
#3.2% error with MclustDA normal
cv <- cvMclustDA(mod3, nfold = 10)
str(cv)
unlist(cv[3:4])
#1,8% error with EDDA


#Model selection
transformation <- function(X, degree=2, orthogonal.poly=TRUE){
  features.poly <- paste("+ poly(", names(X), ", degree=", degree,", 
                         raw=", !orthogonal.poly, ")", sep="", collapse = " ")
  features.log  <- paste("+ log(", names(X), ")", sep="", collapse = " ")
  features.exp.log <- paste("+ I(exp(", names(X), ")*log(", names(X), "))", sep="", collapse = " ")
  features <- paste(features.poly, features.exp.log, features.log, sep="")
  formule <- paste("~ -1 ", features)
  X.transform <- model.matrix(as.formula(formule), data = X)
  colnames(X.transform) <- paste("X", 1:ncol(X.transform), sep='')
  attr(X.transform, "assign") <- NULL
  return(X.transform)
}


##Regsubset
regsubset <- regsubsets(class ~ ., data=data, method = "exhaustive", nvmax = 14)
plot(regsubset, scale = "r2")
plot(regsubset, scale = "bic")
regsub.sum <- summary(regsubset)
regsub.sum
data.frame(
  Adj.R2 = which.max(regsub.sum$adjr2),
  BIC = which.min(regsub.sum$bic)
)

get_model_formula <- function(id, object, outcome){
  # get models data
  models <- summary(object)$which[id,-1]
  # Get outcome variable
  #form <- as.formula(object$call[[2]])
  #outcome <- all.vars(form)[1]
  # Get model predictors
  predictors <- names(which(models == TRUE))
  predictors <- paste(predictors, collapse = "+")
  # Build model formula
  as.formula(paste0(outcome, "~", predictors))
}

#get_model_formula(11, regsubset, "class")
best_models <- c(class~., get_model_formula(10, regsubset, "class"), get_model_formula(11, regsubset, "class"))

###LDA + regsubset + K-FOLD
err = rep(0,20)
err_mat = c()
K=10 
for (f in (1:3)) {
  for (l in (1:20)){
    folds=sample(1:K,nrow(data),replace = TRUE)
    CV <- rep(0,10)
    for (k in (1:K)){
      lda.cv <- lda(best_models[[f]], data = data[folds!=k,])
      pred.cv <- predict(lda.cv, newdata = data[folds==k,])
      confusion.cv <- table(data[folds==k,]$class, pred.cv$class)
      CV[k] <- 1-sum(diag(confusion.cv))/nrow(data[folds==k,])
    }
    err[l] <- mean(CV)
  }
  err_mat <- cbind(err_mat, err)
  print(best_models[[f]])
  print(mean(err))
}
boxplot(err_mat)
##Error => 9.3% pour formula 11 ---- Overfitting bad for LDA

### Naives Bayes + regsub + K-FOLD
err = rep(0,20)
err_mat = c()
K=10 
for (f in (1:3)) {
  for (l in (1:20)){
    folds=sample(1:K,nrow(data),replace = TRUE)
    CV <- rep(0,10)
    for (k in (1:K)){
      naive_bayes.cv <- naive_bayes(best_models[[f]], data = data[folds!=k,])
      pred.cv <- predict(naive_bayes.cv, newdata = data[folds==k,])
      confusion.cv <- table(data[folds==k,]$class, pred.cv)
      CV[k] <- 1-sum(diag(confusion.cv))/nrow(data[folds==k,])
    }
    err[l] <- mean(CV)
  }
  err_mat <- cbind(err_mat, err)
  print(best_models[[f]])
  print(mean(err))
}
boxplot(err_mat)
# err best with formula 11, et variance faibl e=> bon compromis bias var ~~ 3.56% err

### SVM + regsub + K fold with polydot or rbfdot or vanilladot
err = rep(0,20)
err_mat = c()
K=10 
for (f in (1:3)) {
  for (l in (1:20)){
    folds=sample(1:K,nrow(data),replace = TRUE)
    CV <- rep(0,10)
    for (k in (1:K)){
      svm.cv <- ksvm(best_models[[f]], kernel="polydot",kpar="automatic", C=1, data = data[folds!=k,])
      pred.cv <- predict(svm.cv, newdata = data[folds==k,], type="response")
      confusion.cv <- table(data[folds==k,]$class, pred.cv)
      CV[k] <- 1-sum(diag(confusion.cv))/nrow(data[folds==k,])
    }
    err[l] <- mean(CV)
  }
  err_mat <- cbind(err_mat, err)
  print(best_models[[f]])
  print(mean(err))
}
boxplot(err_mat)
# REDO with vanilladot => vanilladot is better

# best with formula 10 and polydot, kpar=list(degree=3,scale=2, offset=2) => 1.4% error
svm.cv <- ksvm(best_models[[2]], kernel="polydot" ,kpar=list(degree=3,scale=2, offset = 2), C=0.1, data = train.data)
pred.cv <- predict(svm.cv, newdata = test.data[,-11], type="response")
confusion.cv <- table(test.data$class, pred.cv)
errSVM <- 1-sum(diag(confusion.cv))/nrow(test.data)
errSVM
svm.cv
# poly error => 0.9%
#training error of 0.7% and 1% on test but hard to explain : as C value is quite a semehow good compromise, the .

#FINDING BEST C optimal

set.seed=(222)
CC <- c(800, 1000, 1200)
err_c <- rep(0, length(CC))
for (c in (1:length(CC))){
  svm.cv <- ksvm(class~., kernel="vanilladot", C=CC[c], data = train.data, cross=5)
  
  pred.cv <- predict(svm.cv, newdata = test.data[,-11], type="response")
  confusion.cv <- table(test.data$class, pred.cv)
  err_c[c] <- 1-sum(diag(confusion.cv))/nrow(test.data)
}
plot(err_c)
# After first ecremage : 1 - 10 - 100, - 1000 - 10 000 => find out 1 000 is best
# 2nd one on 700-1500 range=>

#findind 2000 best
svm.cv <- ksvm(class~., kernel="vanilladot", C=2000, data = train.data, cross=0)
pred.cv <- predict(svm.cv, newdata = test.data[,-11], type="response")
confusion.cv <- table(test.data$class, pred.cv)
confusion.cv
errSVM <- 1-sum(diag(confusion.cv))/nrow(test.data)
errSVM
svm.cv
# 0.6% error and C = 1000 might be a bit large for furhter datas
#n a SVM we are searching for two things: a hyperplane with the largest minimum margin, 
#and a hyperplane that correctly separates as many instances as possible


### Regularization

# RDA  
errRDA <- rep(0,20) 
err_mat = c()
for (f in (1:3)) {
  for (k in (1:20)){
    trainAndTestRDA <- createDataPartition(data, nbdata)
    train.dataRDA <- trainAndTestRDA$train
    test.dataRDA <- trainAndTestRDA$test
    indic <- trainAndTestRDA$smp_ind
    fit.rda <- rda(best_models[[f]], data = train.dataRDA, gamma = 0.05, lambda = 0.2)
    predictions <- fit.rda %>% predict(test.dataRDA[,-11])
    errRDA[k] <- 1 - mean(predictions$class == test.dataRDA$class)
  }
  print(best_models[[f]])
  print(mean(errRDA))
}
# very bad without best models in (10% error)

### ACP - PCA
pca<-princomp(data[,-11])
Z <- pca$scores
lambda<-pca$sdev^2
pairs(Z[,1:14],col=data[,11])
plot(cumsum(lambda)/sum(lambda),type="l",xlab="q",ylab="proportion of explained variance")
q <- 9
X2<-scale(Z[,1:q])
# svm a noyau finding of C best with CV
# SVM avec noyau linéaire
# Réglage de C par validation croisée
yPCA<-as.factor(data[,11])
yPCA
# Split train/test
n<-nrow(X2)
train<-sample(1:n,round(2*n/3))
X.train<-X2[train,]
y.train<-yPCA[train]
X.test<-X2[-train,]
y.test<-yPCA[-train]

CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(CC)
M<-10 # nombre de répétitions de la validation croisée
err<-matrix(0,N,M)
for(k in 1:M){
  for(i in 1:N){
    modS <- ksvm(x=X.train,y=y.train,type="C-svc",kernel="vanilladot",C=CC[i],cross=5)
    pred.cv <- predict(modS, newdata = X.test, type="response")
    confusion.cv <- table(y.test, pred.cv)
    err[i,k] <- 1-sum(diag(confusion.cv))/nrow(X.test)
  }
}
Err<-rowMeans(err)
plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")

svmfitPCA <- ksvm(x=X.train,y=y.train,type="C-svc",kernel="vanilladot",C=1000)
predPCA <- predict(svmfitPCA,newdata=X.test)
table(y.test,predPCA)
errPCA<-mean(y.test != predPCA)
print(errPCA)
#1.6% error

## FDA A FAIRE

## CCL for astronomy => randomForest with mtry = sqrt(p) = 4 best with 1% err
err = rep(0,20)
err_mat = c()
K=10 
for (f in (1:2)) {
  for (l in (1:20)){
    folds=sample(1:K,nrow(data),replace = TRUE)
    CV <- rep(0,10)
    for (k in (1:K)){
      randomForest.cv <- randomForest(best_models[[f]], data = data[folds!=k,], mtry=4)
      pred.cv <- predict(randomForest.cv, newdata = data[folds==k,], type = "response")
      confusion.cv <- table(data[folds==k,]$class, pred.cv)
      CV[k] <- 1-sum(diag(confusion.cv))/nrow(data[folds==k,])
    }
    err[l] <- mean(CV)
  }
  err_mat <- cbind(err_mat, err)
  print(best_models[[f]])
  print(mean(err))
}
boxplot(err_mat)
#model 10 with 1.01% err

#build model and find best mtry
err = rep(0,14)
err_mat = c()
K=10 
for (l in (1:14)){
  folds=sample(1:K,nrow(data),replace = TRUE)
  CV <- rep(0,10)
  for (k in (1:K)){
    randomForest.cv <- randomForest(best_models[[1]], data = data[folds!=k,], mtry=l)
    pred.cv <- predict(randomForest.cv, newdata = data[folds==k,], type = "response")
    confusion.cv <- table(data[folds==k,]$class, pred.cv)
    CV[k] <- 1-sum(diag(confusion.cv))/nrow(data[folds==k,])
  }
  err[l] <- mean(CV)
}
boxplot(err)
err # mtry = 44


#------------- SAMPLE RENEW ------------------------------------------------#

trainAndTest <- createDataPartition(data, nbdata)
train.data <- trainAndTest$train
test.data <- trainAndTest$test
indic <- trainAndTest$smp_ind

#-------------CHOICES MODELS ------------------------------------------------#

randomForest.mFinal <- randomForest(best_models[[2]], data = train.data, mtry=4)
predsRF <- predict(randomForest.mFinal, newdata = test.data[,-11], type = "response")
confusionRF <- table(test.data$class, predsRF)
errRF <- 1 - sum(diag(confusionRF))/nrow(test.data)
errRF
#0.9%
plot(randomForest.mFinal)
randomForest.mFinal

#                             ||||||||||||||||||||||||||

svm.mFinale <- ksvm(class~., kernel="vanilladot", C=2000, data = train.data, cross=0)
predsSVM <- predict(svm.mFinale, newdata = test.data[,-11], type="response")
confusionSVM <- table(test.data$class, predsSVM)
confusionSVM
errSVM <- 1-sum(diag(confusionSVM))/nrow(test.data)
errSVM
svm.mFinale
# 0.6%
#-------------CHOICES MODELS ------------------------------------------------#
