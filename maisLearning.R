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
library(Metrics)
#---------------------------------------------


#Les données concernent le rendement du maïs en France, dans les différents département sur plusieurs années. 
#L'objectif est de prédire le rendement à partir de données climatiques. Il y a 2300 individus et 58 variables : 
#57 prédicteurs.On choisit dès à présent de supprimer la colonne X contenant un identifiant unique n'apportant 
#pas d'informations particulières. on a p < n

#--------------------------------------------------------------------------------------

biasSquared = function(estimate, truth) {
  
  mean((mean(estimate) - truth))**2
} 

variance = function(estimate, truth) {
  var = mean((estimate - mean(estimate))**2)
  return(var)
} 

# ETL Data exploration

data <- read.csv(file="mais_train.csv", header = TRUE, sep = ",", row.names = 1)
names(data)
head(data)
dim(data)
set.seed(111)
boxplot(data$yield_anomaly)  
data
index <- sample(nrow(data), 0.8*nrow(data))
train <- data[index,]
test <- data[-index,]
y.train <- train[,2]
ntrain <- length(train$yield_anomaly)
ntest <- length(test$yield_anomaly)

y.train

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

#on remarque des coefficients relativement forts entre certaines variables explicatives
data[,-2]
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



# Selection des variables

##Subset selection (forward & backward)
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

## Régularisation
#Sélection de la pénalité par validation croisée.
###ridge
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

###LASSO
for (m in (1:length(models))) {
  x.train <- model.matrix(models[[m]], train)[,-1]
  x.test <- model.matrix(models[[m]], test)[,-1]
  cv.lasso <- cv.glmnet(x.train, y.train ,alpha = 1)
  plot(cv.lasso)
  model.lasso <- glmnet(x.train,y.train , alpha = 1, lambda=cv.lasso$lambda.min)
  pred.test <- predict(model.lasso,s=cv.lasso$lambda.min,newx=x.test)
  
  print(data.frame(
    model = m,
    lambda_min = cv.lasso$lambda.min,
    mse_test = mse(pred.test, test$yield_anomaly)
  ))
}

###elastic net
alpha <- c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)
for (m in (1:length(models))) {
  mse <- c()
  for (i in alpha ){
    x.train <- model.matrix(models[[m]], train)[,-1]
    x.test <- model.matrix(models[[m]], test)[,-1]
    
    cv.elnet <- cv.glmnet(x.train, y.train , alpha = i)
    model.elnet <- glmnet(x.train,y.train , lambda=cv.elnet$lambda.min, alpha = i)
    pred.test <- predict(model.elnet,s=cv.elnet$lambda.min,newx=x.test)
    mse <- cbind(mse, mse(pred.test, test$yield_anomaly))
    print(data.frame(
        model = m,
        lambda_min = cv.lasso$lambda.min,
        mse_test = mse(pred.test, test$yield_anomaly)
      ))
  }
  plot(alpha,mse, type = "l")
}

# On se propose de tester 3 régularisations différentes : 
#le ridge (?? = 0), le lasso (?? = 1) et elasticnet dont la valeur de alpha est déterminé par CV pour chacun des modèles
#O.705 enviuron pour tous : peu d'impact

## Curse of dimensionality

###PCR
pcr_model <- pcr(yield_anomaly~., data = train,scale = TRUE, validation = "CV")
validationplot(pcr_model, val.type="MSEP")
pcr_pred <- predict(pcr_model, test, ncomp = 57)
mean((pcr_pred - test[["yield_anomaly"]])^2)
# 0.70255
#il faut 57 composantes principales.
variance(pcr_pred, test$yield_anomaly) #0.33 var
biasSquared(pcr_pred, test$yield_anomaly) # bias 0.00001

#Models
##Linear Regression
for (m in (1:length(models))) {
  model <- train(models[[m]], train ,method = "lm",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))
  predict <- predict(model, test)
  error <- predict - test[["yield_anomaly"]]
  MSE <- mean(error^2)
  print(models[[m]])
  print(MSE)
}
#model 1 0.7014732


##KNN
for (m in (1:length(models))) {
  model <- train(models[[m]], train ,method = "knn",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 10))
  predict <- predict(model, test)
  error <- predict - test[["yield_anomaly"]]
  MSE <- mean(error^2)
  print(models[[m]])
  print(MSE)
}
#0.76 pour model 1, 0.9 pour model 2 et 3, moyen

## Regression multinomial 
#  Le nombre de combinaison est trop grand, ce qui nous empêche de tester.
## GAM
library(gam)
model <- gam(yield_anomaly ~ ns(year_harvest,df=5)+ ns(NUMD,df=5) + ns(IRR,df=5) + ns(ETP_1,df=5)+ ns(ETP_2,df=5)+ ns(ETP_3,df=5)+ ns(ETP_4,df=5)+ ns(ETP_5,df=5)+ ns(ETP_6,df=5)+ ns(ETP_7,df=5)+ ns(ETP_8,df=5)+ ns(ETP_9,df=5)+ ns(PR_1,df=5)+ ns(PR_2,df=5)+ ns(PR_3,df=5)+ ns(PR_4,df=5)+ ns(PR_5,df=5)+ ns(PR_6,df=5)+ ns(PR_7,df=5)+ ns(PR_8,df=5)+ ns(PR_9,df=5)+ ns(RV_1,df=5)+ ns(RV_2,df=5)+ ns(RV_3,df=5)+ ns(RV_4,df=5)+ ns(RV_5,df=5)+ ns(RV_6,df=5)+ ns(RV_7,df=5)+ ns(RV_8,df=5)+ ns(RV_9,df=5)+ ns(SeqPR_1,df=5)+ ns(SeqPR_2,df=5)+ ns(SeqPR_3,df=5)+ ns(SeqPR_4,df=5)+ ns(SeqPR_5,df=5)+ ns(SeqPR_6,df=5)+ ns(SeqPR_7,df=5)+ ns(SeqPR_8,df=5)+ ns(SeqPR_9,df=5)+ ns(Tn_1,df=5)+ ns(Tn_2,df=5)+ ns(Tn_3,df=5)+ ns(Tn_4,df=5)+ ns(Tn_5,df=5)+ ns(Tn_6,df=5)+ ns(Tn_7,df=5)+ ns(Tn_8,df=5)+ ns(Tn_9,df=5)+ ns(Tx_1,df=5)+ ns(Tx_2,df=5)+ ns(Tx_3,df=5)+ ns(Tx_4,df=5)+ ns(Tx_5,df=5)+ ns(Tx_6,df=5)+ ns(Tx_7,df=5)+ ns(Tx_8,df=5)+ ns(Tx_9,df=5), data = train)
predict <- predict(model, test[,-2])
error <- predict - test[["yield_anomaly"]]
MSE <- mean(error^2)
print(MSE)
model <- gam(yield_anomaly ~ ns(IRR,df=5) + ns(ETP_2,df=5)+ ns(ETP_3,df=5)+ ns(ETP_6,df=5)+ ns(ETP_7,df=5)+ ns(PR_4,df=5)+ ns(PR_5,df=5)+ ns(PR_6,df=5)+ ns(PR_7,df=5)+ ns(PR_8,df=5)+ ns(PR_9,df=5)+ ns(RV_2,df=5)+ ns(RV_6,df=5)+ ns(SeqPR_1,df=5)+ ns(SeqPR_2,df=5)+ ns(SeqPR_4,df=5)+ ns(SeqPR_9,df=5)+ ns(Tn_1,df=5)+ ns(Tn_3,df=5)+ ns(Tn_4,df=5)+ ns(Tn_5,df=5)+ ns(Tn_8,df=5)+ ns(Tx_2,df=5)+ ns(Tx_3,df=5), data = train)
predict <- predict(model, test)
error <- predict - test[["yield_anomaly"]]
MSE <- mean(error^2)
print(MSE)
model <- gam(yield_anomaly ~ ns(ETP_2,df=5)+ ns(ETP_3,df=5)+ ns(ETP_6,df=5)+ ns(ETP_7,df=5)+ns(ETP_8,df=5)+ ns(PR_4,df=5)+ ns(PR_5,df=5)+ ns(PR_7,df=5)+ ns(PR_9,df=5)+ ns(RV_2,df=5)+ ns(RV_6,df=5)+ ns(SeqPR_1,df=5)+  ns(SeqPR_4,df=5)+ ns(SeqPR_9,df=5)+ ns(Tn_5,df=5)+ ns(Tn_7,df=5)+ ns(Tx_2,df=5)+ ns(Tx_3,df=5)+ ns(Tx_4,df=5)+ ns(Tx_7,df=5), data = train)
predict <- predict(model, test)
error <- predict - test[["yield_anomaly"]]
MSE <- mean(error^2)
print(MSE)

## SVR
for (m in (1:length(models))) {
  CC<-c(0.01,0.1,1,10,100,1000)
  N<-length(CC)
  err<-rep(0,N)
  for(i in 1:N) {
    err[i]<-cross(ksvm(models[[m]], data = train, kernel = "rbfdot", C=CC[i], cross=5, kpar="automatic"))
  }
  plot(CC,err,type="b",log="x",xlab="C",ylab="CV error")
  
  
  svmfit <- ksvm(models[[m]], data = train ,kernel="rbfdot", C=CC[which.min(err)], epsilon=0.1)
  predict <- predict(svmfit, test)
  error <- predict - test[["yield_anomaly"]]
  MSE <- mean(error^2)
  print(MSE)
}

svmfitRBF <- ksvm(yield_anomaly~., data = train , type="eps-svr", kernel="rbfdot", C=1, scaled= TRUE, epsilon=0.05, kpar=list(sigma=0.0291), cross=5)
predictRBF <- predict(svmfitRBF, test)
error <- predictRBF - test$yield_anomaly
MSE <- mean(error^2)
print(MSE)
variance(predictRBF, test$yield_anomaly) #0.424 var
biasSquared(predictRBF, test$yield_anomaly) # bias 2.6*10-5

#0.632 modele 1 - rbfdot Cost = 1 ; sigma = 0.0291


#laplaciankernel 
# err of 0.56XX   CC find then sigma
MSElaplace <- rep(0,length(models))
for (m in (1:length(models))) {
  S<-c(0.05,0.06, 0.07, 0.08, 0.085)
  N<-length(S)
  err<-rep(0,N)
  for(i in 1:N) {
    svmfitC <- ksvm(yield_anomaly~., data = train, kernel = "laplacedot", C=6, cross=5, kpar=list(sigma=S[i]))
    predict <- predict(svmfitC, test)
    error <- predict - test[["yield_anomaly"]]
    err[i] <- mean(error^2)
  }
  
  plot(S,err,type="b",log="x",xlab="Sigma",ylab="CV error")
  
  
  svmfit <- ksvm(models[[m]], data = train ,kernel="laplacedot", C=6, epsilon=0.1, kpar=list(sigma=S[which.min(err)]))
  predict <- predict(svmfit, test)
  error <- predict - test[["yield_anomaly"]]
  MSElaplace[m] <- mean(error^2)
}
MSElaplace

CC<-c(0.1, 1, 5, 10, 100)
N<-length(CC)
err<-rep(0,N)
for(i in 1:N) {
  svmfitC <- ksvm(yield_anomaly~., data = train, type="eps-svr", kernel = "laplacedot", C=CC[i], cross=5,  epsilon=0.1, kpar=list(sigma=0.085), scaled=TRUE)
  predict <- predict(svmfitC, test)
  error <- predict - test[["yield_anomaly"]]
  err[i] <- mean(error^2)
}
plot(CC,err,type="b",log="x",xlab="Cost",ylab="CV error")

#Finally build model and get a boxplot
NREP = 10
finalErrs = rep(0,10)
for (m in (1:NREP)) {
  index <- sample(nrow(data), 0.8*nrow(data))
  train <- data[index,]
  test <- data[-index,]
  svmfit <- ksvm(yield_anomaly~., data = train , type="eps-svr", kernel="laplacedot", C=6, scaled= TRUE, epsilon=0.1, kpar=list(sigma=0.085), cross=5)
  predict <- predict(svmfit, test)
  error <- predict - test[["yield_anomaly"]]
  finalErrs[m] <- mean(error^2)
}

boxplot(finalErrs)

svmfit <- ksvm(yield_anomaly~., data = train , type="eps-svr", kernel="laplacedot", C=6, scaled= TRUE, epsilon=0.1, kpar=list(sigma=0.085), cross=0)
svmfit
predict <- predict(svmfit, test)
error <- predict - test[["yield_anomaly"]]
MSE <- mean(error^2)
print(MSE)
variance(predict, test$yield_anomaly) #0.48 var
biasSquared(predict, test$yield_anomaly) # bias 1.34*10-5
#MSE of 0.54589 Env


## Keras neural network
library(keras)

model = keras_model_sequential() %>% 
  layer_dense(units=64, activation="relu", input_shape=57) %>% 
  layer_dense(units=40, activation = "relu") %>% 
  layer_dense(units=30, activation = "relu") %>% 
  layer_dense(units=15, activation = "relu") %>% 
  layer_dense(units=1, activation="linear")

model %>% compile(
  loss = "mse",
  optimizer =  "adam", 
  metrics = list("mean_absolute_error")
)

model %>% summary()

index <- sample(nrow(data), 0.8*nrow(data))
train <- data[index,]
test <- data[-index,]
x.train <- train[,-2]
y.train <- train[,2]
x.test <- test[,-2]
y.test <- test[,2]
ntrain <- length(train$yield_anomaly)
ntest <- length(test$yield_anomaly)

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)

history <- model %>% fit(
  as.matrix(x.train),
  as.matrix(y.train),
  epochs = 300,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(early_stop)
)

plot(history)
#We can see an early stop at approx 220 epochs
scores = model %>% evaluate(as.matrix(x.train), as.matrix(y.train), verbose = 0)
print(scores)
#0.5391511 MSE on training datas


testdata <- read.csv(file="testMais.csv", header = TRUE, sep=",")
testdata <- read.table(file="mais_test.txt", sep = "" , header = T , na.strings ="", stringsAsFactors= F)
nrow(testdata)



y_pred = model %>% predict(as.matrix(x.test))
error <- y_pred - y.test
MSE <- mean(error^2)
print(MSE)


c(loss, mse) %<-% (model %>% evaluate(as.matrix(as.matrix(x.test)),as.matrix(y.test), verbose = 0))
loss
mse
# Result on MLP - loss = 0.6069925 and mse = 0.5703435


# Conclusion

"""--
Without real explained analysis for the moment though --
We can clearly point out 2 models that best fits our regression problem here.
We managed some feature extraction/PCA's, Subset / Regularization but loss didn't dropped under 0.7

SVR : Model we built result in an MSE of 0.57XX approx with modele 1 (complete) - laplaciankernel 
Cost = 6 ; sigma = 0.085
"""

#Preds
#------------- SAMPLE RENEW ------------------------------------------------#
biasSquared = function(estimate, truth) {
  
  mean((mean(estimate) - truth))**2
} 

variance = function(estimate, truth) {
  var = mean((estimate - mean(estimate))**2)
  return(var)
} 

#Loading datas
data <- read.csv(file="mais_train.csv", header = TRUE, sep = ",", row.names = 1)
index <- sample(nrow(data), 0.8*nrow(data))
train <- data[index,]
test <- data[-index,]

###TEST
dataT <- read.table("mais.txt", sep = "" , header = T)
dataT <- dataT[!(rownames(dataT) %in% rownames(data)),]
#-------------CHOICES MODELS ------------------------------------------------#
#SVM
svmfitfinal <- ksvm(yield_anomaly~., data = train , type="eps-svr", kernel="laplacedot", C=6, scaled= TRUE, epsilon=0.1, kpar=list(sigma=0.085), cross=0)
predictsvmfinal <- predict(svmfitfinal, dataT[,-2])
errorSVMfinal <- predictsvmfinal - dataT$yield_anomaly
MSESVMfinal <- mean(errorSVMfinal^2)
print(MSESVMfinal) #0.435
variance(predictsvmfinal, test$yield_anomaly) #0.45 var
biasSquared(predictsvmfinal, test$yield_anomaly) # bias 4*10-4
#-------------CHOICES MODELS ------------------------------------------------#
