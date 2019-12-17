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
#---------------------------------------------
data <- read.csv(file="mais_train.csv", header = TRUE, sep = ",")

head(data)
nbdata <- nrow(data)
indSc <- sapply(data, is.numeric)
data[indSc] <- lapply(data[indSc], scale)
data
#Les données concernent le rendement du maïs en France, dans les différents département sur plusieurs années. 
#L'objectif est de prédire le rendement à partir de données climatiques. Il y a 2300 individus et 58 variables : 
#57 prédicteurs.On choisit dès à présent de supprimer la colonne X contenant un identifiant unique n'apportant 
#pas d'informations particulières. on a p < n

