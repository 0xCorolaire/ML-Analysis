classifieur_mais <- function(dataset) {
  # Chargement de l environnement
  load("env.Rdata") 
  library(MASS)
  library(leaps)
  library(kernlab)

  predictions <- predict(svm.reg, newdata = dataset)
  return(predictions)
}

#TEST
data <- read.csv(file="mais_train.csv", header = TRUE, sep = ",", row.names = 1)
dataT <- read.table("mais.txt", sep = "" , header = T)
dataT <- dataT[!(rownames(dataT) %in% rownames(data)),]

errorSVMfinal <- classifieur_mais(dataT[,-2]) - dataT$yield_anomaly
MSESVMfinal <- mean(errorSVMfinal^2)
print(MSESVMfinal) #0.59



classifieur_astronomy <- function(dataset) {
  # Chargement de l environnement
  load("env.Rdata") 
  library(MASS)
  library(leaps)
  
  uselessName <- c("rerun", "run", "camcol", "objid", "field", "specobjid")
  indSc <- sapply(dataset, is.numeric)
  dataset[indSc] <- lapply(dataset[indSc], scale)
  dataset <- dataset[,!names(dataset) %in% uselessName]
  
  predictions <- predict(svm.class, newdata = dataset, type="response")
  return(predictions)
}

#TEST
data <- read.csv(file="astronomy_train.csv", header = TRUE, sep = ",")
dataT <- read.csv(file="astro.csv", header = TRUE, sep = ",")
dataT <- dataT[!(dataT$ra %in% data$ra),]
confusionSVM <- table(dataT$class, classifieur_astronomy(dataT))
confusionSVM
errSVM <- 1-sum(diag(confusionSVM))/nrow(dataT)
errSVM
