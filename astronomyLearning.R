library(MASS)
library(leaps)
library(kernlab)
library(caret)
library(corrplot)

data <- read.csv(file="astronomy_train.csv", header = TRUE, sep = ",")
head(data)
nbdata <- nrow(data)

createDataPartition <- function(data, nb) {
  set.seed(129)
  smp_size <- floor(0.80 * nb)
  train_ind <- sample(seq_len(nb), size = smp_size)
  train <- data[train_ind,]
  test <- data[-train_ind,]
  train
  returnList <- list("train" = train, "test" = test)
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
#KNN, LDA, QDA, LOGIC REG, RANDOM FOREST, SVM


trainAndTest <- createDataPartition(data, nbdata)
train.data <- trainAndTest$train
test.data <- trainAndTest$test

train.data[]

fit.lda <- lda(class ~ ., trainAndTest$train)

pred.lda <- predict(fit.lda, newdata=as.data.frame(test.data))

conf.lda <- table(test.data$class, pred.lda$class)

err.lda <- 100-accuracy(conf.lda)
err.lda #11%













