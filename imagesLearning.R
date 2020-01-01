library(MASS)
library(leaps)
library(kernlab)
library(caret)
library(e1071)
library(nnet)
library(keras)
install.packages('jpeg')
#---------------------------------

#Chargement images
img_car <- array(list.files(path='images_train/car', full.names = TRUE, pattern='*.jpg'))
img_cat <- array(list.files(path='images_train/cat', full.names = TRUE, pattern='*.jpg'))
img_flower <- array(list.files(path='images_train/flower', full.names = TRUE, pattern='*.jpg'))

res <- data.frame('Nombre d\'images'=rep(0,1))
res[,1] <- length(img_car)
res[,2] <- length(img_cat)
res[,3] <- length(img_flower)
colnames(res) <- c('Car', 'Cat', 'Flower')
percentage <- prop.table(res) * 100
cbind(freq=res, percentage=percentage)

y <- c(rep(0, length(img_car)), rep(1, length(img_cat)), rep(2, length(img_flower)))


set.seed(123)
imgs <- c(img_car, img_cat, img_flower)

idx <- sample(seq(1, 3), size = length(imgs), replace = TRUE, prob = c(.6, .2, .2))
imgs_train <- imgs[idx == 1]
imgs_val <- imgs[idx == 2]
imgs_test <- imgs[idx == 3]
y.train <- y[idx == 1]
y.val <- y[idx == 2]
y.test <- y[idx == 3]


read_images <- function(list_files, dim_images){
  images <- array(0, dim=c(length(list_files), dim_images[1], dim_images[2], 3))
  for (i in 1:length(list_files)){
    images[i,,,] <- image_to_array(image_load(path=list_files[i], target_size=dim_images, interpolation="bilinear"))
  }
  return(images)
}


dims <- c(200, 200)
train <- read_images(imgs_train, dims)
test <- read_images(imgs_val, dims)
y.test

nrow(train)

# SVM



svm.fit <- best.svm(pipeline~., data = trainset.df, gamma = 10^(-6:-1), cost = 10^(-1:1))
# Fit predictions and print error matrix
svm.pred <- predict(svm.fit, testset.df[,1:3])
svm.tab <- table(pred = svm.pred, true = testset.df[,4])
print(svm.tab)











