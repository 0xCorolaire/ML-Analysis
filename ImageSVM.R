library(OpenImageR)
library(SpatialPack)

# Functions
processImage = function(img) {
  # Convert to grey
  output = rgb_2gray(img)
  # Resize
  output = resizeImage(output, width = 100, height = 100, method = 'bilinear')
  # Convert to vector
  output = as.vector(output)
  # Cast to numeric
  output = as.numeric(output)
  # Return
  return(output)
}


# Read images
nb_cats = 590
nb_cars = 485
nb_flowers = 521

data = matrix(ncol = 10001, nrow = nb_cats + nb_cars + nb_flowers)

# 0 = cat
# 1 = car
# 2 = flower
for (i in 1:nb_cats) {
  filename = paste('cat_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'cat', filename)
  img = readImage(path)
  img = processImage(img)
  img = matrix(img,100,100)
  image(img)
  img[10001] = 'cat'
  data[i,] = img
}

for (i in 1:nb_cars) {
  filename = paste('car_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'car', filename)
  img = readImage(path)
  img = processImage(img)
  img[10001] = 'car'
  data[nb_cats + i,] = img
}

for (i in 1:nb_flowers) {
  filename = paste('flower_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'flower', filename)
  img = readImage(path)
  img = processImage(img)
  img[10001] = 'flower'
  data[i + nb_cars + nb_cats,] = img
}

x_train <- data[, -10001]
y_train <- data[, 10001]
y_train <- as.factor(y_train)

# PCA
pca<-prcomp(X1)
lambda<-pca$sdev^2

pairs(pca$x[,1:5],col=y,pch=as.numeric(y_train))
plot(cumsum(lambda)/sum(lambda),type="l",xlab="q",ylab="proportion of explained variance")
q<-500
X2<-scale(pca$x[,1:q])

# Definition of the classes for the 2-class problem
z<-(y=="car") | (y=="cat") | (y=="flower")
z<-as.factor(z)
levels(z)<-c("car","cat","flower")

# Split train/test
n<-nrow(X2)
train<-sample(1:n,round(2*n/3))
X.train<-X2[train,]
y.train<-y_train[train]
z.train<-z[train]
X.test<-X2[-train,]
y.test<-y_train[-train]
z.test<-z[-train]


library("kernlab")

# SVM avec noyau linéaire
CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(CC)
M<-10 # nombre de répétitions de la validation croisée
err<-matrix(0,N,M)
for(k in 1:M){
  for(i in 1:N){
    err[i,k]<-cross(ksvm(x=X.train,y=y.train,type="C-svc",kernel="vanilladot",C=CC[i],cross=5))
  }
}
Err<-rowMeans(err)
plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")

# Calcul de l'erreur de test avec la meilleure valeur de C
svmfit<-ksvm(x=X.train,y=y.train,type="C-svc",kernel="vanilladot",C=0.01)
pred<-predict(svmfit,newdata=X.test)
table(y.test,pred)
err<-mean(y.test != pred)
print(err)


# SVM avec noyau gaussien
CC<-c(0.001,0.01,0.1,1,10,100,1000,10e4)
N<-length(CC)
M<-10 # nombre de répétitions de la validation croisée
err<-matrix(0,N,M)
for(k in 1:M){
  for(i in 1:N){
    err[i,k]<-cross(ksvm(x=X.train,y=y.train,type="C-svc",kernel="rbfdot",kpar="automatic",C=CC[i],cross=5))
  }
}
Err<-rowMeans(err)
plot(CC,Err,type="b",log="x",xlab="C",ylab="CV error")

# Calcul de l'erreur de test avec la meilleure valeur de C
svmfit<-ksvm(x=X.train,y=y.train,type="C-svc",kernel="rbfdot",C=10)
pred<-predict(svmfit,newdata=X.test)
table(y.test,pred)
err<-mean(y.test != pred)
print(err)

