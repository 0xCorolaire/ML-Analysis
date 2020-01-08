library(MASS)
library(leaps)
library(kernlab)
library(caret)
library(e1071)
library(nnet)
library(keras)
#---------------------------------
#Ce problème est un problème de classification. 
#Chaque image contient un objet qui est sa classe. Nous avons trois classes: voitures, chats, et fleurs. 
#Il y a au total 1596 images.

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

#labels creation
y <- c(rep(0, length(img_car)), rep(1, length(img_cat)), rep(2, length(img_flower)))


set.seed(123)
imgs <- c(img_car, img_cat, img_flower)
idx <- sample(seq(1, 3), size = length(imgs), replace = TRUE, prob = c(.6, .2, .2))

#train/test/val split
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
val <- read_images(imgs_val, dims)


#MODEL 


model <- keras_model_sequential()

model %>%
  layer_conv_2d(
    filter=16,kernel_size=c(3,3),padding="same",strides=c(2, 2), input_shape = c(dims[1], dims[2], 3) ,activation="relu"
  ) %>%
  layer_conv_2d(filter=16,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_conv_2d(filter=16,kernel_size=c(3,3),padding="same",strides=c(2, 2),activation="relu") %>%
  layer_conv_2d(filter=16,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_dropout(0.5) %>%
  
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",strides=c(2, 2),activation="relu") %>%
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",strides=c(2, 2),activation="relu") %>%
  layer_conv_2d(filter=32,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_dropout(0.5) %>%
  
  layer_conv_2d(filter=64,kernel_size=c(3,3),padding="same",strides=c(2, 2),activation="relu") %>%
  layer_conv_2d(filter=64,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_conv_2d(filter=64,kernel_size=c(3,3),padding="same",strides=c(2, 2),activation="relu") %>%
  layer_conv_2d(filter=64,kernel_size=c(3,3),padding="same",activation="relu") %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>%
  
  layer_dense(64,activation="relu") %>%
  layer_dropout(0.5) %>%
  
  layer_dense(3,activation="softmax")

opt <- optimizer_adam(lr = 0.001)
model %>% compile(loss="sparse_categorical_crossentropy",optimizer=opt,metrics="accuracy")



#Training
early_stopping <- callback_early_stopping(monitor = 'val_acc', patience = 10)
history <- model %>% fit(train, y.train, batch_size=30, epochs=60,
                         validation_data=list(val, y.val), shuffle=TRUE,
                         callbacks = c(early_stopping),verbose=0)


#Saving model and evaluate
model %>% save_model_hdf5("model.h5")




