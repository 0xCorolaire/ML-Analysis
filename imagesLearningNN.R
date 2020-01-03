library(keras)
library(OpenImageR)
library(imager)

install_keras()

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

## Play
path = file.path(getwd(), 'images_train/cat/cat_train_1.jpg')
img = readImage(path)
imageShow(img)
img = rgb_2gray(img)
img = resizeImage(img, width = 100, height = 100)
length(img)
img = processImage(img)
## EndPlay

data = matrix(ncol = 10001, nrow = nb_cats + nb_cars + nb_flowers)

# 0 = cat
# 1 = car
# 2 = flower
for (i in 1:nb_cats) {
  filename = paste('cat_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'cat', filename)
  img = readImage(path)
  img = processImage(img)
  img[10001] = 0
  data[i,] = img
}

for (i in 1:nb_cars) {
  filename = paste('car_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'car', filename)
  img = readImage(path)
  img = processImage(img)
  img[10001] = 1
  data[nb_cats + i,] = img
}

for (i in 1:nb_flowers) {
  filename = paste('flower_train_', i, '.jpg', sep = "")
  path = file.path(getwd(), 'images_train', 'flower', filename)
  img = readImage(path)
  img = processImage(img)
  img[10001] = 2
  data[i + nb_cars + nb_cats,] = img
}

# scale
# s <- apply(data,1,sd)
# ii <- which(s>1)
# data <- data[ii,]


# The mnist dataset is a set of images 28x28 of handwritten numbers.
# mnist <- dataset_mnist()
# x_train <- mnist$train$x
# y_train <- mnist$train$y
# x_test <- mnist$test$x
# y_test <- mnist$test$y

x_train <- data[, -10001]
y_train <- data[, 10001]
y_train <- to_categorical(y_train, 3)

# The x data is a 3-d array (images,width,height) of grayscale values.

# reshape
# dim(data) <- c(nrow(data), 784)
# dim(x_test) <- c(nrow(x_test), 784)

# rescale
# x_train <- x_train / 255
# x_test <- x_test / 255

#y_train <- to_categorical(data, 10001)

# Here we consider the simplest model (linear stack of layers)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 1000, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 500, activation = "relu") %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 3, activation = "softmax")

summary(model)

# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)

# Training
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30,
  validation_split = 0.2
)

plot(history)

# Evaluation of model
model %>% evaluate(x_test, y_test,verbose = 0)
