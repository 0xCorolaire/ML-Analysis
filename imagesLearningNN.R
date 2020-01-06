library(keras)
library(OpenImageR)
library(imager)

install_keras()

## Sequential model
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

x_train <- data[, -10001]
y_train <- data[, 10001]
y_train <- to_categorical(y_train, 3)

#y_train <- to_categorical(data, 10001)

# Here we consider the simplest model (linear stack of layers)
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 625, activation = "relu", input_shape = c(10000)) %>% 
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 312, activation = "relu") %>%
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
  epochs = 100, batch_size = 20,
  validation_split = 0.2
)

plot(history)

# Evaluation of model
model %>% evaluate(x_test, y_test,verbose = 0)

## CNN model
# Read images
train_image_array_gen = flow_images_from_directory(directory = "images_train", generator = image_data_generator(rescale = 1/255),
                                                   target_size = c(100, 100), color_mode = "rgb", classes = NULL,
                                                   class_mode = "categorical", batch_size = 32, shuffle = TRUE,
                                                   seed = NULL, save_to_dir = NULL, save_prefix = "",
                                                   save_format = "jpg", follow_links = FALSE, subset = NULL,
                                                   interpolation = "nearest")
valid_image_array_gen = flow_images_from_directory(directory = "images_validation", generator = image_data_generator(rescale = 1/255),
                                                   target_size = c(100, 100), color_mode = "rgb", classes = NULL,
                                                   class_mode = "categorical", batch_size = 32, shuffle = TRUE,
                                                   seed = NULL, save_to_dir = NULL, save_prefix = "",
                                                   save_format = "jpg", follow_links = FALSE, subset = NULL,
                                                   interpolation = "nearest")

# CNN model (Convolutional NN)
# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(100, 100, 3)) %>%
  layer_activation("relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same") %>%
  layer_activation_leaky_relu(0.5) %>%
  layer_batch_normalization() %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100) %>%
  layer_activation("relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(3) %>% 
  layer_activation("softmax")

# compile
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

train_samples = train_image_array_gen$n
valid_samples = valid_image_array_gen$n
batch_size = 32
epochs = 30

# fit
hist <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
)

plot(history)

