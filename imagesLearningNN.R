library(keras)
library(OpenImageR)
library(imager)

install_keras()

## CNN model =======================================================
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
modelCNN <- keras_model_sequential()

# add layers
modelCNN %>%
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
modelCNN %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

train_samples = train_image_array_gen$n
valid_samples = valid_image_array_gen$n
batch_size = 32
epochs = 30

early_stopping <- callback_early_stopping(monitor = 'val_loss', patience = 3)

# fit
hist <- modelCNN %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # callbacks
  callbacks = c(early_stopping)
)

plot(hist)

modelCNN %>% save_model_hdf5("modelCNN.h5")
modelCNN = load_model_hdf5("modelCNN.h5", compile = TRUE)

# Test with new data
x_test = list()
y_test = array()

for(class_name in c('car', 'cat', 'flower')) {
  for(i in 1:10) {
    path = file.path(getwd(), 'images_test_exterieur', class_name, paste(class_name, i, '.jpg', sep = ""))
    img = readImage(path)
    img = resizeImage(img, width = 100, height = 100, method = 'bilinear')
    ind = i
    if(equals(class_name, 'cat')) {
      ind = ind + 10
    }
    if(equals(class_name, 'flower')) {
      ind = ind + 20
    }
    x_test[[ind]] = img
    y_test[ind] = class_name
  }
}

img_test2 = array(0, dim = c(2, 100, 100, 3))
img_test = image_load(file.path(getwd(), 'images_test_exterieur', 'cat', 'cat10.jpg'), target_size = c(100, 100), interpolation = 'bilinear')
img_test2[1,,,] = image_to_array(img_test)
y = predict_classes(modelCNN, img_test2[,,,])



image_path <- file.path(file.path(getwd(), 'images_test_exterieur', 'cat', 'cat8.jpg'))
test_image <- image_load(image_path, target_size = c(100,100))
image_tensor <- image_to_array(test_image)
image_tensor <- array_reshape(image_tensor, c(1, c(100,100), 3))
image_tensor <- image_tensor / 255
plot(as.raster(image_tensor[1,,,]))

paste0("Probability: ", predict_proba(modelCNN, x=image_tensor))
paste0("Predicted class: ", predict_classes(modelCNN, x=image_tensor))

