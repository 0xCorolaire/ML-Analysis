regresseur_mais <- function(dataset) {
  # Chargement de l environnement
  load("env.Rdata") 
  library(MASS)
  library(leaps)
  library(kernlab)

  predictions <- predict(svm.reg, newdata = dataset)
  return(predictions)
}

classifieur_astronomie <- function(dataset) {
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

classifieur_images <- function(list) {
  library(keras)
  library(jpeg)
  
  load("env.Rdata")
  model = unserialize_model(modelClassifImage, custom_objects = NULL, compile = TRUE)
  
  n<-length(list)
  dim_images <- c(200,200)
  predictions <- rep(0, length(list))
  for(i in 1:n){
    images <- array(0, dim=c(2, dim_images[1], dim_images[2], 3))
    images[1,,,] <- image_to_array(image_load(path=list[i], target_size=dim_images, interpolation = "bilinear"))
    temp <- predict(model, images[,,,], batch_size = 2)[1,]
    predictions[i] <- which.max(temp)
  }
  
  predictions[predictions==1] = 'car'
  predictions[predictions==2] = 'cat'
  predictions[predictions==3] = 'flower'
  
  return(predictions)
}
