classifieur_mais <- function(dataset) {
  # Chargement de l environnement

  library(kernlab)

  predictions <- predict(svmfitfinal, newdata = dataset)
  return(predictions)
}
