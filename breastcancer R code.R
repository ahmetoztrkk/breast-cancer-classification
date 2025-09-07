Sys.setlocale("LC_ALL","English")
library(dplyr)
library(ggplot2)
library("caret")
library(corrplot)
library(scales)
library(class)
library(gridExtra)
library(kknn)
library(pROC)
library(rpart)
library(e1071)
library(randomForest)
library(adabag)
library(gbm)
library(xgboost)
library(readxl)
cancerdata <- read_excel("D:/R/cancerdata.xlsx")
head(canserdata)
##library(fastAdaboost) fastAdaboost ??al??????rsa adaboost daha h??zl?? olur



#Veri on isleme
summary(cancerdata)
levels(cancerdata$diagnosis)
cancerdata$diagnosis <- as.factor(cancerdata$diagnosis)
sum(is.na(cancerdata))



#ID'yi modelden cikar

names(cancerdata) <- make.names(names(cancerdata))
cancerdata1 <- cancerdata[, -c(1)]
summary(cancerdata1)



# Orijinal veri i??in boxplot

numeric_data <- cancerdata1[sapply(cancerdata1, is.numeric)]
boxplot(numeric_data,
        main = "Boxplot (Orijinal Veriler)",
        las = 2,
        col = rainbow(ncol(numeric_data)))

scaled_df <- as.data.frame(scale(numeric_data))

# Standardize edilmi?? veri i??in boxplot
boxplot(scaled_df,
        main = "Boxplot (Standardize Edilmi?? Veriler)",
        las = 2,
        col = rainbow(ncol(scaled_df)))
#korelasyonlar
cor_happiness <- cor(numeric_data, use = "complete.obs")
corrplot.mixed(cor_happiness)
cor(numeric_data)


#de??i??kenlerin histogramlar??
par(mfrow = c(3, 3)) 
for(i in 1:ncol(numeric_data)) {
  hist(numeric_data[[i]], main = names(numeric_data)[i],
       xlab = "", col = "skyblue", border = "white")
}
par(mfrow = c(1, 1))  





###Train-test veri seti ayr??m?? %70 %30

dataset <- cancerdata1
validationIndex <- createDataPartition(cancerdata1$diagnosis, p=0.70, list=FALSE)
set.seed(123)
train <- dataset[validationIndex,]
test <- dataset[-validationIndex,]

###k-nn modeli

# 1. Ba????ms??z de??i??kenleri ay??r ve scale et
train_features <- train[, -which(names(train) == "diagnosis")]
test_features  <- test[, -which(names(test) == "diagnosis")]

# Scale i??lemi (train'e g??re)
train_scaled <- scale(train_features)
test_scaled <- scale(test_features, 
                     center = attr(train_scaled, "scaled:center"), 
                     scale = attr(train_scaled, "scaled:scale"))

train_labels <- train$diagnosis
test_labels  <- test$diagnosis

## 2. En iyi k'y?? belirlemek i??in train verisi ??zerinden cross-validation

set.seed(123)

# Cross-validation ayarlar??
ctrl <- trainControl(method = "cv", number = 10)
grid <- expand.grid(k = seq(1, 30, by = 2))  # sadece tek say??larla bakmak genelde daha sa??l??kl??d??r

## Modeli kur
knn_cv <- train(x = train_scaled, 
                y = train_labels, 
                method = "knn", 
                tuneGrid = grid, 
                trControl = ctrl)

# En iyi k
best_k <- knn_cv$bestTune$k
cat("En iyi k de??eri:", best_k, "\n")

# 3. Test verisi ile modelin performans??n?? de??erlendirme
predictions <- knn(train = train_scaled, test = test_scaled, cl = train_labels, k = best_k)

## Sonu??lar
confusion <- confusionMatrix(predictions, test_labels)
print(confusion)

# AUC ve ROC
pred_probs <- predict(knn_cv, newdata = test_scaled, type = "prob")

pos_class <- levels(test_labels)[2]  
roc_obj <- roc(response = test_labels,
               predictor = pred_probs[, pos_class],
               levels = rev(levels(test_labels)))  

auc_value <- auc(roc_obj)
cat("AUC de??eri:", auc_value, "\n")
plot(roc_obj, main = paste("ROC E??risi - AUC:", round(auc_value, 3)))


## Farkl?? mesafe metrikleri
euclidean_dist <- function(a, b) sqrt(sum((a - b)^2))
manhattan_dist <- function(a, b) sum(abs(a - b))
minkowski_dist <- function(a, b, p = 3) sum(abs(a - b)^p)^(1/p)

# Fonksiyon
knn_custom <- function(train_data, test_data, train_labels, k, dist_fun, p = NULL) {
  n_test <- nrow(test_data)
  pred <- vector("character", n_test)
  
  for (i in 1:n_test) {
    dists <- numeric(nrow(train_data))
    for (j in 1:nrow(train_data)) {
      if (!is.null(p)) {
        dists[j] <- dist_fun(train_data[j, ], test_data[i, ], p)
      } else {
        dists[j] <- dist_fun(train_data[j, ], test_data[i, ])
      }
    }
    neighbors <- order(dists)[1:k]
    votes <- train_labels[neighbors]
    pred[i] <- names(sort(table(votes), decreasing = TRUE))[1]
  }
  factor(pred, levels = levels(train_labels))
}

# Mesafeler i??in performans kar????la??t??rma
results <- list()

# 1. Euclidean  - zaten knn() da euclidean kullan??r ama burada manuel yap??yoruz
pred_euc <- knn_custom(train_scaled, test_scaled, train_labels, best_k, euclidean_dist)
conf_euc <- confusionMatrix(pred_euc, test_labels)
results$Euclidean <- conf_euc

# 2. Manhattan 
pred_man <- knn_custom(train_scaled, test_scaled, train_labels, best_k, manhattan_dist)
conf_man <- confusionMatrix(pred_man, test_labels)
results$Manhattan <- conf_man

# 3. Minkowski
pred_mink <- knn_custom(train_scaled, test_scaled, train_labels, best_k, minkowski_dist, p = 3)
conf_mink <- confusionMatrix(pred_mink, test_labels)
results$Minkowski_p3 <- conf_mink

# Sonu??lar?? yazd??r
for (metric in names(results)) {
  cat("\nMesafe metri??i:", metric, "\n")
  print(results[[metric]])
}

### Support Vector Machines Modeli

## 1. Farkl?? kernel fonksiyonlar??nda support vector machines modeli
kernels <- c("linear", "radial", "polynomial")
svm_results <- list()
accuracies <- numeric(length(kernels))

for (i in seq_along(kernels)) {
  kernel <- kernels[i]
  svm_model <- svm(x = train_scaled, y = train_labels, kernel = kernel, probability = TRUE)
  pred <- predict(svm_model, test_scaled)
  conf <- confusionMatrix(pred, test_labels)
  
  svm_results[[kernel]] <- list(model = svm_model, confusion = conf)
  accuracies[i] <- conf$overall['Accuracy']
  
  cat("\nKernel:", kernel, "\n")
  print(conf)
}

## 2. Do??rulu??a g??re en iyi kerneli se??
best_kernel <- kernels[which.max(accuracies)]
cat("\nEn iyi kernel:", best_kernel, "Accuracy:", max(accuracies), "\n")

## 3. En iyi kernele g??re Tuning
if (best_kernel == "linear") {
  tune_result <- tune(svm, train.x = train_scaled, train.y = train_labels,
                      kernel = best_kernel,
                      ranges = list(cost = seq(0.1, 3, by = 0.1)))
} else {
  tune_result <- tune(svm, train.x = train_scaled, train.y = train_labels,
                      kernel = best_kernel,
                      ranges = list(
                        cost = seq(0.1, 3, by = 0.1),      # 1, 2, 3, ..., 100
                        gamma = seq(0.1, 0.5, by = 0.05) # 0.001, 0.051, 0.101, ..., 1
                      ))
}

# Tuning
best_params <- tune_result$best.parameters
cat("En iyi parametreler:\n")
print(best_params)

## 4. En iyi ????kan parametreler ile final modeli kur
final_model <- svm(x = train_scaled, y = train_labels, 
                   kernel = best_kernel, 
                   cost = best_params$cost, 
                   gamma = ifelse(is.null(best_params$gamma), 0.1, best_params$gamma), 
                   probability = TRUE)

final_pred <- predict(final_model, test_scaled)
final_conf <- confusionMatrix(final_pred, test_labels)

cat("\nFinal Model Performans??:\n")
print(final_conf) #Tuning Ayarlar?? performans?? d??????rebiliyor??,

### Logistic Regression Modeli

# 1. data frame yap
train_df <- as.data.frame(train_scaled)
train_df$diagnosis <- train_labels

test_df <- as.data.frame(test_scaled)
test_df$diagnosis <- test_labels

# 2. Modeli kur 
log_model <- glm(diagnosis ~ ., data = train_df, family = binomial)

# 3. Test
log_probs <- predict(log_model, newdata = test_df, type = "response")
log_pred <- ifelse(log_probs > 0.5, levels(train_labels)[2], levels(train_labels)[1])
log_pred <- factor(log_pred, levels = levels(test_labels))

# 4. Confusion matrix
log_conf <- confusionMatrix(log_pred, test_labels)
cat("\nLogistic Regression Performans??:\n")
print(log_conf)

# 5. ROC ve AUC
roc_log <- roc(response = test_labels, predictor = log_probs, levels = rev(levels(test_labels)))
auc_log <- auc(roc_log)
cat("Logistic Regression AUC:", auc_log, "\n")
plot(roc_log, main = paste("Logistic ROC E??risi - AUC:", round(auc_log, 3)))

### A??a?? tabanl?? Modellemeler

## Decision Tree Modeli

# 1. Modeli kur
tree_model <- rpart(diagnosis ~ ., data = train, method = "class")

# 2. Test
tree_pred <- predict(tree_model, newdata = test, type = "class")

# 3. Confusion matrix
tree_conf <- confusionMatrix(tree_pred, test$diagnosis)
cat("\nDecision Tree Performans??:\n")
print(tree_conf)

# 4. ROC ve AUC
tree_probs <- predict(tree_model, newdata = test, type = "prob")
roc_tree <- roc(response = test$diagnosis,
                predictor = tree_probs[, levels(test$diagnosis)[2]],
                levels = rev(levels(test$diagnosis)))
auc_tree <- auc(roc_tree)
cat("Decision Tree AUC:", auc_tree, "\n")
plot(roc_tree, main = paste("Decision Tree ROC E??risi - AUC:", round(auc_tree, 3)))

# 5. A??ac?? g??rselle??tirme (??al????t??ramad??k)
#library(rpart.plot) 
#rpart.plot(tree_model, main = "Decision Tree")

## Random Forest

rf_model <- randomForest(diagnosis ~ ., data = train, ntree = 500, mtry = floor(sqrt(ncol(train) - 1)), importance = TRUE)

# 2. Test verisinde tahmin yap
rf_pred <- predict(rf_model, newdata = test)

# 3. Confusion Matrix
rf_conf <- confusionMatrix(rf_pred, test$diagnosis)
cat("\nRandom Forest Performans??:\n")
print(rf_conf)

# 4. ROC ve AUC
rf_probs <- predict(rf_model, newdata = test, type = "prob")
roc_rf <- roc(response = test$diagnosis,
              predictor = rf_probs[, levels(test$diagnosis)[2]],
              levels = rev(levels(test$diagnosis)))
auc_rf <- auc(roc_rf)
cat("Random Forest AUC:", auc_rf, "\n")
plot(roc_rf, main = paste("Random Forest ROC E??risi - AUC:", round(auc_rf, 3)))

# 5. De??i??ken ??nem Grafi??i
varImpPlot(rf_model, main = "Random Forest - De??i??ken ??nem Grafi??i")

## Adaboost

# caret ile Adaboost
adaboost_ctrl <- trainControl(method = "cv", number = 2, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(123)
adaboost_model <- train(
  diagnosis ~ ., 
  data = train,                   # ??l??eklenmemi?? orijinal train seti
  method = "AdaBoost.M1",            # caret i??indeki adaboost metodu
  trControl = adaboost_ctrl,      # cross-validation ve ROC i??in ayarlar
  metric = "ROC"                  # optimize etmek istedi??imiz metrik
)

# Modelin en iyi parametreleri
cat("Caret Adaboost en iyi parametreler:\n")
print(adaboost_model$bestTune)

# Test
adaboost_pred_class <- predict(adaboost_model, newdata = test)
adaboost_pred_prob <- predict(adaboost_model, newdata = test, type = "prob")

# Confusion Matrix
adaboost_conf <- confusionMatrix(adaboost_pred_class, test$diagnosis)
cat("\nCaret Adaboost Performans??:\n")
print(adaboost_conf)

# ROC ve AUC
roc_adaboost <- roc(response = test$diagnosis,
                    predictor = adaboost_pred_prob[, levels(test$diagnosis)[2]],
                    levels = rev(levels(test$diagnosis)))
auc_adaboost <- auc(roc_adaboost)
cat("Caret Adaboost AUC:", auc_adaboost, "\n")
plot(roc_adaboost, main = paste("Caret Adaboost ROC E??risi - AUC:", round(auc_adaboost, 3)))

## Gradient Boosting

# trainControl ayarlar??
gbm_ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Parametre performansa etkisi
gbm_grid <- expand.grid(
  n.trees = seq(50, 200, by = 50),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

# Train
gbm_model <- train(
  diagnosis ~ .,
  data = train,
  method = "gbm",
  trControl = gbm_ctrl,
  tuneGrid = gbm_grid,
  metric = "ROC",
  verbose = FALSE
)

# En iyi parametreler
cat("En iyi parametreler:\n")
print(gbm_model$bestTune)

# Test
gbm_pred_class <- predict(gbm_model, newdata = test)
gbm_pred_prob <- predict(gbm_model, newdata = test, type = "prob")

# Confusion matrix
gbm_conf <- confusionMatrix(gbm_pred_class, test$diagnosis)
cat("\nGradient Boosting Performans??:\n")
print(gbm_conf)

# ROC ve AUC
roc_gbm <- roc(response = test$diagnosis,
               predictor = gbm_pred_prob[, levels(test$diagnosis)[2]],
               levels = rev(levels(test$diagnosis)))
auc_gbm <- auc(roc_gbm)
cat("Gradient Boosting AUC:", auc_gbm, "\n")
plot(roc_gbm, main = paste("Gradient Boosting ROC E??risi - AUC:", round(auc_gbm, 3)))

## XGBoost

#xgboost 0-1 aral?????? istedi??inden ba????ml??y?? numerik yapt??m
train_labels_numeric <- ifelse(train_labels == levels(train_labels)[2], 1, 0)
test_labels_numeric <- ifelse(test_labels == levels(test_labels)[2], 1, 0)

# matrix fotmar?? gerekiyor
dtrain <- xgb.DMatrix(data = as.matrix(train_features), label = train_labels_numeric)
dtest <- xgb.DMatrix(data = as.matrix(test_features), label = test_labels_numeric)

# Parametreler
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0
)

# Train
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Test
xgb_pred_prob <- predict(xgb_model, newdata = as.matrix(test_features))

# S??n??f tahmini (0.5 e??ik)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)

# Fakt??r
xgb_pred_factor <- factor(xgb_pred, levels = c(0,1), labels = levels(train_labels))

# Confusion matrix
conf_xgb <- confusionMatrix(xgb_pred_factor, test_labels)
print(conf_xgb)

# ROC ve AUC
roc_xgb <- roc(response = test_labels, predictor = xgb_pred_prob, levels = rev(levels(test_labels)))
auc_xgb <- auc(roc_xgb)
cat("XGBoost AUC:", auc_xgb, "\n")
plot(roc_xgb, main = paste("XGBoost ROC E??risi - AUC:", round(auc_xgb, 3)))


################## ------------------------#######################


#PCA UYGULAMASI SONRASI MODELLER??N UYGULAMASI

install.packages("factoextra")
library(factoextra)
# Train-test veri seti ayr??m??
dataset <- cancerdata1
validationIndex <- createDataPartition(cancerdata1$diagnosis, p=0.70, list=FALSE)
set.seed(123)
train <- dataset[validationIndex,]
test <- dataset[-validationIndex,]

# 1. Ba????ms??z de??i??kenleri ay??r ve scale et
train_features <- train[, -which(names(train) == "diagnosis")]
test_features  <- test[, -which(names(test) == "diagnosis")]

# Standardizasyon
train_scaled <- scale(train_features)
test_scaled <- scale(test_features, 
                     center = attr(train_scaled, "scaled:center"), 
                     scale = attr(train_scaled, "scaled:scale"))

# 2. PCA Uygula
pca <- prcomp(train_scaled, center = FALSE, scale. = FALSE)
summary(pca)  # Varyans a????klama oranlar??na bak

# Bile??en say??s??n?? se?? (??rne??in, %95 varyans?? a????klayan bile??en say??s??)
explained_var <- cumsum(pca$sdev^2) / sum(pca$sdev^2)
num_components <- which(explained_var >= 0.80)[1]
cat("Se??ilen PCA bile??en say??s??:", num_components, "\n")

# PCA d??n??????m??
train_pca <- as.data.frame(pca$x[, 1:num_components])
test_pca <- as.data.frame(predict(pca, newdata = test_scaled)[, 1:num_components])

# Etiketleri tekrar ekle
train_pca$diagnosis <- train$diagnosis
test_pca$diagnosis <- test$diagnosis

pca_result <- prcomp(scaled_df, scale. = TRUE)
fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50))
fviz_pca_var(pca_result,
             col.var = "contrib", # De??i??ken katk??lar??na g??re renk
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), # Renk skalas??
             repel = TRUE,        # Etiketlerin ??st ??ste binmesini engelle
             title = "PCA De??i??ken Biplotu"
)


##############PCA Sonras?? Knn###############
library(class)
library(caret)
library(pROC)

# 1. Etiketleri ay??r
train_labels <- train_pca$diagnosis
test_labels  <- test_pca$diagnosis

# 2. Etiket s??tununu kald??rarak sadece PCA bile??enlerini kullan
train_pca_features <- train_pca[, -which(names(train_pca) == "diagnosis")]
test_pca_features  <- test_pca[, -which(names(test_pca) == "diagnosis")]

# 3. Cross-validation ile en iyi k de??erini bul
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
grid <- expand.grid(k = seq(1, 30, by = 2))

knn_cv <- train(x = train_pca_features,
                y = train_labels,
                method = "knn",
                tuneGrid = grid,
                trControl = ctrl)

# En iyi k
best_k <- knn_cv$bestTune$k
cat("En iyi k degeri:", best_k, "\n")

# 4. Test verisi ile modeli de??erlendir
predictions <- knn(train = train_pca_features,
                   test = test_pca_features,
                   cl = train_labels,
                   k = best_k)

confusion <- confusionMatrix(predictions, test_labels)
print(confusion)

# 5. ROC ve AUC
pred_probs <- predict(knn_cv, newdata = test_pca_features, type = "prob")

# Pozitif s??n??f?? (??rne??in "M" veya "B") belirle
pos_class <- levels(test_labels)[2]
roc_obj <- roc(response = test_labels,
               predictor = pred_probs[, pos_class],
               levels = rev(levels(test_labels)))

auc_value <- auc(roc_obj)
cat("AUC degeri:", auc_value, "\n")
plot(roc_obj, main = paste("ROC Egrisi - AUC:", round(auc_value, 3)))

# ??zel mesafe fonksiyonlar??
euclidean_dist <- function(a, b) sqrt(sum((a - b)^2))
manhattan_dist <- function(a, b) sum(abs(a - b))
minkowski_dist <- function(a, b, p = 3) sum(abs(a - b)^p)^(1/p)

# ??zel k-NN fonksiyonu
knn_custom <- function(train_data, test_data, train_labels, k, dist_fun, p = NULL) {
  n_test <- nrow(test_data)
  pred <- vector("character", n_test)
  
  for (i in 1:n_test) {
    dists <- numeric(nrow(train_data))
    for (j in 1:nrow(train_data)) {
      if (!is.null(p)) {
        dists[j] <- dist_fun(train_data[j, ], test_data[i, ], p)
      } else {
        dists[j] <- dist_fun(train_data[j, ], test_data[i, ])
      }
    }
    neighbors <- order(dists)[1:k]
    votes <- train_labels[neighbors]
    pred[i] <- names(sort(table(votes), decreasing = TRUE))[1]
  }
  factor(pred, levels = levels(train_labels))
}

# Performans kar????la??t??rma
results <- list()

# 1. Euclidean
pred_euc <- knn_custom(train_pca_features, test_pca_features, train_labels, best_k, euclidean_dist)
conf_euc <- confusionMatrix(pred_euc, test_labels)
results$Euclidean <- conf_euc

# 2. Manhattan
pred_man <- knn_custom(train_pca_features, test_pca_features, train_labels, best_k, manhattan_dist)
conf_man <- confusionMatrix(pred_man, test_labels)
results$Manhattan <- conf_man

# 3. Minkowski (p=3)
pred_mink <- knn_custom(train_pca_features, test_pca_features, train_labels, best_k, minkowski_dist, p = 3)
conf_mink <- confusionMatrix(pred_mink, test_labels)
results$Minkowski_p3 <- conf_mink

# Sonu??lar?? yazd??r
for (metric in names(results)) {
  cat("\nMesafe metri??i:", metric, "\n")
  print(results[[metric]])
}

####################### PCA Sonras?? SVM#####################33333
library(e1071)
library(caret)

# 1. Etiketleri ay??r
train_labels <- train_pca$diagnosis
test_labels  <- test_pca$diagnosis

# Sadece PCA bile??enlerini al
train_pca_features <- train_pca[, -which(names(train_pca) == "diagnosis")]
test_pca_features  <- test_pca[, -which(names(test_pca) == "diagnosis")]

# 2. Farkl?? kernel t??rleri ile SVM modelleri
kernels <- c("linear", "radial", "polynomial")
svm_results <- list()
accuracies <- numeric(length(kernels))

for (i in seq_along(kernels)) {
  kernel <- kernels[i]
  svm_model <- svm(x = train_pca_features, y = train_labels, kernel = kernel, probability = TRUE)
  pred <- predict(svm_model, test_pca_features)
  conf <- confusionMatrix(pred, test_labels)
  
  svm_results[[kernel]] <- list(model = svm_model, confusion = conf)
  accuracies[i] <- conf$overall['Accuracy']
  
  cat("\nKernel:", kernel, "\n")
  print(conf)
}

# 3. En iyi kernel se??im
best_kernel <- kernels[which.max(accuracies)]
cat("\nEn iyi kernel:", best_kernel, "Accuracy:", round(max(accuracies), 4), "\n")
# 4. Tuning i??lemi
if (best_kernel == "linear") {
  tune_result <- tune(svm,
                      train.x = train_pca_features,
                      train.y = train_labels,
                      kernel = best_kernel,
                      ranges = list(cost = seq(0.1, 3, by = 0.1)))
} else {
  tune_result <- tune(svm,
                      train.x = train_pca_features,
                      train.y = train_labels,
                      kernel = best_kernel,
                      ranges = list(
                        cost = seq(0.1, 3, by = 0.1),
                        gamma = seq(0.1, 0.5, by = 0.05)
                      ))
}

# 5. En iyi parametreleri al
best_params <- tune_result$best.parameters
cat("En iyi parametreler:\n")
print(best_params)

# 6. Final modeli kur
final_model <- svm(x = train_pca_features,
                   y = train_labels,
                   kernel = best_kernel,
                   cost = best_params$cost,
                   gamma = ifelse(is.null(best_params$gamma), 0.1, best_params$gamma),
                   probability = TRUE)

final_pred <- predict(final_model, test_pca_features)
final_conf <- confusionMatrix(final_pred, test_labels)

cat("\nFinal Model Performans??:\n")
print(final_conf)

################## PCA Sonras?? Lojistik #################  

library(pROC)
library(caret)

# 1. PCA veri setini data.frame olarak haz??rla
train_df <- as.data.frame(train_pca)
test_df  <- as.data.frame(test_pca)

train_labels <- train_df$diagnosis
test_labels  <- test_df$diagnosis

# 2. Lojistik regresyon modeli kur (PCA bile??enlerine g??re)
log_model <- glm(diagnosis ~ ., data = train_df, family = binomial)

# 3. Test seti ??zerinde tahmin yap
log_probs <- predict(log_model, newdata = test_df, type = "response")

# 4. S??n??fland??rma (e??ik 0.5)
log_pred <- ifelse(log_probs > 0.5, levels(train_labels)[2], levels(train_labels)[1])
log_pred <- factor(log_pred, levels = levels(test_labels))

# 5. Confusion Matrix
log_conf <- confusionMatrix(log_pred, test_labels)
cat("\nLogistic Regression Performans?? (PCA sonras??):\n")
print(log_conf)

# 6. ROC ve AUC
roc_log <- roc(response = test_labels, predictor = log_probs, levels = rev(levels(test_labels)))
auc_log <- auc(roc_log)
cat("Logistic Regression AUC:", auc_log, "\n")
plot(roc_log, main = paste("Logistic ROC E??risi - AUC:", round(auc_log, 3)))


######################## PCA Sonras?? Random forest######################3
library(randomForest)

# Modeli kur
rf_model <- randomForest(diagnosis ~ ., data = train_pca, ntree = 500, 
                         mtry = floor(sqrt(ncol(train_pca) - 1)), importance = TRUE)

# Tahmin
rf_pred <- predict(rf_model, newdata = test_pca)

# Confusion Matrix
rf_conf <- confusionMatrix(rf_pred, test_pca$diagnosis)
cat("\nRandom Forest Performans??:\n")
print(rf_conf)

# ROC ve AUC
rf_probs <- predict(rf_model, newdata = test_pca, type = "prob")
roc_rf <- roc(response = test_pca$diagnosis,
              predictor = rf_probs[, levels(test_pca$diagnosis)[2]],
              levels = rev(levels(test_pca$diagnosis)))
auc_rf <- auc(roc_rf)
cat("Random Forest AUC:", auc_rf, "\n")
plot(roc_rf, main = paste("Random Forest ROC E??risi - AUC:", round(auc_rf, 3)))

######################## PCA Sonras?? Adaboost #################
library(adabag)

## Adaboost

# caret ile Adaboost
adaboost_ctrl <- trainControl(method = "cv", number = 2, classProbs = TRUE, summaryFunction = twoClassSummary)

set.seed(123)
adaboost_model <- train(
  diagnosis ~ ., 
  data = train_pca,                   # ??l??eklenmemi?? orijinal train seti
  method = "AdaBoost.M1",            # caret i??indeki adaboost metodu
  trControl = adaboost_ctrl,      # cross-validation ve ROC i??in ayarlar
  metric = "ROC"                  # optimize etmek istedi??imiz metrik
)

# Modelin en iyi parametreleri
cat("Caret Adaboost en iyi parametreler:\n")
print(adaboost_model$bestTune)

# Test
adaboost_pred_class <- predict(adaboost_model, newdata = test_pca)
adaboost_pred_prob <- predict(adaboost_model, newdata = test_pca, type = "prob")

# Confusion Matrix
adaboost_conf <- confusionMatrix(adaboost_pred_class, test_pca$diagnosis)
cat("\nCaret Adaboost Performans??:\n")
print(adaboost_conf)

# ROC ve AUC
roc_adaboost <- roc(response = test$diagnosis,
                    predictor = adaboost_pred_prob[, levels(test$diagnosis)[2]],
                    levels = rev(levels(test$diagnosis)))
auc_adaboost <- auc(roc_adaboost)
cat("Caret Adaboost AUC:", auc_adaboost, "\n")
plot(roc_adaboost, main = paste("Caret Adaboost ROC E??risi - AUC:", round(auc_adaboost, 3)))


###################Gradient Boosting (GBM) (PCA sonras??)##########33
library(gbm)
library(caret)

# TrainControl
gbm_ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

# Grid
gbm_grid <- expand.grid(
  n.trees = seq(50, 200, by = 50),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = 10
)

# Model
gbm_model <- train(
  diagnosis ~ ., data = train_pca,
  method = "gbm",
  trControl = gbm_ctrl,
  tuneGrid = gbm_grid,
  metric = "ROC",
  verbose = FALSE
)

# En iyi parametreler
cat("En iyi parametreler:\n")
print(gbm_model$bestTune)

# Tahmin
gbm_pred_class <- predict(gbm_model, newdata = test_pca)
gbm_pred_prob <- predict(gbm_model, newdata = test_pca, type = "prob")

# Confusion matrix
gbm_conf <- confusionMatrix(gbm_pred_class, test_pca$diagnosis)
cat("\nGradient Boosting Performans??:\n")
print(gbm_conf)

# ROC ve AUC
roc_gbm <- roc(response = test_pca$diagnosis,
               predictor = gbm_pred_prob[, levels(test_pca$diagnosis)[2]],
               levels = rev(levels(test_pca$diagnosis)))
auc_gbm <- auc(roc_gbm)
cat("Gradient Boosting AUC:", auc_gbm, "\n")
plot(roc_gbm, main = paste("Gradient Boosting ROC E??risi - AUC:", round(auc_gbm, 3)))

############################## XGBoost (PCA sonras??)##############
library(xgboost)

# Label'lar?? binary yap
train_labels_numeric <- ifelse(train_pca$diagnosis == levels(train_pca$diagnosis)[2], 1, 0)
test_labels_numeric <- ifelse(test_pca$diagnosis == levels(test_pca$diagnosis)[2], 1, 0)

# XGBoost format??na sok
dtrain <- xgb.DMatrix(data = as.matrix(train_pca[, -ncol(train_pca)]), label = train_labels_numeric)
dtest  <- xgb.DMatrix(data = as.matrix(test_pca[, -ncol(test_pca)]), label = test_labels_numeric)

# Parametreler
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  gamma = 0
)

# Train
set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  print_every_n = 10
)

# Tahmin
xgb_pred_prob <- predict(xgb_model, newdata = dtest)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)
xgb_pred_factor <- factor(xgb_pred, levels = c(0, 1), labels = levels(train_pca$diagnosis))

# Confusion matrix
conf_xgb <- confusionMatrix(xgb_pred_factor, test_pca$diagnosis)
cat("\nXGBoost Performans??:\n")
print(conf_xgb)

# ROC ve AUC
roc_xgb <- roc(response = test_pca$diagnosis, predictor = xgb_pred_prob, levels = rev(levels(test_pca$diagnosis)))
auc_xgb <- auc(roc_xgb)
cat("XGBoost AUC:", auc_xgb, "\n")
plot(roc_xgb, main = paste("XGBoost ROC E??risi - AUC:", round(auc_xgb, 3)))



