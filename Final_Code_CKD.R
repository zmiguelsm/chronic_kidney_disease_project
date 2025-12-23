# ****************************************************************************
# COMPREHENSIVE CHRONIC KIDNEY DISEASE (CKD) ANALYSIS AND MODELING SCRIPT
# ****************************************************************************

# ===========================================================================
# Step 0: Install & Load Required Packages
# ===========================================================================

# Uncomment and run these lines if packages are not installed
install.packages(c("tidyverse", "caret", "randomForest", "ROSE", "xgboost", 
                   "pROC", "ggplot2", "gridExtra", "e1071", "dplyr", "broom", 
                   "recipes", "themis", "Matrix", "smotefamily"))

# Load all required packages
library(tidyverse)  # For data manipulation and visualization
library(caret)      # For model training and evaluation
library(randomForest) # For random forest models
library(ROSE)       # For SMOTE implementation via ovun.sample
library(xgboost)    # For XGBoost models
library(pROC)       # For ROC curve analysis
library(ggplot2)    # For visualization
library(gridExtra)  # For arranging multiple plots
library(e1071)      # For skewness calculations
library(dplyr)      # For data manipulation
library(broom)      # For tidying model output
library(Matrix)     # For sparse matrix operations
library(smotefamily) # For SMOTE implementation
library(recipes)
library(themis)

# Set seed for reproducibility across all models
set.seed(530)

# ===========================================================================
# Step 1: Data Preparation and Exploratory Analysis (Speaker: Ali)
# ===========================================================================

# Load the dataset
data <- read.csv("CKD.csv", na.strings = c("?", "", "NA"))

# View structure and summary of the data
str(data)
#Variable names:
# PatientID, Age, Gender, Ethnicity, SocioeconomicStatus, 
#EducationLevel, BMI, Smoking, AlcoholConsumption, PhysicalActivity, 
#DietQuality, SleepQuality, FamilyHistoryKidneyDisease, FamilyHistoryHypertension, 
#FamilyHistoryDiabetes, PreviousAcuteKidneyInjury, UrinaryTractInfections, SystolicBP, DiastolicBP, 
#FastingBloodSugar, HbA1c, SerumCreatinine, BUNLevels, GFR, ProteinInUrine, ACR, 
#SerumElectrolytesSodium, SerumElectrolytesPotassium, SerumElectrolytesCalcium, SerumElectrolytesPhosphorus, 
#HemoglobinLevels, CholesterolTotal, CholesterolLDL, CholesterolHDL, 
#CholesterolTriglycerides, ACEInhibitors, Diuretics, NSAIDsUse, 
#Statins, AntidiabeticMedications, Edema, FatigueLevels, NauseaVomiting, MuscleCramps, Itching, 
#QualityOfLifeScore, HeavyMetalsExposure, OccupationalExposureChemicals, WaterQuality, MedicalCheckupsFrequency, MedicationAdherence, 
#HealthLiteracy, Diagnosis, DoctorInCharge

summary(data)

# Check missing values in each column
missing_values <- colSums(is.na(data))
print(missing_values)

# Create target variable (1 = CKD, 0 = Not CKD)
unique(data$Diagnosis)
data$target <- as.factor(data$Diagnosis)

# Define categorical variables
categorical_vars <- c(
  "Gender", "Ethnicity", "SocioeconomicStatus", "EducationLevel",
  "Smoking", "FamilyHistoryKidneyDisease", "FamilyHistoryHypertension",
  "FamilyHistoryDiabetes", "PreviousAcuteKidneyInjury", "UrinaryTractInfections",
  "ACEInhibitors", "Diuretics", "Statins", "AntidiabeticMedications",
  "Edema", "HeavyMetalsExposure", "OccupationalExposureChemicals",
  "WaterQuality", "Diagnosis"
)

# Convert categorical variables to factors
data[categorical_vars] <- lapply(data[categorical_vars], factor)

# Remove unnecessary columns
data <- data %>% 
  select(-PatientID, -DoctorInCharge, -Diagnosis)

# Check class distribution
table(data$target)

# Visualize class distribution
count_df <- data %>%
  count(target) %>%
  mutate(target = factor(target, levels = c(0, 1), labels = c("Not CKD", "CKD")))

# Plot with value labels
ggplot(count_df, aes(x = target, y = n, fill = target)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = n), vjust = -0.5, size = 3) +
  scale_fill_manual(values = c("Not CKD" = "skyblue", "CKD" = "tomato")) +
  labs(
    title = "Class Distribution of CKD Diagnosis",
    x = "Diagnosis",
    y = "Count",
    fill = "Class"
  ) +
  theme_minimal(base_size = 14)

# Create both 80-20 and 50-50 train-test splits
# 80-20 Split
train_index_80 <- createDataPartition(data$target, p = 0.8, list = FALSE)
data.train.80 <- data[train_index_80, ]
data.test.80 <- data[-train_index_80, ]

# 50-50 Split
train_index_50 <- createDataPartition(data$target, p = 0.5, list = FALSE)
data.train.50 <- data[train_index_50, ]
data.test.50 <- data[-train_index_50, ]

# ===========================================================================
# Step 2: Logistic Regression Models (Speaker: Shylendra)
# ===========================================================================

# ------------ 80-20 Split Logistic Regression Model ------------

glm_model <- glm(target ~ ., data = data.train.80, family = binomial)

# Summary of the model
summary(glm_model)

#Dropping irrelevant variables
simplified_glm_model <- glm(
  target ~ Gender + EducationLevel + DietQuality + SystolicBP + DiastolicBP +
    FastingBloodSugar + SerumCreatinine + BUNLevels + GFR +
    ProteinInUrine + SerumElectrolytesSodium + Edema + MuscleCramps + Itching,
  data = data.train.80,
  family = binomial
)

summary(simplified_glm_model)

# Predict on test data
probs <- predict(simplified_glm_model, newdata = data.test.80, type = "response")
preds <- ifelse(probs > 0.5, 1, 0)

# Evaluate
confusionMatrix(as.factor(preds), data.test.80$target, positive = "1")

# ------------ 50-50 Split Logistic Regression Model ------------

glm_model_2 <- glm(target ~ ., data = data.train.50, family = binomial)

# Summary of the model
summary(glm_model_2)

#Dropping irrelevant variables
simplified_glm_model_2 <- glm(
  target ~ Gender + EducationLevel + DietQuality + SystolicBP + DiastolicBP +
    FastingBloodSugar + SerumCreatinine + BUNLevels + GFR +
    ProteinInUrine + SerumElectrolytesSodium + Edema + MuscleCramps + Itching,
  data = data.train.50,
  family = binomial
)

summary(simplified_glm_model_2)

# Predict on test data
probs <- predict(simplified_glm_model_2, newdata = data.test.50, type = "response")
preds <- ifelse(probs > 0.5, 1, 0)

# Evaluate
confusionMatrix(as.factor(preds), data.test.50$target, positive = "1")

# ---------- Plots for Logistic Regression Model --------------

# Plot ROC
roc_obj <- roc(as.numeric(as.character(data.test.50$target)), probs)
plot(roc_obj, col = "blue", main = "ROC Curve")
auc(roc_obj)  # Area under the curve

# Confusion Matrix Heatmap
cm <- confusionMatrix(as.factor(preds), data.test.50$target, positive = "1")
cm_df <- as.data.frame(cm$table)

ggplot(cm_df, aes(Prediction, Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  geom_text(aes(label = Freq), vjust = 1.5, size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix", x = "Predicted", y = "Actual") +
  theme_minimal()

# Coefficient Log-odds
coef_df <- tidy(simplified_glm_model_2) %>%
  filter(term != "(Intercept)") %>%
  mutate(term = reorder(term, estimate))

ggplot(coef_df, aes(x = estimate, y = term)) +
  geom_point(color = "red") +
  geom_vline(xintercept = 0, linetype = "dashed") +
  labs(title = "Logistic Regression Coefficients",
       x = "Estimate (log-odds)", y = "Feature") +
  theme_minimal()


# ===========================================================================
# Step 3: Random Forest Models (without SMOTE) (Speaker: Jose)
# ===========================================================================

# ------------ 80-20 Split Random Forest Model (No SMOTE) ------------
rf.class <- randomForest(target ~ .,
                         data=data.train.80, mtry=round(sqrt(ncol(data.train.80) - 1)), 
                         importance=TRUE)

importance(rf.class)
varImpPlot(rf.class)

p <- ncol(data.train.80) - 1
oob.error.class <- double(p) #initialize empty vector

for(m in 1:p) {
  fit <- randomForest(target ~ ., data=data.train.80, mtry=m, ntree=175)
  conf.mat <- fit$err.rate[175]
  oob.error.class[m] <- fit$err.rate[175, 'OOB']
} 

matplot(1:p, oob.error.class, pch=19, col="red", type="b", ylab="Misclassification Error", xlab="mtry")

best.mtry <- which.min(oob.error.class)

#Getting the best RF
rf.class.best <- randomForest(target ~ .,
                              data=data.train.80, mtry= best.mtry,
                              importance=TRUE,xtest = data.test.80[ ,-52],
                              ytest=data.test.80$target)

rf.class.best #we can see that the class.error for 0, is very high! Why, class imbalance

# Extract predictions from the model
preds <- rf.class.best$test$predicted

# Compute the confusion matrix
caret::confusionMatrix(as.factor(preds),
                       as.factor(data.test.80$target),
                       positive = "1")

# ------------ 50-50 Split Random Forest Model (No SMOTE) ------------
rf.class <- randomForest(target ~ .,
                         data=data.train.50, mtry=round(sqrt(ncol(data.train.50) - 1)), 
                         importance=TRUE)

importance(rf.class)
varImpPlot(rf.class)

p <- ncol(data.train.50) - 1
oob.error.class <- double(p) #initialize empty vector

for(m in 1:p) {
  fit <- randomForest(target ~ ., data=data.train.50, mtry=m, ntree=175)
  conf.mat <- fit$err.rate[175]
  oob.error.class[m] <- fit$err.rate[175, 'OOB']
} 

matplot(1:p, oob.error.class, pch=19, col="red", type="b", ylab="Misclassification Error", xlab="mtry")

best.mtry <- which.min(oob.error.class)

#Getting the best RF
rf.class.best <- randomForest(target ~ .,
                              data=data.train.50, mtry= best.mtry,
                              importance=TRUE,xtest = data.test.50[ ,-52],
                              ytest=data.test.50$target)

rf.class.best #we can see that the class.error for 0, is very high! Why, class imbalance

# Extract predictions from the model
preds <- rf.class.best$test$predicted

# Compute the confusion matrix
caret::confusionMatrix(as.factor(preds),
                       as.factor(data.test.50$target),
                       positive = "1")

# Load necessary package
library(pROC)

# Get the predicted probabilities for class "1"
probs <- rf.class.best$test$votes[, "1"]

# Compute the ROC curve and AUC
roc_obj <- roc(response = data.test.50$target, predictor = probs)

# Plot the ROC curve
plot(roc_obj, col = "blue", main = "ROC Curve for Random Forest")

# Print the AUC value
auc(roc_obj)
# ------------ 80-20 Split Random Forest Model (SMOTE) ------------

## STEP 1: Create the recipe
# rec <- recipes::recipe(target ~ ., data = data.train.80) %>%
## One-hot encode them EXCEPT the outcome
#  step_dummy(all_nominal(), -all_outcomes()) %>%
## Apply SMOTE (balanced to 1:1)
#  step_smote(target, over_ratio = 1) %>%
#  recipes::prep()

## STEP 2: Get the SMOTE'd training data
# data.train.80.smote <- juice(rec)

## STEP 3: One-hot encode test set the same way
# data.test.80.encoded <- bake(rec, new_data = data.test.80)

## STEP 4: Train Random Forest
# rf.class.smote <- randomForest(target ~ ., 
#                               data = data.train.80.smote,
#                               mtry = best.mtry,
#                               importance = TRUE,
#                               xtest = data.test.80.encoded[, -which(names(data.test.80.encoded) == "target")],
#                               ytest = data.test.80.encoded$target)

# rf.class.smote

## Extract predictions from the model
# preds <- rf.class.smote$test$predicted

## Compute the confusion matrix
# caret::confusionMatrix(as.factor(preds),
#                       as.factor(data.test.80.encoded$target),
#                       positive = "1")

# ------------ 50-50 Split Random Forest Model (SMOTE) ------------

## STEP 1: Create the recipe
# rec <- recipes::recipe(target ~ ., data = data.train.50) %>%
## One-hot encode them EXCEPT the outcome
#  step_dummy(all_nominal(), -all_outcomes()) %>%
## Apply SMOTE (balanced to 1:1)
#  step_smote(target, over_ratio = 1) %>%
#  recipes::prep()

## STEP 2: Get the SMOTE'd training data
# data.train.50.smote <- juice(rec)

## STEP 3: One-hot encode test set the same way
# data.test.50.encoded <- bake(rec, new_data = data.test.50)

## STEP 4: Train Random Forest
# rf.class.smote <- randomForest(target ~ ., 
#                               data = data.train.50.smote,
#                               mtry = best.mtry,
#                               importance = TRUE,
#                               xtest = data.test.50.encoded[, -which(names(data.test.50.encoded) == "target")],
#                               ytest = data.test.50.encoded$target)

#rf.class.smote

## Extract predictions from the model
# preds <- rf.class.smote$test$predicted

## Compute the confusion matrix
# caret::confusionMatrix(as.factor(preds),
#                       as.factor(data.test.50.encoded$target),
#                       positive = "1")


# ===========================================================================
# Step 5: XGBoost Model (with SMOTE + Hyperparameter Tuning)
# ===========================================================================

# ------------ 80-20 Split XGBoost Model ------------

# Convert training data for XGBoost
# First, convert factor variables to numeric with one-hot encoding
dummies_model_80 <- dummyVars(~ ., data = data.train.80 %>% select(-target))
train_features_80 <- predict(dummies_model_80, newdata = data.train.80 %>% select(-target)) %>% as.data.frame()
test_features_80 <- predict(dummies_model_80, newdata = data.test.80 %>% select(-target)) %>% as.data.frame()

# Convert target to numeric for SMOTE
train_labels_80 <- as.numeric(data.train.80$target) - 1  # 0/1 format for SMOTE

# Apply SMOTE
smote_data_80 <- SMOTE(X = train_features_80, target = train_labels_80, K = 5)
train_features_80_smote <- smote_data_80$data[, -ncol(smote_data_80$data)]
train_labels_80_smote <- smote_data_80$data$class

# Check class distribution after SMOTE
table(train_labels_80_smote)

# Convert data to XGBoost format
dtrain_80 <- xgb.DMatrix(data = as.matrix(train_features_80_smote), 
                         label = train_labels_80_smote)
dtest_80 <- xgb.DMatrix(data = as.matrix(test_features_80), 
                        label = as.numeric(data.test.80$target) - 1)

# Set XGBoost parameters
xgb_params_80 <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train XGBoost model
xgb_model_80 <- xgb.train(
  params = xgb_params_80,
  data = dtrain_80,
  nrounds = 100,
  watchlist = list(train = dtrain_80, test = dtest_80),
  early_stopping_rounds = 10,
  verbose = 1
)

# Feature importance
importance_matrix_80 <- xgb.importance(feature_names = colnames(train_features_80_smote), 
                                       model = xgb_model_80)
xgb.plot.importance(importance_matrix_80[1:10,])

# Predict on test data
xgb_probs_80 <- predict(xgb_model_80, as.matrix(test_features_80))
xgb_preds_80 <- ifelse(xgb_probs_80 > 0.5, 1, 0)

# Evaluate model performance
xgb_cm_80 <- confusionMatrix(as.factor(xgb_preds_80), data.test.80$target, positive = "1")
print(xgb_cm_80)

# ROC curve and AUC
xgb_roc_80 <- roc(response = as.numeric(as.character(data.test.80$target)), 
                  predictor = xgb_probs_80)
xgb_auc_80 <- auc(xgb_roc_80)

# Plot ROC curve
plot(xgb_roc_80, col = "orange", lwd = 2, 
     main = "ROC Curve - XGBoost with SMOTE (80-20 Split)")
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.6, 0.2, labels = paste("AUC =", round(xgb_auc_80, 3)), 
     col = "black", cex = 1.2)


# ------------ 50-50 Split XGBoost Model ------------

# Convert training data for XGBoost
# First, convert factor variables to numeric with one-hot encoding
dummies_model_50 <- dummyVars(~ ., data = data.train.50 %>% select(-target))
train_features_50 <- predict(dummies_model_50, newdata = data.train.50 %>% select(-target)) %>% as.data.frame()
test_features_50 <- predict(dummies_model_50, newdata = data.test.50 %>% select(-target)) %>% as.data.frame()

# Convert target to numeric for SMOTE
train_labels_50 <- as.numeric(data.train.50$target) - 1  # 0/1 format for SMOTE

# Apply SMOTE
smote_data_50 <- SMOTE(X = train_features_50, target = train_labels_50, K = 5)
train_features_50_smote <- smote_data_50$data[, -ncol(smote_data_50$data)]
train_labels_50_smote <- smote_data_50$data$class

# Check class distribution after SMOTE
table(train_labels_50_smote)

# Convert data to XGBoost format
dtrain_50 <- xgb.DMatrix(data = as.matrix(train_features_50_smote), 
                         label = train_labels_50_smote)
dtest_50 <- xgb.DMatrix(data = as.matrix(test_features_50), 
                        label = as.numeric(data.test.50$target) - 1)

# Set XGBoost parameters
xgb_params_50 <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train XGBoost model
xgb_model_50 <- xgb.train(
  params = xgb_params_50,
  data = dtrain_50,
  nrounds = 100,
  watchlist = list(train = dtrain_50, test = dtest_50),
  early_stopping_rounds = 10,
  verbose = 1
)

# Feature importance
importance_matrix_50 <- xgb.importance(feature_names = colnames(train_features_50_smote), 
                                       model = xgb_model_50)
xgb.plot.importance(importance_matrix_50[1:10,])

# Predict on test data
xgb_probs_50 <- predict(xgb_model_50, as.matrix(test_features_50))
xgb_preds_50 <- ifelse(xgb_probs_50 > 0.5, 1, 0)

# Evaluate model performance
xgb_cm_50 <- confusionMatrix(as.factor(xgb_preds_50), data.test.50$target, positive = "1")
print(xgb_cm_50)

# ROC curve and AUC
xgb_roc_50 <- roc(response = as.numeric(as.character(data.test.50$target)), 
                  predictor = xgb_probs_50)
xgb_auc_50 <- auc(xgb_roc_50)

# Plot ROC curve
plot(xgb_roc_50, col = "orange", lwd = 2, 
     main = "ROC Curve - XGBoost with SMOTE (50-50 Split)")
abline(a = 0, b = 1, lty = 2, col = "gray")
text(0.6, 0.2, labels = paste("AUC =", round(xgb_auc_50, 3)), 
     col = "black", cex = 1.2)