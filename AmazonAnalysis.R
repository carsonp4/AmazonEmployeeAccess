# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)

# Reading in Data
setwd("~/Desktop/Stat348/AmazonEmployeeAccess/")
train <- vroom("train.csv")
test <- vroom("test.csv")
train$ACTION <- as.factor(train$ACTION)

# Visualizations
#library(ggmosaic)
#train$RESOURCE <- as.factor(train$RESOURCE)
#train$ACTION <- as.factor(train$ACTION)
#ggplot(data=train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))

# Prepping data

#ACTION	ACTION is 1 if the resource was approved, 0 if the resource was not
#RESOURCE	An ID for each resource
#MGR_ID	The EMPLOYEE ID of the manager of the current EMPLOYEE ID record; an employee may have only one manager at a time
#ROLE_ROLLUP_1	Company role grouping category id 1 (e.g. US Engineering)
#ROLE_ROLLUP_2	Company role grouping category id 2 (e.g. US Retail)
#ROLE_DEPTNAME	Company role department description (e.g. Retail)
#ROLE_TITLE	Company role business title description (e.g. Senior Engineering Retail Manager)
#ROLE_FAMILY_DESC	Company role family extended description (e.g. Retail Manager, Software Engineering)
#ROLE_FAMILY	Company role family description (e.g. Retail Manager)
#ROLE_CODE	Company role code; this code is unique to each role (e.g. Manager)


my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_factor_predictors(), threshold = .005) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))  #target encoding

  
bake(prep(my_recipe), new_data = train)


# Logistic Regression -----------------------------------------------------

log_mod <- logistic_reg() %>% #Type of model
  set_engine("glm")

log_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(log_mod) %>%
  fit(data = train) # Fit the workflow

log_preds <- predict(log_wf,
                     new_data=test,
                     type="prob") # "class" or "prob" (see doc)

log_submit <- as.data.frame(cbind(test$id, log_preds$.pred_1))
colnames(log_submit) <- c("id", "ACTION")
write_csv(log_submit, "log_submit.csv")

# Penalized Logistic Regression -------------------------------------------

penlog_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penlog_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(penlog_mod)

penlog_tuning_grid <- grid_regular(penalty(),
                                   mixture(),
                                   levels = 10)
folds <- vfold_cv(train, v = 5, repeats=1)

penlog_cv <- penlog_wf %>% 
  tune_grid(resamples = folds,
            grid = penlog_tuning_grid,
            metrics = metric_set(roc_auc))#, f_meas, sens, recall, spec,precision, accuracy)) #Or leave metrics NULL

penlog_bestTune <- penlog_cv %>% 
  select_best("roc_auc")

penlog_final_wf <- penlog_wf %>% 
  finalize_workflow(penlog_bestTune) %>% 
  fit(data = train)

penlog_preds <- predict(penlog_final_wf,
                        new_data=test,
                        type="prob")

penlog_submit <- as.data.frame(cbind(test$id, penlog_preds$.pred_1))
colnames(penlog_submit) <- c("id", "ACTION")
write_csv(penlog_submit, "penlog_submit.csv")


# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
          set_engine("ranger") %>%
          set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train - 1))), min_n(), levels=20)

folds <- vfold_cv(train, v = 5, repeats = 1)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(roc_auc))

rf_bestTune <- rf_results %>% 
  select_best("roc_auc")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="prob")

rf_submit <- as.data.frame(cbind(test$id, rf_preds$.pred_1))
colnames(rf_submit) <- c("id", "ACTION")
write_csv(rf_submit, "rf_submit.csv")


# Naive Bayes -------------------------------------------------------------

nb_mod <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>% 
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_mod)

nb_tuning_grid <- grid_regular(Laplace(),smoothness(),
                                   levels = 5)
folds <- vfold_cv(train, v = 2, repeats=1)

nb_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_tuning_grid,
            metrics = metric_set(roc_auc))

nb_bestTune <- nb_results %>% 
  select_best("roc_auc")

nb_final_wf <- nb_wf %>% 
  finalize_workflow(nb_bestTune) %>% 
  fit(data=train)

nb_preds <- predict(nb_final_wf,
                    new_data=test,
                    type="prob")

nb_submit <- as.data.frame(cbind(test$id, nb_preds$.pred_1))
colnames(nb_submit) <- c("id", "ACTION")
write_csv(nb_submit, "nb_submit.csv")


# KNN ---------------------------------------------------------------------
knn_mod <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")
  
knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_mod)

knn_tuning_grid <- grid_regular(neighbors(),levels = 5)
folds <- vfold_cv(train, v = 2, repeats=1)

knn_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = knn_tuning_grid,
            metrics = metric_set(roc_auc))

knn_bestTune <- knn_results %>% 
  select_best("roc_auc")

knn_final_wf <- knn_wf %>% 
  finalize_workflow(knn_bestTune) %>% 
  fit(data=train)

knn_preds <- predict(knn_final_wf,
                    new_data=test,
                    type="prob")

knn_submit <- as.data.frame(cbind(test$id, knn_preds$.pred_1))
colnames(knn_submit) <- c("id", "ACTION")
write_csv(knn_submit, "knn_submit.csv")


# Principle Component Dimension Reduction ---------------------------------

pcrd_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .005) %>% # combines categorical values that occur <5% into an "other" value
  step_dummy(all_nominal_predictors()) %>% # dummy variable encoding
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)

bake(prep(pcrd_recipe), new_data = train)


# Support Vector Machines -------------------------------------------------

svm_mod <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% 
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(svm_mod)

svm_tuning_grid <- grid_regular(rbf_sigma(), cost(),levels = 10)

folds <- vfold_cv(train, v = 5, repeats=1)

svm_results <- svm_wf %>% 
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(roc_auc))

svm_bestTune <- svm_results %>% 
  select_best("roc_auc")

svm_final_wf <- svm_wf %>% 
  finalize_workflow(svm_bestTune) %>% 
  fit(data=train)

svm_preds <- predict(svm_final_wf,
                    new_data=test,
                    type="prob")

svm_submit <- as.data.frame(cbind(test$id, svm_preds$.pred_1))
colnames(svm_submit) <- c("id", "ACTION")
write_csv(svm_submit, "svm_submit.csv")


# Balancing ---------------------------------------------------------------

smote_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_smote(all_outcomes(), neighbors=4)

smote_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

smote_wf <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(smote_mod)

smote_tuning_grid <- grid_regular(mtry(c(1,ncol(train - 1))), min_n(), levels=8)

folds <- vfold_cv(train, v = 5, repeats = 1)

smote_results <- smote_wf %>% 
  tune_grid(resamples = folds,
            grid = smote_tuning_grid,
            metrics = metric_set(roc_auc))

smote_bestTune <- smote_results %>% 
  select_best("roc_auc")

smote_final_wf <- smote_wf %>% 
  finalize_workflow(smote_bestTune) %>% 
  fit(data=train)

smote_preds <- predict(smote_final_wf,
                    new_data=test,
                    type="prob")

smote_submit <- as.data.frame(cbind(test$id, smote_preds$.pred_1))
colnames(smote_submit) <- c("id", "ACTION")
write_csv(smote_submit, "smote_submit.csv")


# Boost -------------------------------------------------------------------
folds <- vfold_cv(train, v = 5)

#rf, smote balanced

boost_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .8) %>% 
  step_smote(all_outcomes(), neighbors=5)
  

boost_mod <- boost_tree(trees = 100,
                        min_n = tune(),
                        tree_depth = tune(),
                        learn_rate = tune(),
                        loss_reduction = tune()) %>%
  set_engine("xgboost") %>% 
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>% 
  add_model(boost_mod) 

boost_params <- dials::parameters(min_n(),
                                    tree_depth(),
                                    learn_rate(),
                                    loss_reduction())
boost_tuning_grid <- grid_max_entropy(boost_params, size = 30)

boost_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = boost_tuning_grid,
            metrics = metric_set(roc_auc))

boost_bestTune <- boost_results %>% 
  select_best("roc_auc")

boost_final_wf <- boost_wf %>% 
  finalize_workflow(boost_bestTune) %>% 
  fit(data=train)

boost_preds <- predict(boost_final_wf,
                       new_data=test,
                       type="prob")

boost_submit <- as.data.frame(cbind(test$id, boost_preds$.pred_1))
colnames(boost_submit) <- c("id", "ACTION")

#boost_submit$ACTION <- ifelse(boost_submit$ACTION > 1, 1, boost_submit$ACTION)
#boost_submit$ACTION <- ifelse(boost_submit$ACTION < 0, 0, boost_submit$ACTION)
write_csv(boost_submit, "boost_submit.csv")
