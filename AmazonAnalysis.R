# Loading Packages
library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(rpart)
library(ranger)
library(stacks)
library(embed)

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
