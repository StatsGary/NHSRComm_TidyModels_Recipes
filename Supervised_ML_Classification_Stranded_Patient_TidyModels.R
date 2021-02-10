# Load in libraries

library(tidymodels)
library(readr)
library(broom)
library(broom.mixed)
library(skimr)
library(dplyr)
library(magrittr)
library(parallel)
library(doParallel)
library(rockthemes)
library(vip)

####################################### STEP ONE - READ IN THE DATA AND DO SIMPLE TRANSFORMS ############################

# Read in the data
names(strand_pat)
strand_pat <- read_csv("Data/Stranded_Data.csv") %>% 
  setNames(c("stranded_class", "age", "care_home_ref_flag", "medically_safe_flag", 
             "hcop_flag", "needs_mental_health_support_flag", "previous_care_in_last_12_month", "admit_date", "frail_descrip")) %>% 
  mutate(stranded_class = factor(stranded_class)) %>% 
  drop_na()

####################################### STEP TWO - TEST CLASS IMBALANCE ################################################

class_bal_table <- table(strand_pat$stranded_class)
print(class_bal_table)
prop_tab <- prop.table(class_bal_table)
print(prop_tab)
upsample_ratio <- class_bal_table[2] / sum(class_bal_table)
#There is a slight class imbalance with the data - we will do this in the recipes steps

####################################### STEP THREE - OBSERVE DATA STRUCTURES#############################################
# Convert admit date
strand_pat$admit_date <- as.Date(strand_pat$admit_date, format="%d/%m/%Y")
str(strand_pat)
factors <- names(select_if(strand_pat, is.factor))
numbers <- names(select_if(strand_pat, is.numeric))
characters <- names(select_if(strand_pat, is.character))


# This separate the data frame too allow you to quickly view what we will need to transform in the recipes step


####################################### STEP FOUR- DATA PARTITIONING WITH RSAMPLE########################################

# Partition into training and hold out test / validation sample
set.seed(123)
split <- rsample::initial_split(strand_pat, prop=3/4)
train_data <- rsample::training(split)
test_data <- rsample::testing(split)

####################################### STEP FIVE - CREATE THE RECIPE AND MODEL STEPS ########################################

# Partition into training and hold out test / validation sample

#https://recipes.tidymodels.org/reference/index.html

stranded_rec <- 
  recipe(stranded_class ~ ., data=train_data) %>% 
  step_date(admit_date, features = c("dow", "month")) %>% 
  step_rm(admit_date) %>% 
  step_upsample(stranded_class, over_ratio = as.numeric(upsample_ratio)) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

test_data %>% 
  anti_join(train_data)


####################################### STEP SIX- START MODELLING #############################################
# Initialise model

lr_mod <- 
  parsnip::logistic_reg() %>% 
  set_engine("glm")

# Create model workflow
strand_wf <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(stranded_rec)

print(strand_wf)

strand_fit <- 
  strand_wf %>% 
  fit(data = train_data)

# Create the model fit
strand_fitted <- strand_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

# Add significance column

strand_fitted <- strand_fitted  %>% 
  mutate(Significance = ifelse(p.value < 0.05, "Significant", "Insignificant")) %>% 
  arrange(desc(p.value)) 

plot <- strand_fitted %>% 
  ggplot(data = strand_fitted, mapping = aes(x=term, y=p.value, fill=Significance)) +
  geom_col() + theme(axis.text.x = element_text(
                                        face="bold", color="#0070BA", 
                                        size=8, angle=90)
                                                ) + labs(y="P value", x="Terms", 
                                                         title="P value significance chart",
                                                         subtitle="A chart to represent the significant variables in the model",
                                                         caption="Produced by Gary Hutson")

print("Creating plot of P values")
print(plot)
ggsave("Figures/p_val_plot.png", plot)

################################## STEP SEVEN - PREDICT WITH HOLDOUT DATASET ##################################

class_pred <- predict(strand_fit, test_data)
prob_pred <- predict(strand_fit, test_data, type="prob")
lr_predictions <- data.frame(class_pred, prob_pred) %>% 
  setNames(c("LR_Class", "LR_NotStrandedProb", "LR_StrandedProb"))

# Bind on to test data

stranded_preds <- test_data %>% 
  bind_cols(lr_predictions)

str(stranded_preds)

################################## STEP EIGHT - EVALUATE FIT WITH YARDSTICK AND CARET###########################

#Evaluate ROC curve

roc_plot <- 
  stranded_preds %>% 
  roc_curve(truth = stranded_class, LR_NotStrandedProb) %>% 
  autoplot

ggsave("Figures/log_reg_roc_plot.png", roc_plot)

# Evaluate on confusion matrix

library(caret)
cm <- caret::confusionMatrix(stranded_preds$stranded_class,
                       stranded_preds$LR_Class, 
                       positive="Stranded")

# As it is a binary classification problem this can be evaluated on the custom confusion matrix plot I provide with this session
source("Functions/confusion_matrix_plot_function.R")

cm_plot <- conf_matrix_plot(cm, class_label1 = "Not Stranded", 
                     class_label2 = "Stranded")

ggsave("Figures/conf_mat_plot_LR.png", last_plot())


# Save the data so far and load in next workbook


########################### STEP NINE - IMPROVE MODEL WITH RESAMPLING (RSAMPLE) AND TUNE###################################

set.seed(123)
ten_fold <- vfold_cv(train_data, v=10)

# Use the previous stranded workflow to improve

set.seed(123)
lr_fit_rs <- 
  strand_wf %>% 
  fit_resamples(ten_fold)

# To collect the resmaples you need to call collect_metrics to average out the accuracy for that model

collected_mets <- tune::collect_metrics(lr_fit_rs)

# Now I can compare the accuracy from the previous test set I had already generated a confusion matrix for
accuracy_resamples <- collected_mets$mean[1] * 100
accuracy_validation_set <- as.numeric(cm$overall[1] * 100)
print(cat(paste0("The true accuracy of the model is between the resample testing:", 
            round(accuracy_resamples,2), "\nThe validation sample: ",
            round(accuracy_validation_set,2), ".")))


########################## STEP 10 - IMPROVE MODEL WITH A DIFFERENT MODEL SELECTION AND RESAMPLING ################


rf_mod <- 
  rand_forest(trees=500) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

# Fit the model
rf_fit <- 
  rf_mod %>% 
  fit(stranded_class ~ ., data = train_data)

# Fit the model to resamples

rf_wf <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_formula(stranded_class ~ .)

set.seed(123)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(ten_fold)

# Collect the metrics using another model with resampling

rf_resample_mean_preds <- tune::collect_metrics(rf_fit_rs)

# Resampling allows us to get a better estimate of the true performance of the model with new data


###################### STEP 11 - IMPROVE THE MODEL WITH HYPERPARAMETER TUNING WITH DIALS #######################

# We will manually tune a tree to see if we can obtain better results than before

tune_tree <- 
  decision_tree(
    cost_complexity = tune(), 
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

# Tune here acts as a placeholder to say we are going to come back later and create what is
# known as a tuning grid


grid_tree_tune <- grid_regular(dials::cost_complexity(),
                               dials::tree_depth(), 
                               levels = 10)
print(tail(grid_tree_tune))

# We will use these parameters on our package. The dials package makes the job of hyperparameter tuning much simpiler by 
# the potential to create these grids


# Create the workflow

all_cores <- parallel::detectCores(logical = FALSE)-1
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)

set.seed(123)
tree_wf <- workflow() %>% 
  add_model(tune_tree) %>% 
  add_formula(stranded_class ~ .)

tree_pred_tuned <- 
  tree_wf %>% 
  tune::tune_grid(
    resamples = ten_fold, 
    grid = grid_tree_tune
  )

# Collect the metrics from the grid tuning process

tune_plot <- tree_pred_tuned %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1, alpha = 0.7) +
  geom_point(size = 1.5) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0) + theme_minimal()

print(tune_plot)

# To get the best ROC - area under the curve value we will use the following:

tree_pred_tuned %>% 
  tune::show_best("roc_auc")

# Select the best tree

best_tree <- tree_pred_tuned %>% 
  tune::select_best("roc_auc")

# Use the best tree from the tuning process to 

final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

print(final_wf)

# Make a prediction with the final tree

final_tree_pred <- 
  final_wf %>% 
  fit(data = train_data)

print(final_tree_pred)


# Pull the workflow fit for the tree
final_tree_pred %>% 
  pull_workflow_fit() %>% 
  vip()


# Create the final prediction
final_fit <- 
  final_wf %>% 
  last_fit(split)


# Get the fits from the resamples using the tune package

final_fit_fitted_metrics <- final_fit %>% 
  collect_metrics() 

final_fit_predictions <- final_fit %>% 
  collect_predictions()

print(final_fit_fitted_metrics)


final_fit_predictions %>% 
  roc_curve(stranded_class, `.pred_Not Stranded`) %>% 
  autoplot()


# To find the values of any parsnip model use the below

args(decision_tree)
args(logistic_reg)
args(rand_forest)

# Full list of models here: 
