library(tidymodels)
library(readr)
library(broom.mixed)

###########################Exercise 1 - Build a model##############################
#-------------->https://www.tidymodels.org/start/models/<-----------------#

urchins <- read_csv("https://tidymodels.org/start/models/urchins.csv") %>% 
setNames(c("food_regime", "initial_volume", "width")) %>% 
mutate(food_regime = factor(food_regime, levels = c("Initial", "Low", "High")))

# Plot the data 

ggplot(urchins,
       aes(x = initial_volume, 
           y = width, 
           group = food_regime, 
           col = food_regime)) + 
  geom_point() + 
  geom_smooth(method = lm, se = FALSE) +
  scale_color_viridis_d(option = "plasma", end = .7)


# Build and fit a model 
linear_reg()

lm_mod <- linear_reg() %>% 
  set_engine("lm")

# Fit a linear model 

lm_fit <- 
  lm_mod %>% 
  fit(width ~ initial_volume * food_regime, data=urchins)

print(lm_fit)

#Tidy the data

tidy(lm_fit)

# Predict new points with a new model 

new_x <- expand.grid(initial_volume = 20, 
                         food_regime = c("Initial", "Low", "High"))
new_x

# Generate a new prediction 

mean_pred <- predict(lm_fit, new_data = new_x)
print(mean_pred)

# Predict confidence interval 

conf_int_pred <- predict(lm_fit, 
                         new_data = new_x, 
                         type = "conf_int")
print(conf_int_pred)

# Combine mean prediction and confidence intervals

plot_data <- 
  new_x %>% 
  bind_cols(mean_pred) %>% 
  bind_cols(conf_int_pred)

# Prepare plot with confidence intervals
ggplot(plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, 
                    ymax = .pred_upper),
                width = .2) + 
  labs(y = "urchin size")



# Model with a different engine

prior_dist <- rstanarm::student_t(df = 1)

set.seed(123)

# make the parsnip model
bayes_mod <-   
  linear_reg() %>% 
  set_engine("stan", 
             prior_intercept = prior_dist, 
             prior = prior_dist) 

# train the model
bayes_fit <- 
  bayes_mod %>% 
  fit(width ~ initial_volume * food_regime, data = urchins)

print(bayes_fit, digits = 5)

tidy(bayes_fit, conf.int = TRUE)

bayes_plot_data <- 
  new_x %>% 
  bind_cols(predict(bayes_fit, new_data = new_x)) %>% 
  bind_cols(predict(bayes_fit, new_data = new_x, type = "conf_int"))

ggplot(bayes_plot_data, aes(x = food_regime)) + 
  geom_point(aes(y = .pred)) + 
  geom_errorbar(aes(ymin = .pred_lower, ymax = .pred_upper), width = .2) + 
  labs(y = "urchin size") + 
  ggtitle("Bayesian model with t(1) prior distribution")


# https://www.tidymodels.org/find/parsnip/


###################Exercise 2 - Preprocess data with recipes###############
#-------------->https://www.tidymodels.org/start/recipes/<-----------------#
library(tidymodels)      # for the recipes package, along with the rest of tidymodels
# Helper packages
library(nycflights13)    # for flight data
library(skimr)           # for variable summaries


set.seed(123)

flight_data <- 
  flights %>% 
  mutate(
    # Convert the arrival delay to a factor
    arr_delay = ifelse(arr_delay >= 30, "late", "on_time"),
    arr_delay = factor(arr_delay),
    # We will use the date (not date-time) in the recipe below
    date = as.Date(time_hour)
  ) %>% 
  # Include the weather data
  inner_join(weather, by = c("origin", "time_hour")) %>% 
  # Only retain the specific columns we will use
  select(dep_time, flight, origin, dest, air_time, distance, 
         carrier, date, arr_delay, time_hour) %>% 
  # Exclude missing data
  na.omit() %>% 
  # For creating models, it is better to have qualitative columns
  # encoded as factors (instead of character strings)
  mutate_if(is.character, as.factor)


flight_data %>% 
  count(arr_delay) %>% 
  mutate(prop = n/sum(n))

glimpse(flight_data)


flight_data %>% 
  skimr::skim(dest, carrier) 


###################### Data splitting ##########################

set.seed(555)
# Put 3/4 of the data into the training set 
data_split <- initial_split(flight_data, prop = 3/4)

# Create data frames for the two sets:
train_data <- training(data_split)
test_data  <- testing(data_split)

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") 

#Now we can add roles to this recipe. We can use the update_role() function to let recipes know that flight and time_hour are variables with a custom role that we called "ID" (a role can have any character value). Whereas our formula included all variables in the training set other than arr_delay as predictors, this tells the recipe to keep these two variables but not use them as either outcomes or predictors.

summary(flights_rec)


# Create features
flight_data %>% 
  distinct(date) %>% 
  mutate(numeric_date = as.numeric(date)) 

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>%               
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes())

test_data %>% 
  distinct(dest) %>% 
  anti_join(train_data)

# Cool recipes function to get rid of zero variance 

flights_rec <- 
  recipe(arr_delay ~ ., data = train_data) %>% 
  update_role(flight, time_hour, new_role = "ID") %>% 
  step_date(date, features = c("dow", "month")) %>% 
  step_holiday(date, holidays = timeDate::listHolidays("US")) %>% 
  step_rm(date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors())

# Fit a logistic regression model to the recipe

lr_mod <- 
  logistic_reg() %>% 
  set_engine("glm")

flights_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(flights_rec)
flights_wflow

flights_fit <- 
  flights_wflow %>% 
  fit(data = train_data)

flights_fit %>% 
  pull_workflow_fit() %>% 
  tidy()

# Make predictions with the new model

predict(flights_fit, test_data)

flights_pred <- 
  predict(flights_fit, test_data, type="prob") %>% 
  bind_cols(predict(flights_fit, test_data)) %>% 
  bind_cols(test_data %>% select(arr_delay, time_hour, flight))


flights_pred %>% 
  roc_curve(truth = arr_delay, .pred_late) %>% 
  autoplot()

flights_pred %>% 
  roc_auc(truth = arr_delay, .pred_late)

###################Exercise 3 - Evaluate Model with resampling###############
#-------------->https://www.tidymodels.org/start/resampling/<-----------------#
library(tidymodels) # for the rsample package, along with the rest of tidymodels
# Helper packages
library(modeldata)  # for the cells data
data(cells, package = "modeldata")
cells

cells %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)

# Strata split helps with class imbalance issues
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)
nrow(cell_train)
nrow(cell_train)/nrow(cells)

# training set proportions by class
cell_train %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

cell_test %>% 
  count(class) %>% 
  mutate(prop = n/sum(n))

#Random forest models are ensembles of decision trees. A large number of decision tree models are created for the ensemble based on slightly different versions of the training set. When creating the individual decision trees, the fitting process encourages them to be as diverse as possible. The collection of trees are combined into the random forest model and, when a new sample is predicted, the votes from each tree are used to calculate the final predicted value for the new sample. For categorical outcome variables like class in our cells data example, the majority vote across all the trees in the random forest determines the predicted class for the new sample.

rf_mod <- 
  rand_forest(trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

set.seed(234)
rf_fit <- 
  rf_mod %>% 
  fit(class ~ ., data = cell_train)
rf_fit

rf_training_pred <- 
  predict(rf_fit, cell_train) %>% 
  bind_cols(predict(rf_fit, cell_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(cell_train %>% 
              select(class))

# Make predictions with yardstick package

rf_training_pred %>%                # training set predictions
  roc_auc(truth = class, .pred_PS)

rf_training_pred %>%                # training set predictions
  accuracy(truth = class, .pred_class)

rf_testing_pred <- 
  predict(rf_fit, cell_test) %>% 
  bind_cols(predict(rf_fit, cell_test, type = "prob")) %>% 
  bind_cols(cell_test %>% select(class))

rf_testing_pred %>%                   # test set predictions
  roc_auc(truth = class, .pred_PS)

rf_testing_pred %>%                   # test set predictions
  accuracy(truth = class, .pred_class)


# Models like random forests, neural networks, and other black-box methods can essentially memorize the training set. Re-predicting that same set should always result in nearly perfect results.
#The training set does not have the capacity to be a good arbiter of performance. It is not an independent piece of information; predicting the training set can only reflect what the model already knows.

set.seed(345)
folds <- vfold_cv(cell_train, v = 10)
folds

# Fit the resamples to the data 

rf_wf <- 
  workflow() %>%
  add_model(rf_mod) %>%
  add_formula(class ~ .)

set.seed(456)
rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

rf_fit_rs

# Collect metrics to summarise the splits by 10 fold

collect_metrics(rf_fit_rs)


###################Exercise 4 - Tune Model Parameters####################
#https://www.tidymodels.org/start/tuning/
library(tidymodels)  # for the tune package, along with the rest of tidymodels
# Helper packages
library(modeldata)   # for the cells data
library(vip)         # for variable importance plots


data(cells, package = "modeldata")
cells


set.seed(123)
cell_split <- initial_split(cells %>% select(-case), 
                            strata = class)
cell_train <- training(cell_split)
cell_test  <- testing(cell_split)

tune_spec <- 
  decision_tree(
    cost_complexity = tune(),
    tree_depth = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

print(tune_spec)

#Dials package
tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 5)

tree_grid %>% 
  count(tree_depth)

set.seed(234)
cell_folds <- vfold_cv(cell_train)


set.seed(345)

tree_wf <- workflow() %>%
  add_model(tune_spec) %>%
  add_formula(class ~ .)

tree_res <- 
  tree_wf %>% 
  tune_grid(
    resamples = cell_folds,
    grid = tree_grid
  )

tree_res


# Collect the metrics from the combination of hyperparameters
tree_res %>% 
  collect_metrics()

tree_res %>%
  collect_metrics() %>%
  mutate(tree_depth = factor(tree_depth)) %>%
  ggplot(aes(cost_complexity, mean, color = tree_depth)) +
  geom_line(size = 1.5, alpha = 0.6) +
  geom_point(size = 2) +
  facet_wrap(~ .metric, scales = "free", nrow = 2) +
  scale_x_log10(labels = scales::label_number()) +
  scale_color_viridis_d(option = "plasma", begin = .9, end = 0)


tree_res %>%
  show_best("roc_auc")

# Shows all the ROC statistics
best_tree <- tree_res %>%
  select_best("roc_auc")

best_tree

# Finalise the model

final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_wf

# Now fit the tuning to the training data

final_tree <- 
  final_wf %>% 
  fit(data = cell_train)

print(final_tree)

# Check global variable importance

library(vip)
library(ggthemes)

final_tree %>% 
  pull_workflow_fit() %>% 
  vip()


# The final fit
final_fit <- 
  final_wf %>%
  last_fit(cell_split) 

final_fit %>%
  collect_metrics()

final_fit %>%
  collect_predictions() %>% 
  roc_curve(class, .pred_PS) %>% 
  autoplot() + theme_economist()

args(decision_tree)


###################Exercise 5 - Predictive Modelling Case Study####################
#https://www.tidymodels.org/start/case-study/

library(tidymodels)  
# Helper packages
library(readr)       # for importing data
library(vip)         # for variable importance plots

library(tidymodels)
library(readr)

hotels <- 
  read_csv('https://tidymodels.org/start/case-study/hotels.csv') %>%
  mutate_if(is.character, as.factor) 

dim(hotels)
glimpse(hotels)

hotels %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))

set.seed(123)
splits      <- initial_split(hotels, strata = children)

hotel_other <- training(splits)
hotel_test  <- testing(splits)


hotel_other %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))

hotel_test  %>% 
  count(children) %>% 
  mutate(prop = n/sum(n))

# Do a validation and training split
set.seed(234)
val_set <- validation_split(hotel_other, 
                            strata = children, 
                            prop = 0.80)
val_set
    
# Fit Model
lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")


holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")


# Create the recipe
lr_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date, holidays = holidays) %>% 
  step_rm(arrival_date) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

# Create the workflow

lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)

print(lr_workflow)

# Create tuning grid

lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
lr_reg_grid %>% top_n(-5) # lowest penalty values
lr_reg_grid %>% top_n(-5) # lowest penalty values

lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot 

# Get the top models

top_models <-
  lr_res %>% 
  show_best("roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models


lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(12)
lr_best

lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(children, .pred_children) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)



# Build a tree based ensemble

# Get ready for parellel processing threading
cores <- parallel::detectCores()
cores

# Fit model
rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

# Create recipe
rf_recipe <- 
  recipe(children ~ ., data = hotel_other) %>% 
  step_date(arrival_date) %>% 
  step_holiday(arrival_date) %>% 
  step_rm(arrival_date) 

# Create workflow
rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

# Train and tune the model
rf_mod
rf_mod %>%    
  parameters()  


#Tune grid via grid tuning
set.seed(345)
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))


rf_res %>% 
  show_best(metric = "roc_auc")
