## ----setup--------------------------------------------------------------------
# nolint start
library(mlexperiments)


## -----------------------------------------------------------------------------
library(mlbench)
data("DNA")
dataset <- DNA |>
  data.table::as.data.table() |>
  na.omit()

feature_cols <- colnames(dataset)[1:180]
target_col <- "Class"


## -----------------------------------------------------------------------------
seed <- 123
if (isTRUE(as.logical(Sys.getenv("_R_CHECK_LIMIT_CORES_")))) {
  # on cran
  ncores <- 2L
} else {
  ncores <- ifelse(
    test = parallel::detectCores() > 4,
    yes = 4L,
    no = ifelse(
      test = parallel::detectCores() < 2L,
      yes = 1L,
      no = parallel::detectCores()
    )
  )
}
options("mlexperiments.bayesian.max_init" = 10L)


## -----------------------------------------------------------------------------
data_split <- splitTools::partition(
  y = dataset[, get(target_col)],
  p = c(train = 0.7, test = 0.3),
  type = "stratified",
  seed = seed
)

train_x <- model.matrix(
  ~ -1 + .,
  dataset[data_split$train, .SD, .SDcols = feature_cols]
)
train_y <- dataset[data_split$train, get(target_col)]


test_x <- model.matrix(
  ~ -1 + .,
  dataset[data_split$test, .SD, .SDcols = feature_cols]
)
test_y <- dataset[data_split$test, get(target_col)]


## -----------------------------------------------------------------------------
fold_list <- splitTools::create_folds(
  y = train_y,
  k = 3,
  type = "stratified",
  seed = seed
)


## -----------------------------------------------------------------------------
# required learner arguments, not optimized
learner_args <- list(
  l = 2,
  test = parse(text = "fold_test$x"),
  use.all = FALSE
)

# set arguments for predict function and performance metric,
# required for mlexperiments::MLCrossValidation and
# mlexperiments::MLNestedCV
predict_args <- list(type = "response")
performance_metric <- metric("bacc")
performance_metric_args <- NULL
return_models <- FALSE

# required for grid search and initialization of bayesian optimization
parameter_grid <- expand.grid(
  k = seq(4, 68, 6)
)
# reduce to a maximum of 10 rows
if (nrow(parameter_grid) > 10) {
  set.seed(123)
  sample_rows <- sample(seq_len(nrow(parameter_grid)), 10, FALSE)
  parameter_grid <- kdry::mlh_subset(parameter_grid, sample_rows)
}

# required for bayesian optimization
parameter_bounds <- list(k = c(2L, 80L))
optim_args <- list(
  iters.n = ncores,
  kappa = 3.5,
  acq = "ucb"
)


## -----------------------------------------------------------------------------
tuner <- mlexperiments::MLTuneParameters$new(
  learner = LearnerKnn$new(),
  strategy = "grid",
  ncores = ncores,
  seed = seed
)

tuner$parameter_grid <- parameter_grid
tuner$learner_args <- learner_args
tuner$split_type <- "stratified"

tuner$set_data(
  x = train_x,
  y = train_y
)

tuner_results_grid <- tuner$execute(k = 3)
#> 
#> Parameter settings [===================>------------------------------------------------------------------------------] 2/10 ( 20%)
#> Parameter settings [============================>---------------------------------------------------------------------] 3/10 ( 30%)
#> Parameter settings [======================================>-----------------------------------------------------------] 4/10 ( 40%)
#> Parameter settings [================================================>-------------------------------------------------] 5/10 ( 50%)
#> Parameter settings [==========================================================>---------------------------------------] 6/10 ( 60%)
#> Parameter settings [====================================================================>-----------------------------] 7/10 ( 70%)
#> Parameter settings [=============================================================================>--------------------] 8/10 ( 80%)
#> Parameter settings [=======================================================================================>----------] 9/10 ( 90%)
#> Parameter settings [=================================================================================================] 10/10 (100%)

head(tuner_results_grid)
#>    setting_id metric_optim_mean  k l use.all
#> 1:          1         0.1669134 16 2   FALSE
#> 2:          2         0.1256584 64 2   FALSE
#> 3:          3         0.1870928 10 2   FALSE
#> 4:          4         0.1364111 34 2   FALSE
#> 5:          5         0.1243125 58 2   FALSE
#> 6:          6         0.1462841 28 2   FALSE


## -----------------------------------------------------------------------------
tuner <- mlexperiments::MLTuneParameters$new(
  learner = LearnerKnn$new(),
  strategy = "bayesian",
  ncores = ncores,
  seed = seed
)

tuner$parameter_grid <- parameter_grid
tuner$parameter_bounds <- parameter_bounds

tuner$learner_args <- learner_args
tuner$optim_args <- optim_args

tuner$split_type <- "stratified"

tuner$set_data(
  x = train_x,
  y = train_y
)

tuner_results_bayesian <- tuner$execute(k = 3)
#> 
#> Registering parallel backend using 4 cores.

head(tuner_results_bayesian)
#>    Epoch setting_id  k gpUtility acqOptimum inBounds Elapsed      Score metric_optim_mean errorMessage l use.all
#> 1:     0          1 16        NA      FALSE     TRUE   1.061 -0.1651140         0.1651140           NA 2   FALSE
#> 2:     0          2 64        NA      FALSE     TRUE   1.131 -0.1261065         0.1261065           NA 2   FALSE
#> 3:     0          3 10        NA      FALSE     TRUE   1.060 -0.1835086         0.1835086           NA 2   FALSE
#> 4:     0          4 34        NA      FALSE     TRUE   1.074 -0.1377516         0.1377516           NA 2   FALSE
#> 5:     0          5 58        NA      FALSE     TRUE   1.101 -0.1247624         0.1247624           NA 2   FALSE
#> 6:     0          6 28        NA      FALSE     TRUE   1.046 -0.1462823         0.1462823           NA 2   FALSE


## -----------------------------------------------------------------------------
validator <- mlexperiments::MLCrossValidation$new(
  learner = LearnerKnn$new(),
  fold_list = fold_list,
  ncores = ncores,
  seed = seed
)

validator$learner_args <- tuner$results$best.setting[-1]

validator$predict_args <- predict_args
validator$performance_metric <- performance_metric
validator$performance_metric_args <- performance_metric_args
validator$return_models <- return_models

validator$set_data(
  x = train_x,
  y = train_y
)

validator_results <- validator$execute()
#> 
#> CV fold: Fold1
#> 
#> CV fold: Fold2
#> CV progress [======================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> CV fold: Fold3
#> CV progress [==========================================================================================================] 3/3 (100%)

head(validator_results)
#>     fold performance  k l use.all
#> 1: Fold1   0.8931022 58 2   FALSE
#> 2: Fold2   0.8445084 58 2   FALSE
#> 3: Fold3   0.9010913 58 2   FALSE


## -----------------------------------------------------------------------------
validator <- mlexperiments::MLNestedCV$new(
  learner = LearnerKnn$new(),
  strategy = "grid",
  fold_list = fold_list,
  k_tuning = 3L,
  ncores = ncores,
  seed = seed
)

validator$parameter_grid <- parameter_grid
validator$learner_args <- learner_args
validator$split_type <- "stratified"

validator$predict_args <- predict_args
validator$performance_metric <- performance_metric
validator$performance_metric_args <- performance_metric_args
validator$return_models <- return_models

validator$set_data(
  x = train_x,
  y = train_y
)

validator_results <- validator$execute()
#> 
#> CV fold: Fold1
#> 
#> Parameter settings [===================>------------------------------------------------------------------------------] 2/10 ( 20%)
#> Parameter settings [============================>---------------------------------------------------------------------] 3/10 ( 30%)
#> Parameter settings [======================================>-----------------------------------------------------------] 4/10 ( 40%)
#> Parameter settings [================================================>-------------------------------------------------] 5/10 ( 50%)
#> Parameter settings [==========================================================>---------------------------------------] 6/10 ( 60%)
#> Parameter settings [====================================================================>-----------------------------] 7/10 ( 70%)
#> Parameter settings [=============================================================================>--------------------] 8/10 ( 80%)
#> Parameter settings [=======================================================================================>----------] 9/10 ( 90%)
#> Parameter settings [=================================================================================================] 10/10 (100%)                                                                                                                                    
#> CV fold: Fold2
#> CV progress [======================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> Parameter settings [===================>------------------------------------------------------------------------------] 2/10 ( 20%)
#> Parameter settings [============================>---------------------------------------------------------------------] 3/10 ( 30%)
#> Parameter settings [======================================>-----------------------------------------------------------] 4/10 ( 40%)
#> Parameter settings [================================================>-------------------------------------------------] 5/10 ( 50%)
#> Parameter settings [==========================================================>---------------------------------------] 6/10 ( 60%)
#> Parameter settings [====================================================================>-----------------------------] 7/10 ( 70%)
#> Parameter settings [=============================================================================>--------------------] 8/10 ( 80%)
#> Parameter settings [=======================================================================================>----------] 9/10 ( 90%)
#> Parameter settings [=================================================================================================] 10/10 (100%)                                                                                                                                    
#> CV fold: Fold3
#> CV progress [==========================================================================================================] 3/3 (100%)
#>                                                                                                                                     
#> Parameter settings [===================>------------------------------------------------------------------------------] 2/10 ( 20%)
#> Parameter settings [============================>---------------------------------------------------------------------] 3/10 ( 30%)
#> Parameter settings [======================================>-----------------------------------------------------------] 4/10 ( 40%)
#> Parameter settings [================================================>-------------------------------------------------] 5/10 ( 50%)
#> Parameter settings [==========================================================>---------------------------------------] 6/10 ( 60%)
#> Parameter settings [====================================================================>-----------------------------] 7/10 ( 70%)
#> Parameter settings [=============================================================================>--------------------] 8/10 ( 80%)
#> Parameter settings [=======================================================================================>----------] 9/10 ( 90%)
#> Parameter settings [=================================================================================================] 10/10 (100%)

head(validator_results)
#>     fold performance  k l use.all
#> 1: Fold1   0.8863818 64 2   FALSE
#> 2: Fold2   0.8396360 64 2   FALSE
#> 3: Fold3   0.9000926 64 2   FALSE


## -----------------------------------------------------------------------------
validator <- mlexperiments::MLNestedCV$new(
  learner = LearnerKnn$new(),
  strategy = "bayesian",
  fold_list = fold_list,
  k_tuning = 3L,
  ncores = ncores,
  seed = seed
)

validator$parameter_grid <- parameter_grid
validator$learner_args <- learner_args
validator$split_type <- "stratified"


validator$parameter_bounds <- parameter_bounds
validator$optim_args <- optim_args

validator$predict_args <- predict_args
validator$performance_metric <- performance_metric
validator$performance_metric_args <- performance_metric_args
validator$return_models <- return_models

validator$set_data(
  x = train_x,
  y = train_y
)

validator_results <- validator$execute()
#> 
#> CV fold: Fold1
#> 
#> Registering parallel backend using 4 cores.
#> 
#> CV fold: Fold2
#> CV progress [======================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> Registering parallel backend using 4 cores.
#> 
#> CV fold: Fold3
#> CV progress [==========================================================================================================] 3/3 (100%)
#> Registering parallel backend using 4 cores.

head(validator_results)
#>     fold performance  k l use.all
#> 1: Fold1   0.8702444 28 2   FALSE
#> 2: Fold2   0.8396360 64 2   FALSE
#> 3: Fold3   0.9010913 58 2   FALSE


## ----include=FALSE------------------------------------------------------------
# nolint end

