## ----include = FALSE----------------------------------------------------------
# nolint start


## ----eval=FALSE---------------------------------------------------------------
# knn_fit <- function(x, y, ncores, seed, ...) {
#   kwargs <- list(...)
#   stopifnot("k" %in% names(kwargs))
#   args <- kdry::list.append(list(train = x, cl = y), kwargs)
#   args$prob <- TRUE
#   set.seed(seed)
#   fit <- do.call(class::knn, args)
#   return(fit)
# }


## ----eval=FALSE---------------------------------------------------------------
# knn_predict <- function(model, newdata, ncores, ...) {
#   kwargs <- list(...)
#   stopifnot("type" %in% names(kwargs))
#   if (kwargs$type == "response") {
#     return(model)
#   } else if (kwargs$type == "prob") {
#     # there is no knn-model but the probabilities predicted for the test data
#     return(attributes(model)$prob)
#   }
# }


## ----eval=FALSE---------------------------------------------------------------
# knn_optimization <- function(x, y, params, fold_list, ncores, seed) {
#   stopifnot(is.list(params), "k" %in% names(params))
#   # initialize a dataframe to store the results
#   results_df <- data.table::data.table(
#     "fold" = character(0),
#     "metric" = numeric(0)
#   )
#   # we do not need test here as it is defined explicitly below
#   params[["test"]] <- NULL
#   # loop over the folds
#   for (fold in names(fold_list)) {
#     # get row-ids of the current fold
#     train_idx <- fold_list[[fold]]
#     # create learner-arguments
#     args <- kdry::list.append(
#       list(
#         x = kdry::mlh_subset(x, train_idx),
#         test = kdry::mlh_subset(x, -train_idx),
#         y = kdry::mlh_subset(y, train_idx),
#         use.all = FALSE,
#         ncores = ncores,
#         seed = seed
#       ),
#       params
#     )
#     set.seed(seed)
#     cvfit <- do.call(knn_fit, args)
#     # optimize mean misclassification error
#     FUN <- metric("MMAC") # nolint
#     err <- FUN(predictions = knn_predict(
#       model = cvfit,
#       newdata = kdry::mlh_subset(x, -train_idx),
#       ncores = ncores,
#       type = "response"
#       ),
#       ground_truth = kdry::mlh_subset(y, -train_idx)
#     )
#     results_df <- data.table::rbindlist(
#       l = list(results_df, list("fold" = fold, "validation_metric" = err)),
#       fill = TRUE
#     )
#   }
#   res <- list("metric_optim_mean" = mean(results_df$validation_metric))
#   return(res)
# }


## ----eval=FALSE---------------------------------------------------------------
# knn_bsF <- function(...) { # nolint
#   params <- list(...)
#   # call to knn_optimization here with ncores = 1, since the Bayesian search
#   # is parallelized already / "FUN is fitted n times in m threads"
#   set.seed(seed)#, kind = "L'Ecuyer-CMRG")
#   bayes_opt_knn <- knn_optimization(
#     x = x,
#     y = y,
#     params = params,
#     fold_list = method_helper$fold_list,
#     ncores = 1L, # important, as bayesian search is already parallelized
#     seed = seed
#   )
#   ret <- kdry::list.append(
#     list("Score" = bayes_opt_knn$metric_optim_mean),
#     bayes_opt_knn
#   )
#   return(ret)
# }


## ----eval=FALSE---------------------------------------------------------------
# # define the objects / functions that need to be exported to each cluster
# # for parallelizing the Bayesian optimization.
# knn_ce <- function() {
#   c("knn_optimization", "knn_fit", "knn_predict", "metric", ".format_xy")
# }


## ----eval=FALSE---------------------------------------------------------------
# LearnerKnn <- R6::R6Class( # nolint
#   classname = "LearnerKnn",
#   inherit = mlexperiments::MLLearnerBase,
#   public = list(
#     initialize = function() {
#       if (!requireNamespace("class", quietly = TRUE)) {
#         stop(
#           paste0(
#             "Package \"class\" must be installed to use ",
#             "'learner = \"LearnerKnn\"'."
#           ),
#           call. = FALSE
#         )
#       }
#       super$initialize(
#         metric_optimization_higher_better = FALSE # classification error
#       )
# 
#       private$fun_fit <- knn_fit
#       private$fun_predict <- knn_predict
#       private$fun_optim_cv <- knn_optimization
#       private$fun_bayesian_scoring_function <- knn_bsF
# 
#       self$environment <- "mlexperiments"
#       self$cluster_export <- knn_ce()
#     }
#   )
# )


## -----------------------------------------------------------------------------
library(mlexperiments)
library(mlbench)

data("DNA")
dataset <- DNA |>
  data.table::as.data.table() |>
  na.omit()

seed <- 123
feature_cols <- colnames(dataset)[1:180]

train_x <- model.matrix(
  ~ -1 + .,
  dataset[, .SD, .SDcols = feature_cols]
)
train_y <- dataset[, get("Class")]

ncores <- ifelse(
  test = parallel::detectCores() > 4,
  yes = 4L,
  no = ifelse(
    test = parallel::detectCores() < 2L,
    yes = 1L,
    no = parallel::detectCores()
  )
)
if (isTRUE(as.logical(Sys.getenv("_R_CHECK_LIMIT_CORES_")))) {
  # on cran
  ncores <- 2L
}


## -----------------------------------------------------------------------------
param_list_knn <- expand.grid(
  k = seq(4, 68, 8),
  l = 0,
  test = parse(text = "fold_test$x")
)

knn_bounds <- list(k = c(2L, 80L))

optim_args <- list(
  iters.n = ncores,
  kappa = 3.5,
  acq = "ucb"
)


## -----------------------------------------------------------------------------
knn_tune_bayesian <- mlexperiments::MLTuneParameters$new(
  learner = LearnerKnn$new(),
  strategy = "bayesian",
  ncores = ncores,
  seed = seed
)

knn_tune_bayesian$parameter_bounds <- knn_bounds
knn_tune_bayesian$parameter_grid <- param_list_knn
knn_tune_bayesian$split_type <- "stratified"
knn_tune_bayesian$optim_args <- optim_args

# set data
knn_tune_bayesian$set_data(
  x = train_x,
  y = train_y
)

results <- knn_tune_bayesian$execute(k = 3)
#> 
#> Registering parallel backend using 4 cores.

head(results)
#>    Epoch setting_id  k gpUtility acqOptimum inBounds Elapsed      Score metric_optim_mean errorMessage l
#> 1:     0          1  4        NA      FALSE     TRUE   2.153 -0.2247332         0.2247332           NA 0
#> 2:     0          2 12        NA      FALSE     TRUE   2.274 -0.1600753         0.1600753           NA 0
#> 3:     0          3 20        NA      FALSE     TRUE   2.006 -0.1381042         0.1381042           NA 0
#> 4:     0          4 28        NA      FALSE     TRUE   2.329 -0.1403013         0.1403013           NA 0
#> 5:     0          5 36        NA      FALSE     TRUE   2.109 -0.1315129         0.1315129           NA 0
#> 6:     0          6 44        NA      FALSE     TRUE   2.166 -0.1258632         0.1258632           NA 0


## -----------------------------------------------------------------------------
knn_tune_grid <- mlexperiments::MLTuneParameters$new(
  learner = LearnerKnn$new(),
  strategy = "grid",
  ncores = ncores,
  seed = seed
)

knn_tune_grid$parameter_grid <- param_list_knn
knn_tune_grid$split_type <- "stratified"

# set data
knn_tune_grid$set_data(
  x = train_x,
  y = train_y
)

results <- knn_tune_grid$execute(k = 3)
#> 
#> Parameter settings [=====================>---------------------------------------------------------------------------] 2/9 ( 22%)
#> Parameter settings [===============================>-----------------------------------------------------------------] 3/9 ( 33%)
#> Parameter settings [==========================================>------------------------------------------------------] 4/9 ( 44%)
#> Parameter settings [=====================================================>-------------------------------------------] 5/9 ( 56%)
#> Parameter settings [================================================================>--------------------------------] 6/9 ( 67%)
#> Parameter settings [==========================================================================>----------------------] 7/9 ( 78%)
#> Parameter settings [=====================================================================================>-----------] 8/9 ( 89%)
#> Parameter settings [=================================================================================================] 9/9 (100%)                                                                                                                                  

head(results)
#>    setting_id metric_optim_mean  k l
#> 1:          1         0.2187696  4 0
#> 2:          2         0.1597615 12 0
#> 3:          3         0.1349655 20 0
#> 4:          4         0.1406152 28 0
#> 5:          5         0.1318267 36 0
#> 6:          6         0.1258632 44 0


## -----------------------------------------------------------------------------
fold_list <- splitTools::create_folds(
  y = train_y,
  k = 3,
  type = "stratified",
  seed = seed
)
str(fold_list)
#> List of 3
#>  $ Fold1: int [1:2124] 1 2 3 4 5 7 9 10 11 12 ...
#>  $ Fold2: int [1:2124] 1 2 3 6 8 9 11 13 16 17 ...
#>  $ Fold3: int [1:2124] 4 5 6 7 8 10 12 14 15 16 ...


## -----------------------------------------------------------------------------
knn_cv <- mlexperiments::MLCrossValidation$new(
  learner = LearnerKnn$new(),
  fold_list = fold_list,
  seed = seed
)

best_grid_result <- knn_tune_grid$results$best.setting
best_grid_result
#> $setting_id
#> [1] 9
#> 
#> $k
#> [1] 68
#> 
#> $l
#> [1] 0
#> 
#> $test
#> expression(fold_test$x)

knn_cv$learner_args <- best_grid_result[-1]

knn_cv$predict_args <- list(type = "response")
knn_cv$performance_metric <- metric("ACC")
knn_cv$return_models <- TRUE

# set data
knn_cv$set_data(
  x = train_x,
  y = train_y
)

results <- knn_cv$execute()
#> 
#> CV fold: Fold1
#> 
#> CV fold: Fold2
#> CV progress [====================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> CV fold: Fold3
#> CV progress [========================================================================================================] 3/3 (100%)
#>                                                                                                                                   

head(results)
#>     fold performance  k l
#> 1: Fold1   0.8926554 68 0
#> 2: Fold2   0.8747646 68 0
#> 3: Fold3   0.8596987 68 0


## -----------------------------------------------------------------------------
knn_cv_nested_bayesian <- mlexperiments::MLNestedCV$new(
  learner = LearnerKnn$new(),
  strategy = "bayesian",
  fold_list = fold_list,
  k_tuning = 3L,
  ncores = ncores,
  seed = seed
)

knn_cv_nested_bayesian$parameter_grid <- param_list_knn
knn_cv_nested_bayesian$parameter_bounds <- knn_bounds
knn_cv_nested_bayesian$split_type <- "stratified"
knn_cv_nested_bayesian$optim_args <- optim_args

knn_cv_nested_bayesian$predict_args <- list(type = "response")
knn_cv_nested_bayesian$performance_metric <- metric("ACC")

# set data
knn_cv_nested_bayesian$set_data(
  x = train_x,
  y = train_y
)

results <- knn_cv_nested_bayesian$execute()
#> 
#> CV fold: Fold1
#> 
#> Registering parallel backend using 4 cores.
#> 
#> CV fold: Fold2
#> CV progress [====================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> Registering parallel backend using 4 cores.
#> 
#> CV fold: Fold3
#> CV progress [========================================================================================================] 3/3 (100%)
#>                                                                                                                                   
#> Registering parallel backend using 4 cores.

head(results)
#>     fold performance  k l
#> 1: Fold1   0.8926554 68 0
#> 2: Fold2   0.8747646 68 0
#> 3: Fold3   0.8596987 68 0


## -----------------------------------------------------------------------------
knn_cv_nested_grid <- mlexperiments::MLNestedCV$new(
  learner = LearnerKnn$new(),
  strategy = "grid",
  fold_list = fold_list,
  k_tuning = 3L,
  ncores = ncores,
  seed = seed
)

knn_cv_nested_grid$parameter_grid <- param_list_knn
knn_cv_nested_grid$split_type <- "stratified"

knn_cv_nested_grid$predict_args <- list(type = "response")
knn_cv_nested_grid$performance_metric <- metric("ACC")

# set data
knn_cv_nested_grid$set_data(
  x = train_x,
  y = train_y
)

results <- knn_cv_nested_grid$execute()
#> 
#> CV fold: Fold1
#> 
#> Parameter settings [=====================>---------------------------------------------------------------------------] 2/9 ( 22%)
#> Parameter settings [===============================>-----------------------------------------------------------------] 3/9 ( 33%)
#> Parameter settings [==========================================>------------------------------------------------------] 4/9 ( 44%)
#> Parameter settings [=====================================================>-------------------------------------------] 5/9 ( 56%)
#> Parameter settings [================================================================>--------------------------------] 6/9 ( 67%)
#> Parameter settings [==========================================================================>----------------------] 7/9 ( 78%)
#> Parameter settings [=====================================================================================>-----------] 8/9 ( 89%)
#> Parameter settings [=================================================================================================] 9/9 (100%)                                                                                                                                  
#> CV fold: Fold2
#> CV progress [====================================================================>-----------------------------------] 2/3 ( 67%)
#> 
#> Parameter settings [=====================>---------------------------------------------------------------------------] 2/9 ( 22%)
#> Parameter settings [===============================>-----------------------------------------------------------------] 3/9 ( 33%)
#> Parameter settings [==========================================>------------------------------------------------------] 4/9 ( 44%)
#> Parameter settings [=====================================================>-------------------------------------------] 5/9 ( 56%)
#> Parameter settings [================================================================>--------------------------------] 6/9 ( 67%)
#> Parameter settings [==========================================================================>----------------------] 7/9 ( 78%)
#> Parameter settings [=====================================================================================>-----------] 8/9 ( 89%)
#> Parameter settings [=================================================================================================] 9/9 (100%)                                                                                                                                  
#> CV fold: Fold3
#> CV progress [========================================================================================================] 3/3 (100%)
#>                                                                                                                                   
#> Parameter settings [=====================>---------------------------------------------------------------------------] 2/9 ( 22%)
#> Parameter settings [===============================>-----------------------------------------------------------------] 3/9 ( 33%)
#> Parameter settings [==========================================>------------------------------------------------------] 4/9 ( 44%)
#> Parameter settings [=====================================================>-------------------------------------------] 5/9 ( 56%)
#> Parameter settings [================================================================>--------------------------------] 6/9 ( 67%)
#> Parameter settings [==========================================================================>----------------------] 7/9 ( 78%)
#> Parameter settings [=====================================================================================>-----------] 8/9 ( 89%)
#> Parameter settings [=================================================================================================] 9/9 (100%)                                                                                                                                  

head(results)
#>     fold performance  k l
#> 1: Fold1   0.8945386 52 0
#> 2: Fold2   0.8747646 68 0
#> 3: Fold3   0.8596987 68 0


## ----include=FALSE------------------------------------------------------------
# nolint end

