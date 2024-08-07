.grid_optimize <- function(
    self, private,
    x,
    y,
    method_helper
) {
  stopifnot(
    "`parameter_grid` must have more than one row" =
      nrow(method_helper$execute_params$parameter_grid) > 1L
  )
  ngrid <- nrow(method_helper$execute_params$parameter_grid)
  # init a progress bar
  pb <- progress::progress_bar$new(
    format = "\nParameter settings [:bar] :current/:total (:percent)",
    total = ngrid
  )

  optim_results <- lapply(
    X = seq_len(ngrid),
    FUN = function(setting_id) {

      # increment progress bar
      pb$tick()

      # get the relevant row from param_list with the hyperparameters to use in
      # this loop
      # this code is required to have names arguments and allow selection of
      # expressions (which is not possible with data.table)
      grid_search_params <- sapply(
        X = colnames(method_helper$execute_params$parameter_grid),
        FUN = function(x) {
          mhcn <- colnames(method_helper$execute_params$parameter_grid)
          xcol <- which(mhcn == x)
          method_helper$execute_params$parameter_grid[
            setting_id, get(mhcn[xcol])
          ]
        },
        simplify = FALSE,
        USE.NAMES = TRUE
      )

      params <- .method_params_refactor(
        grid_search_params,
        method_helper
      )

      # FUN <- eval(parse(text = paste0(
      #   private$method, "_cv"
      # )))
      FUN <- self$learner$cross_validation # nolint

      fun_parameters <- list(
        "x" = x,
        "y" = y,
        "params" = params,
        "fold_list" = method_helper$fold_list,
        "ncores" = private$ncores,
        "seed" = private$seed
      )

      set.seed(private$seed)
      fit_grid <- do.call(FUN, fun_parameters)

      # remove case_weights from params, otherwise displaying is very strange
      if ("case_weights" %in% names(params)) {
        params$case_weights <- NULL
      }

      ret <- data.table::as.data.table(
        c(
          list("setting_id" = setting_id),
          fit_grid,
          params[
            setdiff(names(params), names(fit_grid))
          ]
        )
      )
      #%return(ret[, .SD, .SDcols = colnames(ret)[!sapply(ret, is.expression)]])
      return(ret)
    }
  )
  return(optim_results)
}
