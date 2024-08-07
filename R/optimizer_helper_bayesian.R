.bayesian_optimize <- function(
    self, private,
    x,
    y,
    method_helper
) {
  stopifnot(
    "`parameter_bounds` must not be empty for Bayesian optimization" =
      !is.null(self$parameter_bounds),
    "`ncores` must be >1L when using Bayesian optimization" =
      private$ncores > 1L)
  if (self$optim_args$parallel) {
    stopifnot(
      "`learner$cluster_export` must not be empty when using Bayesian \
      optimization" = !is.null(self$learner$cluster_export))
    message(sprintf(
      "\nRegistering parallel backend using %s cores.",
      private$ncores
    ))

    cl <- kdry::pch_register_parallel(private$ncores)

    self$optim_args$iters.k <- private$ncores

    on.exit(
      expr = {
        kdry::pch_clean_up(cl)
        # reset random number generator
        RNGkind(kind = "default")
        invisible(gc())
      }
    )
    # cluster options
    cluster_options <- kdry::misc_subset_options("mlexperiments")
    # required for cluster export
    assign(
      x = "seed",
      value = private$seed
    )
    # export from current env
    parallel::clusterExport(
      cl = cl,
      varlist = c(
        "x", "y", "seed", "method_helper", # , "ncores" #, "cluster_load"
        "cluster_options"
      ),
      envir = environment()
    )

    # export from global env
    # if (private$method %in% options("mlexperiments.learner")) {
    if (self$learner$environment != -1L) {
      # https://stackoverflow.com/questions/67595111/r-package-design-how-to-
      # export-internal-functions-to-a-cluster
      #%ns <- asNamespace("mlexperiments")
      stopifnot(
        "`learner$environment` must be a character" =
          is.character(self$learner$environment)
      )
      ns <- asNamespace(self$learner$environment)
      parallel::clusterExport(
        cl = cl,
        #% varlist = unclass(
        #%   utils::lsf.str(
        #%     envir = ns,
        #%     all = TRUE
        #% )),
        varlist = self$learner$cluster_export,
        envir = as.environment(ns)
      )
    } else {
      parallel::clusterExport(
        cl = cl,
        varlist = self$learner$cluster_export,
        envir = -1L
      )
    }
    parallel::clusterSetRNGStream(
      cl = cl,
      iseed = private$seed
    )
    parallel::clusterEvalQ(
      cl = cl,
      expr = {
        # set cluster options
        options(cluster_options)
        #%lapply(cluster_load, library, character.only = TRUE)
        ## not necessary since using ::-notation everywhere
        RNGkind("L'Ecuyer-CMRG")
        # set seed in each job for reproducibility
        set.seed(seed) #, kind = "L'Ecuyer-CMRG")
      }
    )
  }

  # in any case, update gsPoints here, as default calculation fails when
  # calling bayesOpt with do.call
  if (identical(str2lang("pmax(100, length(bounds)^3)"),
                self$optim_args[["gsPoints"]])) {
    self$optim_args[["gsPoints"]] <- pmax(100, length(self$parameter_bounds)^3)
  }

  args <- kdry::list.append(
    list(
      # for each method, a bayesian scoring function is required
      # FUN = eval(parse(text = paste0(
      #   private$method, "_bsF"
      # ))),
      FUN = self$learner$bayesian_scoring_function,
      bounds = self$parameter_bounds,
      initGrid = method_helper$execute_params$parameter_grid
    ),
    self$optim_args
  )

  # avoid error when setting initGrid / or initPoints
  if (!is.null(method_helper$execute_params$parameter_grid)) {
    args <- args[names(args) != "initPoints"]
  } else {
    args <- args[names(args) != "initGrid"]
  }

  set.seed(private$seed)
  opt_obj <- do.call(ParBayesianOptimization::bayesOpt, args)
  return(opt_obj)
}

.bayesopt_postprocessing <- function(self, private, object) {
  stopifnot("`object` is not of class `bayesOpt`" =
              inherits(x = object, what = "bayesOpt"))
  exl_cols <- vapply(
    X = private$method_helper$execute_params$params_not_optimized,
    FUN = is.expression,
    FUN.VALUE = logical(1L)
  )

  # remove case_weights from params, otherwise displaying is very strange
  if ("case_weights" %in% names(self$learner_args)) {
    exl_cols["case_weights"] <- TRUE
  }
  optim_results <- cbind(
      data.table::as.data.table(object$scoreSummary),
      data.table::as.data.table(
        private$method_helper$execute_params$params_not_optimized[!exl_cols]
      )
    )

  colnames(optim_results)[grepl(
    pattern = "Iteration", x = colnames(optim_results))
  ] <- "setting_id"

  return(optim_results)
}
