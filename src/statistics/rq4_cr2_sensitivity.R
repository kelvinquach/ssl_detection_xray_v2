# S6.14 RQ4 CR2/Satterthwaite sensitivity analysis core.
# Sensitivity only: does not replace the primary KR mixed-effects analysis.

RQ4_CR2_FORMULA_TEXT <- "Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell"
RQ4_CR2_COVARIANCE_TYPE <- "CR2"
RQ4_CR2_CLUSTER_VARIABLE <- "training_seed"
RQ4_CR2_DF_METHOD <- "Satterthwaite"
RQ4_CR2_ALTERNATIVE <- "greater"
RQ4_CR2_CI_LEVEL <- 0.95
RQ4_CR2_EFFECT_UNIT <- "delta_mAP_test change per +1 pseudo AP point"

assert_rq4_cr2_dependencies <- function() {
  if (!requireNamespace("clubSandwich", quietly = TRUE)) {
    stop("clubSandwich is required for S6.14 CR2 sensitivity analysis")
  }
  observed <- as.character(utils::packageVersion("clubSandwich"))
  if (observed != "0.6.2") {
    stop(sprintf("clubSandwich version mismatch: expected 0.6.2, observed %s", observed))
  }
  exports <- getNamespaceExports("clubSandwich")
  required <- c("vcovCR", "coef_test", "conf_int")
  missing <- setdiff(required, exports)
  if (length(missing) > 0L) {
    stop(sprintf("Missing required clubSandwich exports: %s", paste(missing, collapse = ", ")))
  }
  invisible(TRUE)
}

run_rq4_cr2_sensitivity <- function(df) {
  assert_rq4_cr2_dependencies()

  if (!exists("validate_rq4_input", mode = "function")) {
    stop("validate_rq4_input() from rq4_mixed_model.R must be loaded")
  }
  if (!exists("prepare_rq4_analysis_dataset", mode = "function")) {
    stop("prepare_rq4_analysis_dataset() from rq4_mixed_model.R must be loaded")
  }

  validate_rq4_input(df)
  dat <- prepare_rq4_analysis_dataset(df)

  if (nrow(dat) != 80L) stop("RQ4 CR2 sensitivity requires exactly 80 rows")
  if (length(unique(dat$training_seed)) != 10L) stop("RQ4 CR2 sensitivity requires exactly 10 training_seed clusters")
  if (length(unique(dat$architecture_budget_cell)) != 8L) stop("RQ4 CR2 sensitivity requires exactly 8 architecture-budget cells")

  fit <- stats::lm(
    Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell,
    data = dat
  )

  vcov_cr2 <- clubSandwich::vcovCR(
    fit,
    cluster = dat$training_seed,
    type = "CR2"
  )

  coef_table <- clubSandwich::coef_test(
    fit,
    vcov = vcov_cr2,
    test = "Satterthwaite",
    alternative = RQ4_CR2_ALTERNATIVE
  )

  slope_term <- "X_per_1_pseudo_AP_point"
  if (!(slope_term %in% rownames(coef_table))) {
    stop("RQ4 CR2 slope term missing from coefficient table")
  }
  slope_row <- coef_table[slope_term, , drop = FALSE]

  ci_table <- clubSandwich::conf_int(
    fit,
    vcov = vcov_cr2,
    level = RQ4_CR2_CI_LEVEL,
    test = "Satterthwaite"
  )
  if (!(slope_term %in% rownames(ci_table))) {
    stop("RQ4 CR2 slope term missing from confidence-interval table")
  }
  ci_row <- ci_table[slope_term, , drop = FALSE]

  beta <- as.numeric(slope_row$beta)
  se <- as.numeric(slope_row$SE)
  t_stat <- as.numeric(slope_row$tstat)
  df_satt <- as.numeric(slope_row$df_Satt)
  p_one_sided_greater <- as.numeric(slope_row$p_Satt)
  ci_lower <- as.numeric(ci_row$CI_L)
  ci_upper <- as.numeric(ci_row$CI_U)

  numeric_values <- c(beta, se, t_stat, df_satt, p_one_sided_greater, ci_lower, ci_upper)
  if (any(!is.finite(numeric_values))) stop("Non-finite RQ4 CR2 sensitivity result")
  if (se <= 0) stop("RQ4 CR2 sensitivity SE must be positive")
  if (df_satt <= 0) stop("RQ4 CR2 sensitivity Satterthwaite df must be positive")
  if (p_one_sided_greater < 0 || p_one_sided_greater > 1) stop("Invalid one-sided p-value")
  if (ci_lower > ci_upper) stop("Invalid confidence interval ordering")

  list(
    analysis = "RQ4_CR2_SATTERTHWAITE_SENSITIVITY",
    sensitivity_only = TRUE,
    replaces_primary_kr = FALSE,
    formula = RQ4_CR2_FORMULA_TEXT,
    covariance_type = RQ4_CR2_COVARIANCE_TYPE,
    cluster_variable = RQ4_CR2_CLUSTER_VARIABLE,
    cluster_count = length(unique(dat$training_seed)),
    architecture_budget_cell_count = length(unique(dat$architecture_budget_cell)),
    row_count = nrow(dat),
    df_method = RQ4_CR2_DF_METHOD,
    hypothesis = list(
      null = "beta_Q <= 0",
      alternative = "beta_Q > 0",
      p_value_sidedness = "one-sided greater"
    ),
    effect = list(
      term = slope_term,
      estimate = beta,
      standard_error_cr2 = se,
      t_statistic = t_stat,
      df_satterthwaite = df_satt,
      p_one_sided_greater = p_one_sided_greater,
      ci_level = RQ4_CR2_CI_LEVEL,
      ci_sidedness = "two-sided",
      ci_lower = ci_lower,
      ci_upper = ci_upper,
      unit = RQ4_CR2_EFFECT_UNIT
    ),
    model = fit,
    vcov_cr2 = vcov_cr2,
    analysis_dataset = dat
  )
}
