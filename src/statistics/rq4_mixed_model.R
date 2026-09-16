# S6.05 RQ4 primary mixed-model implementation.
# Locked protocol: Y ~ X + architecture_budget_cell + (1 | training_seed), REML, Kenward-Roger.

RQ4_ARCHITECTURES <- c("R50", "Swin-T")
RQ4_BUDGETS <- c("1%", "5%", "10%", "20%")
RQ4_CELL_LEVELS <- c(
  "R50__1pct", "R50__5pct", "R50__10pct", "R50__20pct",
  "Swin-T__1pct", "Swin-T__5pct", "Swin-T__10pct", "Swin-T__20pct"
)
RQ4_OFFICIAL_TRAINING_SEEDS <- c(
  204886845, 1480646854, 1798418854, 2045683682, 1814859839,
  1603952859, 1878351743, 875651179, 477581743, 869675675
)
RQ4_EXPECTED_ROWS <- 80L
RQ4_EXPECTED_SEED_COUNT <- 10L
RQ4_EXPECTED_CELL_COUNT <- 8L
RQ4_Q_SCALE <- 0.01
RQ4_CI_LEVEL <- 0.95
RQ4_FORMULA_TEXT <- "Y ~ X + architecture_budget_cell + (1 | training_seed)"
RQ4_ESTIMATION <- "REML"
RQ4_FIXED_EFFECT_INFERENCE <- "Kenward-Roger"
RQ4_BETA_ALTERNATIVE <- "greater"
RQ4_EFFECT_UNIT <- "downstream AP-point change per +1 pseudo AP point"

rq4_cell_id <- function(architecture, budget) {
  budget_tag <- sub("%$", "pct", budget)
  paste0(architecture, "__", budget_tag)
}

validate_rq4_input <- function(df) {
  required <- c("architecture", "budget", "seed_index", "training_seed", "Q_pseudo", "delta_mAP_test")
  if (!is.data.frame(df)) stop("RQ4 input must be a data.frame")
  missing <- setdiff(required, names(df))
  if (length(missing) > 0L) stop(paste("missing required columns:", paste(missing, collapse=", ")))
  if (nrow(df) != RQ4_EXPECTED_ROWS) stop(sprintf("RQ4 requires exactly %d rows; observed=%d", RQ4_EXPECTED_ROWS, nrow(df)))
  if (anyNA(df[required])) stop("RQ4 input contains missing values")
  if (!all(df$architecture %in% RQ4_ARCHITECTURES)) stop("RQ4 input contains invalid architecture")
  if (!all(df$budget %in% RQ4_BUDGETS)) stop("RQ4 input contains invalid budget")
  if (!all(is.finite(df$Q_pseudo))) stop("Q_pseudo must be finite")
  if (!all(is.finite(df$delta_mAP_test))) stop("delta_mAP_test must be finite")
  if (!all(df$seed_index %in% seq_len(RQ4_EXPECTED_SEED_COUNT))) stop("seed_index must be in 1..10")
  seed_map <- setNames(RQ4_OFFICIAL_TRAINING_SEEDS, seq_len(RQ4_EXPECTED_SEED_COUNT))
  expected_seed <- unname(seed_map[as.character(df$seed_index)])
  if (!all(df$training_seed == expected_seed)) stop("training_seed does not match locked seed_index mapping")
  cell <- rq4_cell_id(df$architecture, df$budget)
  if (!all(cell %in% RQ4_CELL_LEVELS)) stop("invalid architecture_budget_cell encoding")
  key <- paste(cell, df$training_seed, sep="::")
  if (anyDuplicated(key)) stop("duplicate architecture-budget-seed row")
  tab <- table(factor(cell, levels=RQ4_CELL_LEVELS))
  if (!all(tab == RQ4_EXPECTED_SEED_COUNT)) stop("each architecture-budget cell must contain exactly 10 seeds")
  for (cl in RQ4_CELL_LEVELS) {
    observed <- sort(df$training_seed[cell == cl])
    expected <- sort(RQ4_OFFICIAL_TRAINING_SEEDS)
    if (!identical(as.numeric(observed), as.numeric(expected))) stop(paste("cell does not contain exact locked seed set:", cl))
  }
  invisible(TRUE)
}

prepare_rq4_analysis_dataset <- function(df) {
  validate_rq4_input(df)
  df$architecture <- factor(df$architecture, levels=RQ4_ARCHITECTURES)
  df$budget <- factor(df$budget, levels=RQ4_BUDGETS)
  df$architecture_budget_cell <- rq4_cell_id(as.character(df$architecture), as.character(df$budget))
  df$architecture_budget_cell <- factor(df$architecture_budget_cell, levels=RQ4_CELL_LEVELS)
  df$training_seed <- factor(df$training_seed, levels=RQ4_OFFICIAL_TRAINING_SEEDS)
  ord <- order(df$architecture_budget_cell, df$seed_index)
  df <- df[ord, , drop=FALSE]
  df$Q_WC <- ave(df$Q_pseudo, df$architecture_budget_cell, FUN=function(z) z - mean(z))
  df$X_per_1_pseudo_AP_point <- df$Q_WC / RQ4_Q_SCALE
  df$Y <- df$delta_mAP_test
  centered <- tapply(df$Q_WC, df$architecture_budget_cell, sum)
  if (any(abs(centered) > 1e-12)) stop("within-cell centering invariant failed")
  rownames(df) <- NULL
  df
}

fit_rq4_primary_mixed_model <- function(df) {
  for (pkg in c("lme4", "lmerTest", "pbkrtest")) {
    if (!requireNamespace(pkg, quietly=TRUE)) stop(paste("required R package unavailable:", pkg))
  }
  dat <- prepare_rq4_analysis_dataset(df)
  model <- lmerTest::lmer(
    Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell + (1 | training_seed),
    data=dat,
    REML=TRUE
  )
  sm <- summary(model, ddf=RQ4_FIXED_EFFECT_INFERENCE)
  ct <- coef(sm)
  term <- "X_per_1_pseudo_AP_point"
  if (!(term %in% rownames(ct))) stop("RQ4 slope term missing from Kenward-Roger coefficient table")
  estimate <- unname(ct[term, "Estimate"])
  se <- unname(ct[term, "Std. Error"])
  df_kr <- unname(ct[term, "df"])
  t_value <- unname(ct[term, "t value"])
  p_one_sided <- stats::pt(t_value, df=df_kr, lower.tail=FALSE)
  crit <- stats::qt(1 - (1 - RQ4_CI_LEVEL) / 2, df=df_kr)
  ci_low <- estimate - crit * se
  ci_high <- estimate + crit * se
  conv_messages <- model@optinfo$conv$lme4$messages
  converged <- is.null(conv_messages)
  singular <- lme4::isSingular(model, tol=1e-4)
  result <- list(
    formula=RQ4_FORMULA_TEXT,
    estimation=RQ4_ESTIMATION,
    fixed_effect_inference=RQ4_FIXED_EFFECT_INFERENCE,
    beta_Q_alternative=RQ4_BETA_ALTERNATIVE,
    effect_unit=RQ4_EFFECT_UNIT,
    effect_estimate=estimate,
    standard_error=se,
    df=df_kr,
    t_statistic=t_value,
    two_sided_95_ci=list(low=ci_low, high=ci_high),
    one_sided_p_value=p_one_sided,
    convergence_status=if (converged) "CONVERGED" else "NOT_CONVERGED",
    convergence_messages=if (converged) character(0) else as.character(conv_messages),
    singularity_status=if (singular) "SINGULAR" else "NON_SINGULAR",
    n_rows=nrow(dat),
    n_training_seeds=length(unique(dat$training_seed)),
    n_architecture_budget_cells=length(unique(dat$architecture_budget_cell))
  )
  list(analysis_dataset=dat, model=model, result=result)
}
