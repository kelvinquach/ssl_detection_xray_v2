#!/usr/bin/env Rscript

# S10.1.13 RQ4 downstream orchestration.
# Reuses frozen S6 RQ4 primary + CR2 implementations unchanged.
# Adds only authority-required diagnostics and exactly-10 leave-one-training-seed influence refits.
# Seed-influence refits do NOT compute p-values/CIs and do NOT alter/exclude seeds from the primary model.

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 4L) {
  stop("usage: Rscript S10_1_13_compute_rq4.R <input_csv> <output_dir> <rq4_mixed_model.R> <rq4_cr2_sensitivity.R>")
}

input_csv <- normalizePath(args[[1]], mustWork=TRUE)
output_dir <- args[[2]]
mixed_model_path <- normalizePath(args[[3]], mustWork=TRUE)
cr2_path <- normalizePath(args[[4]], mustWork=TRUE)

dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

source(mixed_model_path)
source(cr2_path)

if (!requireNamespace("jsonlite", quietly=TRUE)) stop("jsonlite is required")

write_json <- function(path, obj) {
  jsonlite::write_json(
    obj,
    path=path,
    pretty=TRUE,
    auto_unbox=TRUE,
    digits=17,
    na="null",
    null="null"
  )
  cat("\n", file=path, append=TRUE)
}

write_csv_lf <- function(path, df) {
  con <- file(path, open="wb")
  on.exit(close(con), add=TRUE)
  txt <- paste(capture.output(
    utils::write.table(
      df,
      file="",
      sep=",",
      row.names=FALSE,
      col.names=TRUE,
      quote=TRUE,
      qmethod="double",
      na=""
    )
  ), collapse="\n")
  writeBin(charToRaw(paste0(txt, "\n")), con)
}

raw <- utils::read.csv(
  input_csv,
  stringsAsFactors=FALSE,
  check.names=FALSE
)

validate_rq4_input(raw)
cat("VALIDATE_RQ4_INPUT=PASS\n")

primary <- fit_rq4_primary_mixed_model(raw)
dat <- primary$analysis_dataset
model <- primary$model
primary_result <- primary$result

if (nrow(dat) != 80L) stop("RQ4 prepared dataset must have 80 rows")

analysis_dataset <- data.frame(
  architecture=as.character(dat$architecture),
  budget=as.character(dat$budget),
  architecture_budget_cell=as.character(dat$architecture_budget_cell),
  seed_index=as.integer(dat$seed_index),
  training_seed=as.numeric(as.character(dat$training_seed)),
  Q_pseudo=as.numeric(dat$Q_pseudo),
  Q_WC=as.numeric(dat$Q_WC),
  X_per_1_pseudo_AP_point=as.numeric(dat$X_per_1_pseudo_AP_point),
  delta_mAP_test=as.numeric(dat$delta_mAP_test),
  stringsAsFactors=FALSE
)

if (any(!is.finite(as.matrix(analysis_dataset[, c(
  "seed_index","training_seed","Q_pseudo","Q_WC",
  "X_per_1_pseudo_AP_point","delta_mAP_test"
)])))) stop("non-finite analysis dataset value")

center_sums <- tapply(
  analysis_dataset$Q_WC,
  analysis_dataset$architecture_budget_cell,
  sum
)
if (max(abs(center_sums)) >= 1e-12) stop("full-data within-cell centering failed")

analysis_csv <- file.path(output_dir, "rq4_analysis_dataset.raw.csv")
write_csv_lf(analysis_csv, analysis_dataset)

primary_json <- file.path(output_dir, "rq4_primary_mixed_model.raw.json")
write_json(primary_json, primary_result)

fitted_values <- as.numeric(stats::fitted(model))
residual_values <- as.numeric(stats::residuals(model))
if (length(fitted_values) != 80L || length(residual_values) != 80L) {
  stop("RQ4 diagnostics require 80 fitted/residual values")
}
if (any(!is.finite(c(fitted_values, residual_values)))) {
  stop("non-finite fitted/residual values")
}

qq <- stats::qqnorm(residual_values, plot.it=FALSE)
if (length(qq$x) != 80L || length(qq$y) != 80L) stop("RQ4 Q-Q requires 80 points")
if (any(!is.finite(c(qq$x, qq$y)))) stop("non-finite Q-Q values")

residual_points <- vector("list", 80L)
for (i in seq_len(80L)) {
  residual_points[[i]] <- list(
    row_index=as.integer(i),
    architecture=analysis_dataset$architecture[[i]],
    budget=analysis_dataset$budget[[i]],
    seed_index=as.integer(analysis_dataset$seed_index[[i]]),
    training_seed=as.numeric(analysis_dataset$training_seed[[i]]),
    fitted=as.numeric(fitted_values[[i]]),
    residual=as.numeric(residual_values[[i]])
  )
}

qq_points <- vector("list", 80L)
for (i in seq_len(80L)) {
  qq_points[[i]] <- list(
    qq_index=as.integer(i),
    theoretical_quantile=as.numeric(qq$x[[i]]),
    sample_residual=as.numeric(qq$y[[i]])
  )
}

full_beta <- as.numeric(primary_result$effect_estimate)

refit_one <- function(omit_seed) {
  d <- raw[raw$training_seed != omit_seed, , drop=FALSE]

  if (nrow(d) != 72L) stop("RQ4 seed-influence refit must have exactly 72 rows")
  if (length(unique(d$training_seed)) != 9L) stop("RQ4 seed-influence refit must retain exactly 9 seeds")

  d$architecture <- factor(d$architecture, levels=RQ4_ARCHITECTURES)
  d$budget <- factor(d$budget, levels=RQ4_BUDGETS)

  cell_text <- paste0(
    as.character(d$architecture),
    "__",
    sub("%", "pct", as.character(d$budget), fixed=TRUE)
  )
  d$architecture_budget_cell <- factor(cell_text, levels=RQ4_CELL_LEVELS)

  retained_seed_levels <- as.character(
    RQ4_OFFICIAL_TRAINING_SEEDS[RQ4_OFFICIAL_TRAINING_SEEDS != omit_seed]
  )
  d$training_seed <- factor(
    as.character(d$training_seed),
    levels=retained_seed_levels
  )

  cell_counts <- table(d$architecture_budget_cell)
  if (!all(as.integer(cell_counts) == 9L)) {
    stop("each architecture-budget cell must retain exactly 9 rows")
  }

  d$Q_WC <- ave(
    d$Q_pseudo,
    d$architecture_budget_cell,
    FUN=function(z) z - mean(z)
  )
  d$X_per_1_pseudo_AP_point <- d$Q_WC / RQ4_Q_SCALE
  d$Y <- d$delta_mAP_test

  refit_center_sums <- tapply(
    d$Q_WC,
    d$architecture_budget_cell,
    sum
  )
  max_center_error <- max(abs(refit_center_sums))
  if (!is.finite(max_center_error) || max_center_error >= 1e-12) {
    stop("RQ4 seed-influence within-cell centering failed")
  }

  m <- lmerTest::lmer(
    Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell + (1 | training_seed),
    data=d,
    REML=TRUE
  )

  beta <- as.numeric(
    lme4::fixef(m)[["X_per_1_pseudo_AP_point"]]
  )
  if (!is.finite(beta)) stop("non-finite RQ4 seed-influence beta")

  conv_messages <- m@optinfo$conv$lme4$messages
  converged <- is.null(conv_messages) || length(conv_messages) == 0L
  singular <- lme4::isSingular(m, tol=1e-4)

  data.frame(
    omitted_seed_index=as.integer(match(omit_seed, RQ4_OFFICIAL_TRAINING_SEEDS)),
    omitted_training_seed=as.numeric(omit_seed),
    retained_seed_count=9L,
    row_count=72L,
    effect_estimate=beta,
    delta_from_full=beta-full_beta,
    effect_direction=if (beta > 0) "POSITIVE" else if (beta < 0) "NEGATIVE" else "ZERO",
    convergence_status=if (converged) "CONVERGED" else "NOT_CONVERGED",
    singularity_status=if (singular) "SINGULAR" else "NON_SINGULAR",
    max_abs_within_cell_center_sum=max_center_error,
    stringsAsFactors=FALSE
  )
}

seed_influence <- do.call(
  rbind,
  lapply(RQ4_OFFICIAL_TRAINING_SEEDS, refit_one)
)

if (nrow(seed_influence) != 10L) stop("RQ4 seed influence requires exactly 10 analyses")
if (!all(seed_influence$retained_seed_count == 9L)) stop("RQ4 seed influence must retain 9 seeds")
if (!all(seed_influence$row_count == 72L)) stop("RQ4 seed influence must use 72 rows")
if (any(!is.finite(seed_influence$effect_estimate))) stop("non-finite seed-influence effect estimate")

seed_influence_csv <- file.path(output_dir, "rq4_seed_influence.raw.csv")
write_csv_lf(seed_influence_csv, seed_influence)

diagnostics_obj <- list(
  analysis="RQ4_PRIMARY_MIXED_MODEL_DIAGNOSTICS",
  role="DIAGNOSTIC_ONLY",
  diagnostics_not_used_for_model_selection=TRUE,
  seed_influence_not_used_for_primary_seed_exclusion=TRUE,
  primary_model=list(
    convergence_status=primary_result$convergence_status,
    convergence_messages=primary_result$convergence_messages,
    singularity_status=primary_result$singularity_status,
    row_count=80L,
    training_seed_count=10L,
    architecture_budget_cell_count=8L,
    effect_estimate=full_beta
  ),
  residual_vs_fitted=list(
    target="MODEL_RESIDUALS_VS_FITTED",
    point_count=80L,
    residual_mean=as.numeric(mean(residual_values)),
    residual_sd=as.numeric(stats::sd(residual_values)),
    points=residual_points
  ),
  qq=list(
    target="MODEL_RESIDUALS",
    point_count=80L,
    points=qq_points
  ),
  seed_influence=list(
    method="EXACTLY_10_LEAVE_ONE_TRAINING_SEED_OUT_REFITS",
    refit_formula="Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell + (1 | training_seed)",
    estimation="REML",
    predictor_recentered_within_architecture_budget_cell_for_each_refit=TRUE,
    analysis_count=10L,
    retained_seed_count_per_analysis=9L,
    row_count_per_analysis=72L,
    loso_p_value_computed=FALSE,
    loso_ci_computed=FALSE,
    primary_model_unchanged=TRUE,
    primary_seed_exclusion_allowed=FALSE,
    converged_count=as.integer(sum(seed_influence$convergence_status == "CONVERGED")),
    singular_count=as.integer(sum(seed_influence$singularity_status == "SINGULAR")),
    positive_effect_count=as.integer(sum(seed_influence$effect_direction == "POSITIVE")),
    negative_effect_count=as.integer(sum(seed_influence$effect_direction == "NEGATIVE")),
    zero_effect_count=as.integer(sum(seed_influence$effect_direction == "ZERO")),
    min_effect_estimate=as.numeric(min(seed_influence$effect_estimate)),
    max_effect_estimate=as.numeric(max(seed_influence$effect_estimate)),
    max_abs_delta_from_full=as.numeric(max(abs(seed_influence$delta_from_full)))
  )
)

diagnostics_json <- file.path(output_dir, "rq4_diagnostics.raw.json")
write_json(diagnostics_json, diagnostics_obj)

cr2 <- run_rq4_cr2_sensitivity(raw)
cr2_obj <- list(
  analysis=cr2$analysis,
  sensitivity_only=cr2$sensitivity_only,
  replaces_primary_kr=cr2$replaces_primary_kr,
  formula=cr2$formula,
  covariance_type=cr2$covariance_type,
  cluster_variable=cr2$cluster_variable,
  cluster_count=cr2$cluster_count,
  architecture_budget_cell_count=cr2$architecture_budget_cell_count,
  row_count=cr2$row_count,
  df_method=cr2$df_method,
  hypothesis=cr2$hypothesis,
  effect=cr2$effect
)

cr2_json <- file.path(output_dir, "rq4_cr2_sensitivity.raw.json")
write_json(cr2_json, cr2_obj)

runtime_obj <- list(
  R=as.character(getRversion()),
  packages=list(
    lme4=as.character(utils::packageVersion("lme4")),
    lmerTest=as.character(utils::packageVersion("lmerTest")),
    pbkrtest=as.character(utils::packageVersion("pbkrtest")),
    jsonlite=as.character(utils::packageVersion("jsonlite")),
    clubSandwich=as.character(utils::packageVersion("clubSandwich"))
  )
)
runtime_json <- file.path(output_dir, "rq4_runtime_environment.raw.json")
write_json(runtime_json, runtime_obj)

cat("=== S10.1.13 RQ4 RAW MATERIALIZATION ===\n")
cat("ANALYSIS_ROWS=", nrow(analysis_dataset), "\n", sep="")
cat("PRIMARY_EFFECT_ESTIMATE=", format(primary_result$effect_estimate, digits=17), "\n", sep="")
cat("PRIMARY_ONE_SIDED_P=", format(primary_result$one_sided_p_value, digits=17), "\n", sep="")
cat("PRIMARY_CONVERGENCE=", primary_result$convergence_status, "\n", sep="")
cat("PRIMARY_SINGULARITY=", primary_result$singularity_status, "\n", sep="")
cat("RESIDUAL_VS_FITTED_POINTS=80\n")
cat("QQ_POINTS=80\n")
cat("SEED_INFLUENCE_ANALYSES=", nrow(seed_influence), "\n", sep="")
cat("SEED_INFLUENCE_CONVERGED_COUNT=", sum(seed_influence$convergence_status == "CONVERGED"), "\n", sep="")
cat("SEED_INFLUENCE_SINGULAR_COUNT=", sum(seed_influence$singularity_status == "SINGULAR"), "\n", sep="")
cat("SEED_INFLUENCE_POSITIVE_COUNT=", sum(seed_influence$effect_direction == "POSITIVE"), "\n", sep="")
cat("SEED_INFLUENCE_MIN_BETA=", format(min(seed_influence$effect_estimate), digits=17), "\n", sep="")
cat("SEED_INFLUENCE_MAX_BETA=", format(max(seed_influence$effect_estimate), digits=17), "\n", sep="")
cat("SEED_INFLUENCE_MAX_ABS_DELTA_FROM_FULL=", format(max(abs(seed_influence$delta_from_full)), digits=17), "\n", sep="")
cat("SEED_INFLUENCE_LOSO_P_VALUE_COMPUTED=FALSE\n")
cat("SEED_INFLUENCE_LOSO_CI_COMPUTED=FALSE\n")
cat("CR2_EFFECT_ESTIMATE=", format(cr2$effect$estimate, digits=17), "\n", sep="")
cat("CR2_ONE_SIDED_P=", format(cr2$effect$p_one_sided_greater, digits=17), "\n", sep="")
cat("S10_1_13_RQ4_RAW_MATERIALIZATION=PASS\n")
