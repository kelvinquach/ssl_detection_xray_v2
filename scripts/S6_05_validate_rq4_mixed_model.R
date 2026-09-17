#!/usr/bin/env Rscript
# S6.05 validator for locked RQ4 REML + Kenward-Roger mixed model.

args <- commandArgs(trailingOnly=TRUE)
repo_root <- if(length(args)>=1L) normalizePath(args[[1]], mustWork=TRUE) else normalizePath(".", mustWork=TRUE)
output_path <- if(length(args)>=2L) args[[2]] else ""
core_path <- file.path(repo_root,"src","statistics","rq4_mixed_model.R")
if(!file.exists(core_path)) stop("missing RQ4 core implementation")
source(core_path)

sha256_file <- function(path) {
  out <- system2("sha256sum", path, stdout=TRUE, stderr=TRUE)
  if(length(out) < 1L) stop(paste("sha256sum failed for", path))
  value <- strsplit(out[[1]], "[[:space:]]+")[[1]][1]
  if(!grepl("^[0-9a-f]{64}$", value)) stop(paste("invalid SHA-256 for", path))
  value
}

validator_path <- file.path(repo_root,"scripts","S6_05_validate_rq4_mixed_model.R")
HISTORICAL_S6_04_EVIDENCE_SHA256 <- "3984a439d209a96d58f4a15924b67e50690a8d9580a96cbbee5852fc076682c1"
for(pkg in c("lme4","lmerTest","pbkrtest","jsonlite")) {
  if(!requireNamespace(pkg,quietly=TRUE)) stop(paste("required package unavailable:",pkg))
}
checks <- list()
add_check <- function(name,ok,observed){
  checks[[length(checks)+1L]] <<- list(name=name,status=if(isTRUE(ok)) "PASS" else "FAIL",observed=observed)
}
expect_failure <- function(name,expr){
  msg <- NULL
  ok <- FALSE
  tryCatch({force(expr)},error=function(e){ok <<- TRUE; msg <<- conditionMessage(e)})
  add_check(name,ok,if(is.null(msg)) "NO_ERROR" else msg)
}

# Contract checks.
add_check("contract::formula",identical(RQ4_FORMULA_TEXT,"Y ~ X + architecture_budget_cell + (1 | training_seed)"),RQ4_FORMULA_TEXT)
add_check("contract::estimation",identical(RQ4_ESTIMATION,"REML"),RQ4_ESTIMATION)
add_check("contract::fixed_effect_inference",identical(RQ4_FIXED_EFFECT_INFERENCE,"Kenward-Roger"),RQ4_FIXED_EFFECT_INFERENCE)
add_check("contract::beta_alternative",identical(RQ4_BETA_ALTERNATIVE,"greater"),RQ4_BETA_ALTERNATIVE)
add_check("contract::effect_unit",identical(RQ4_EFFECT_UNIT,"downstream AP-point change per +1 pseudo AP point"),RQ4_EFFECT_UNIT)
add_check("contract::architectures",identical(RQ4_ARCHITECTURES,c("R50","Swin-T")),RQ4_ARCHITECTURES)
add_check("contract::budgets",identical(RQ4_BUDGETS,c("1%","5%","10%","20%")),RQ4_BUDGETS)
add_check("contract::cell_count",RQ4_EXPECTED_CELL_COUNT==8L,RQ4_EXPECTED_CELL_COUNT)
add_check("contract::seed_count",RQ4_EXPECTED_SEED_COUNT==10L,RQ4_EXPECTED_SEED_COUNT)
add_check("contract::row_count",RQ4_EXPECTED_ROWS==80L,RQ4_EXPECTED_ROWS)
add_check("contract::q_scale",isTRUE(all.equal(RQ4_Q_SCALE,0.01,tolerance=0)),RQ4_Q_SCALE)
add_check("contract::ci_level",isTRUE(all.equal(RQ4_CI_LEVEL,0.95,tolerance=0)),RQ4_CI_LEVEL)
add_check("contract::cell_encoding",identical(RQ4_CELL_LEVELS,c("R50__1pct","R50__5pct","R50__10pct","R50__20pct","Swin-T__1pct","Swin-T__5pct","Swin-T__10pct","Swin-T__20pct")),RQ4_CELL_LEVELS)
add_check("contract::ordered_training_seeds",identical(as.numeric(RQ4_OFFICIAL_TRAINING_SEEDS),c(204886845,1480646854,1798418854,2045683682,1814859839,1603952859,1878351743,875651179,477581743,869675675)),RQ4_OFFICIAL_TRAINING_SEEDS)

# Deterministic synthetic 80-row fixture.
base_q <- c(0.28,0.32,0.36,0.40,0.30,0.34,0.38,0.42)
cell_intercept <- c(0.10,0.20,0.30,0.40,0.15,0.25,0.35,0.45)
seed_effect <- c(-0.18,-0.12,-0.08,-0.03,0.00,0.02,0.05,0.09,0.11,0.14)
q_pattern <- seq(-0.045,0.045,length.out=10)
beta_true <- 0.12
rows <- vector("list",80L)
k <- 1L
cell_index <- 1L
for(a in RQ4_ARCHITECTURES){
  for(b in RQ4_BUDGETS){
    shift <- (cell_index-1L) %% 10L
    idx <- ((seq_len(10L)-1L+shift) %% 10L)+1L
    q_dev <- q_pattern[idx]
    for(s in seq_len(10L)){
      residual <- 0.025*sin(cell_index*1.7+s*0.9)+0.012*cos(cell_index*0.4-s*1.3)
      x_true <- q_dev[s]/0.01
      rows[[k]] <- data.frame(architecture=a,budget=b,seed_index=s,training_seed=RQ4_OFFICIAL_TRAINING_SEEDS[s],Q_pseudo=base_q[cell_index]+q_dev[s],delta_mAP_test=cell_intercept[cell_index]+beta_true*x_true+seed_effect[s]+residual,stringsAsFactors=FALSE)
      k <- k+1L
    }
    cell_index <- cell_index+1L
  }
}
fixture <- do.call(rbind,rows)
fit <- fit_rq4_primary_mixed_model(fixture)
dat <- fit$analysis_dataset
r <- fit$result

add_check("positive::rows",nrow(dat)==80L,nrow(dat))
add_check("positive::cells",length(unique(dat$architecture_budget_cell))==8L,length(unique(dat$architecture_budget_cell)))
add_check("positive::seeds",length(unique(dat$training_seed))==10L,length(unique(dat$training_seed)))
add_check("positive::cell_counts",all(table(dat$architecture_budget_cell)==10L),as.list(table(dat$architecture_budget_cell)))
center_sums <- tapply(dat$Q_WC,dat$architecture_budget_cell,sum)
add_check("positive::within_cell_centering",max(abs(center_sums))<1e-12,max(abs(center_sums)))
add_check("positive::x_scaling",max(abs(dat$X_per_1_pseudo_AP_point-dat$Q_WC/0.01))<1e-12,max(abs(dat$X_per_1_pseudo_AP_point-dat$Q_WC/0.01)))
add_check("positive::formula",identical(r$formula,RQ4_FORMULA_TEXT),r$formula)
add_check("positive::reml",identical(r$estimation,"REML") && lme4::isREML(fit$model),list(result=r$estimation,isREML=lme4::isREML(fit$model)))
add_check("positive::kr_inference",identical(r$fixed_effect_inference,"Kenward-Roger"),r$fixed_effect_inference)
add_check("positive::alternative_greater",identical(r$beta_Q_alternative,"greater"),r$beta_Q_alternative)
add_check("positive::effect_estimate",isTRUE(all.equal(r$effect_estimate,0.12052482619044901,tolerance=1e-12)),r$effect_estimate)
add_check("positive::kr_se",isTRUE(all.equal(r$standard_error,0.00083286843267676904,tolerance=1e-12)),r$standard_error)
add_check("positive::kr_df",isTRUE(all.equal(r$df,62.031103884805717,tolerance=1e-10)),r$df)
add_check("positive::kr_t",isTRUE(all.equal(r$t_statistic,144.7105226489283,tolerance=1e-10)),r$t_statistic)
add_check("positive::one_sided_p",isTRUE(all.equal(r$one_sided_p_value,1.7546992557096755e-80,tolerance=1e-10)),r$one_sided_p_value)
add_check("positive::ci95_low",isTRUE(all.equal(r$two_sided_95_ci$low,0.11885996252200141,tolerance=1e-12)),r$two_sided_95_ci$low)
add_check("positive::ci95_high",isTRUE(all.equal(r$two_sided_95_ci$high,0.1221896898588966,tolerance=1e-12)),r$two_sided_95_ci$high)
add_check("positive::convergence",identical(r$convergence_status,"CONVERGED"),r$convergence_status)
add_check("positive::non_singular",identical(r$singularity_status,"NON_SINGULAR"),r$singularity_status)
add_check("positive::finite_inference",all(is.finite(c(r$effect_estimate,r$standard_error,r$df,r$t_statistic,r$one_sided_p_value,r$two_sided_95_ci$low,r$two_sided_95_ci$high))),unlist(r[c("effect_estimate","standard_error","df","t_statistic","one_sided_p_value")]))
add_check("positive::one_sided_direction",r$t_statistic>0 && r$one_sided_p_value<0.5,list(t=r$t_statistic,p=r$one_sided_p_value))

# Fail-closed fixtures.
expect_failure("negative::missing_row",validate_rq4_input(fixture[-1,,drop=FALSE]))
bad_seed <- fixture; bad_seed$training_seed[1] <- 123L
expect_failure("negative::bad_seed_mapping",validate_rq4_input(bad_seed))
dup <- fixture; dup[2,] <- dup[1,]
expect_failure("negative::duplicate_cell_seed",validate_rq4_input(dup))
bad_q <- fixture; bad_q$Q_pseudo[1] <- Inf
expect_failure("negative::nonfinite_q",validate_rq4_input(bad_q))
bad_y <- fixture; bad_y$delta_mAP_test[1] <- NaN
expect_failure("negative::nonfinite_y",validate_rq4_input(bad_y))
bad_arch <- fixture; bad_arch$architecture[1] <- "INVALID"
expect_failure("negative::invalid_architecture",validate_rq4_input(bad_arch))
bad_budget <- fixture; bad_budget$budget[1] <- "99%"
expect_failure("negative::invalid_budget",validate_rq4_input(bad_budget))

failed <- Filter(function(x)!identical(x$status,"PASS"),checks)
report <- list(
  task="S6.05",
  rq="RQ4",
  tracker_action="Implement mixed model",
  status=if(length(failed)==0L) "PASS" else "FAIL",
  checks_total=length(checks),
  checks_passed=length(checks)-length(failed),
  failed_count=length(failed),
  failed_checks=failed,
  scientific_contract=list(
    formula=RQ4_FORMULA_TEXT,
    estimation=RQ4_ESTIMATION,
    fixed_effect_inference=RQ4_FIXED_EFFECT_INFERENCE,
    beta_Q_alternative=RQ4_BETA_ALTERNATIVE,
    effect_unit=RQ4_EFFECT_UNIT,
    ci="TWO_SIDED_95_PERCENT",
    predictor_centering="WITHIN_ARCHITECTURE_BUDGET_CELL",
    predictor_scale="Q_WC / 0.01",
    inferential_replication_unit="TRAINING_SEED_TRAINED_RUN",
    n_training_seeds=10L,
    n_architecture_budget_cells=8L,
    max_rows=80L
  ),
  positive_fixture=list(
    synthetic_only=TRUE,
    true_beta=beta_true,
    rows=nrow(dat),
    architecture_budget_cells=length(unique(dat$architecture_budget_cell)),
    training_seeds=length(unique(dat$training_seed)),
    max_abs_within_cell_center_sum=max(abs(center_sums)),
    golden_result=list(
      effect_estimate=r$effect_estimate,
      kenward_roger_standard_error=r$standard_error,
      kenward_roger_df=r$df,
      t_statistic=r$t_statistic,
      one_sided_p_value=r$one_sided_p_value,
      ci95_low=r$two_sided_95_ci$low,
      ci95_high=r$two_sided_95_ci$high,
      convergence_status=r$convergence_status,
      singularity_status=r$singularity_status
    )
  ),
  scientific_boundaries=list(
    fixture_only_no_official_results=TRUE,
    official_training_authorized=FALSE,
    final_test_authorized=FALSE,
    test_used=FALSE,
    hidden_unlabeled_gt_used=FALSE,
    holm_f2_applied=FALSE,
    cr2_sensitivity_implemented=FALSE
  ),
  historical_s6_04_evidence_sha256_before_extension=HISTORICAL_S6_04_EVIDENCE_SHA256,
  source_sha256=list(
    "src/statistics/rq4_mixed_model.R"=sha256_file(core_path),
    "scripts/S6_05_validate_rq4_mixed_model.R"=sha256_file(validator_path)
  ),
  statistical_runtime=list(
    R=as.character(getRversion()),
    lme4=as.character(utils::packageVersion("lme4")),
    lmerTest=as.character(utils::packageVersion("lmerTest")),
    pbkrtest=as.character(utils::packageVersion("pbkrtest")),
    jsonlite=as.character(utils::packageVersion("jsonlite"))
  ),
  checks=checks
)
text <- jsonlite::toJSON(report,pretty=TRUE,auto_unbox=TRUE,null="null",digits=17)
cat(text,"\n")
if(nzchar(output_path)){
  dir.create(dirname(output_path),recursive=TRUE,showWarnings=FALSE)
  writeLines(text,output_path,useBytes=TRUE)
}
quit(status=if(identical(report$status,"PASS")) 0L else 1L)
