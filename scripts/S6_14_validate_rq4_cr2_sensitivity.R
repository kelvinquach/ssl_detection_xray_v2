# S6.14 validator for RQ4 CR2/Satterthwaite sensitivity.
source("/tmp/rq4_mixed_model.R")
source("/tmp/rq4_cr2_sensitivity.R")

checks <- 0L
pass <- function(name) {
  checks <<- checks + 1L
  cat(sprintf("PASS[%02d]=%s\n", checks, name))
}
assert_true <- function(x, name) {
  if (!isTRUE(x)) stop(sprintf("FAIL=%s", name))
  pass(name)
}
assert_close <- function(x, y, tol, name) {
  if (length(x) != 1L || length(y) != 1L || !is.finite(x) || !is.finite(y) || abs(x-y) > tol) {
    stop(sprintf("FAIL=%s observed=%s expected=%s", name, format(x, digits=17), format(y, digits=17)))
  }
  pass(name)
}

seeds <- c(204886845,1480646854,1798418854,2045683682,1814859839,1603952859,1878351743,875651179,477581743,869675675)
architectures <- c("R50","Swin-T")
budgets <- c("1%","5%","10%","20%")
q_shift <- seq(-0.045,0.045,length.out=10)
seed_effect <- c(-0.18,-0.12,-0.08,-0.03,0.00,0.02,0.05,0.09,0.11,0.14)
rows <- vector("list",80L)
k <- 1L
cell_index <- 0L
for (a in architectures) {
  for (b in budgets) {
    cell_index <- cell_index + 1L
    q_base <- 0.24 + 0.025 * cell_index
    cell_effect <- 0.035 * cell_index
    for (si in seq_along(seeds)) {
      residual <- 0.014*sin(cell_index*1.31 + si*0.77) + 0.006*cos(cell_index*0.43 - si*1.17)
      rows[[k]] <- data.frame(
        architecture=a,
        budget=b,
        seed_index=si,
        training_seed=seeds[si],
        Q_pseudo=q_base+q_shift[si],
        delta_mAP_test=cell_effect+0.13*(q_shift[si]/0.01)+seed_effect[si]+residual,
        stringsAsFactors=FALSE
      )
      k <- k + 1L
    }
  }
}
fixture <- do.call(rbind,rows)

assert_true(nrow(fixture)==80L,"fixture_has_80_rows")
assert_true(length(unique(fixture$training_seed))==10L,"fixture_has_10_training_seeds")
assert_true(length(unique(paste(fixture$architecture,fixture$budget,sep="__")))==8L,"fixture_has_8_cells")
validate_rq4_input(fixture)
pass("existing_s6_05_input_validator_accepts_fixture")
prepared <- prepare_rq4_analysis_dataset(fixture)
assert_true(nrow(prepared)==80L,"prepared_dataset_has_80_rows")
assert_true(max(abs(tapply(prepared$Q_WC,prepared$architecture_budget_cell,sum))) < 1e-12,"within_cell_q_centering")
assert_true(max(abs(prepared$X_per_1_pseudo_AP_point-prepared$Q_WC/0.01)) < 1e-12,"scaled_predictor_definition")

assert_rq4_cr2_dependencies()
pass("clubsandwich_dependency_contract")
res <- run_rq4_cr2_sensitivity(fixture)

assert_true(identical(res$analysis,"RQ4_CR2_SATTERTHWAITE_SENSITIVITY"),"analysis_identity")
assert_true(isTRUE(res$sensitivity_only),"sensitivity_only_true")
assert_true(identical(res$replaces_primary_kr,FALSE),"does_not_replace_primary_kr")
assert_true(identical(res$formula,"Y ~ X_per_1_pseudo_AP_point + architecture_budget_cell"),"formula_contract")
assert_true(identical(res$covariance_type,"CR2"),"cr2_covariance_contract")
assert_true(identical(res$cluster_variable,"training_seed"),"cluster_variable_contract")
assert_true(res$cluster_count==10L,"cluster_count_10")
assert_true(res$architecture_budget_cell_count==8L,"cell_count_8")
assert_true(res$row_count==80L,"analysis_row_count_80")
assert_true(identical(res$df_method,"Satterthwaite"),"satterthwaite_df_contract")
assert_true(identical(res$hypothesis$null,"beta_Q <= 0"),"directional_null_contract")
assert_true(identical(res$hypothesis$alternative,"beta_Q > 0"),"directional_alternative_contract")
assert_true(identical(res$hypothesis$p_value_sidedness,"one-sided greater"),"one_sided_p_contract")
assert_true(identical(res$effect$ci_sidedness,"two-sided"),"two_sided_ci_contract")
assert_close(res$effect$ci_level,0.95,1e-15,"ci_level_95pct")
assert_true(is.finite(res$effect$estimate),"finite_beta")
assert_true(is.finite(res$effect$standard_error_cr2) && res$effect$standard_error_cr2>0,"positive_finite_cr2_se")
assert_true(is.finite(res$effect$df_satterthwaite) && res$effect$df_satterthwaite>0,"positive_finite_satterthwaite_df")
assert_true(is.finite(res$effect$p_one_sided_greater) && res$effect$p_one_sided_greater>=0 && res$effect$p_one_sided_greater<=1,"valid_one_sided_p")
assert_true(res$effect$ci_lower<=res$effect$estimate && res$effect$estimate<=res$effect$ci_upper,"estimate_inside_95pct_ci")
assert_true(identical(res$effect$unit,"delta_mAP_test change per +1 pseudo AP point"),"effect_unit_contract")
assert_true(identical(attr(res$vcov_cr2,"type"),"CR2"),"vcov_object_is_cr2")

slope <- "X_per_1_pseudo_AP_point"
direct_greater <- clubSandwich::coef_test(res$model,vcov=res$vcov_cr2,test="Satterthwaite",alternative="greater")
assert_true(slope %in% rownames(direct_greater),"direct_slope_row_present")
assert_close(res$effect$estimate,as.numeric(direct_greater[slope,"beta"]),1e-14,"reported_beta_matches_direct_clubsandwich")
assert_close(res$effect$standard_error_cr2,as.numeric(direct_greater[slope,"SE"]),1e-14,"reported_se_matches_direct_clubsandwich")
assert_close(res$effect$df_satterthwaite,as.numeric(direct_greater[slope,"df_Satt"]),1e-12,"reported_df_matches_direct_clubsandwich")
assert_close(res$effect$p_one_sided_greater,as.numeric(direct_greater[slope,"p_Satt"]),1e-15,"reported_one_sided_p_matches_direct_clubsandwich")

direct_ci <- clubSandwich::conf_int(res$model,vcov=res$vcov_cr2,level=0.95,test="Satterthwaite")
assert_true(slope %in% rownames(direct_ci),"direct_ci_slope_row_present")
assert_close(res$effect$ci_lower,as.numeric(direct_ci[slope,"CI_L"]),1e-14,"reported_ci_lower_matches_direct_clubsandwich")
assert_close(res$effect$ci_upper,as.numeric(direct_ci[slope,"CI_U"]),1e-14,"reported_ci_upper_matches_direct_clubsandwich")

raw_fit <- stats::lm(Y ~ Q_WC + architecture_budget_cell,data=res$analysis_dataset)
raw_v <- clubSandwich::vcovCR(raw_fit,cluster=res$analysis_dataset$training_seed,type="CR2")
raw_beta <- as.numeric(stats::coef(raw_fit)["Q_WC"])
raw_se <- sqrt(as.numeric(raw_v["Q_WC","Q_WC"]))
assert_close(res$effect$estimate,raw_beta*0.01,1e-12,"raw_qwc_to_per_ap_beta_scale_equivalence")
assert_close(res$effect$standard_error_cr2,raw_se*0.01,1e-12,"raw_qwc_to_per_ap_se_scale_equivalence")

bad <- fixture[-1,,drop=FALSE]
bad_failed <- inherits(try(run_rq4_cr2_sensitivity(bad),silent=TRUE),"try-error")
assert_true(bad_failed,"invalid_79_row_fixture_rejected")

cat(sprintf("S6_14_CR2_FIXTURE=%d/%d_PASS\n",checks,checks))
cat("S6_14_RQ4_CR2_VALIDATOR=PASS\n")
cat("TEST_USED=False\n")
cat("HIDDEN_U_GT_USED=False\n")
cat("PRIMARY_KR_REPLACED=False\n")
