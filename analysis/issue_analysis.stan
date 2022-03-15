
data {
  int<lower=1> n_items; //number of nootropics
  int<lower=1> n_users;
  int<lower=1> n_issues;
  int<lower=1> n;
  int<lower=1, upper=n_users> users[n];
  int<lower=1, upper=n_items> items[n];
  int<lower=1, upper=n_issues> issues[n];
  int<lower=1, upper=2> ui[n];
  int<lower=0, upper=2> report[n]; // 0 for none or other issue checked, 1 for empty, 2 for checked
}

parameters {
  vector<lower=0, upper=1>[n_users] proba_none_user;
  vector<lower=0, upper=1>[n_users] proba_empty_user_true_issue;
  //vector<lower=0, upper=1>[n_issues] proba_empty_issue_true_issue;
  real logit_proba_empty_ui_true_issue_raw;
  real logit_proba_none_ui_raw;
  
  matrix[n_items, n_issues] logit_item_issues_proba_true_issue; // This is what we want, i.e Proba of issue for an item
  vector[n_issues] mean_logit_item_issues_proba_true_issue;
  vector<lower=0>[n_issues] sd_logit_item_issues_proba_true_issue;
}

transformed parameters {
  //vector[n_users] logit_proba_empty_user_true_issue;
  //vector[n_issues] logit_proba_empty_issue_true_issue;
  vector[2] logit_proba_empty_ui_true_issue;
  vector[2] logit_proba_none_ui;
  matrix[n_items, n_issues] item_issues_proba_true_issue;
  //matrix[n_items, n_issues] logit_item_issues_proba_true_issue; // This is what we want, i.e Proba of issue for an item

  
  //for (i in 1:n_issues) {
    //col(logit_item_issues_proba_true_issue, i) ~ normal(mean_logit_item_issues_proba_true_issue[i], sd_logit_item_issues_proba_true_issue[i]);
    //logit_item_issues_proba_true_issue[ ,i] = mean_logit_item_issues_proba_true_issue[i] + col(logit_item_issues_proba_true_issue_raw, i) * sd_logit_item_issues_proba_true_issue[i];
  //}

  //logit_proba_empty_user_true_issue = logit(proba_empty_user_true_issue);
  logit_proba_empty_ui_true_issue[1] = logit_proba_empty_ui_true_issue_raw;
  logit_proba_empty_ui_true_issue[2] = 0;
  logit_proba_none_ui[1] = logit_proba_none_ui_raw;
  logit_proba_none_ui[2] = 0;
  item_issues_proba_true_issue = inv_logit(logit_item_issues_proba_true_issue);
  

  
}

model {
  // Proba of entering none (or another issue) given no issue
  vector[n] proba_none_no_issue = inv_logit(logit(proba_none_user[users]) + logit_proba_none_ui[ui]);
  // Proba of not entering anything given issue
  vector[n] proba_empty_true_issue = inv_logit(logit(proba_empty_user_true_issue[users]) + logit_proba_empty_ui_true_issue[ui]);// + 
                                                  //logit_proba_empty_issue_true_issue[issues];
                                                  //logit_proba_empty_ui_true_issue[ui];
  // Proba of not entering anything given no issue
  vector[n] proba_empty_no_issue = 1 - proba_none_no_issue;
  // Proba of checking this issue given isse
  vector[n] proba_checked_true_issue = 1 - proba_empty_true_issue;
  
  //proba_none_user ~ beta(0.3, 0.6);
  proba_empty_user_true_issue ~ beta(0.9, 0.7);
  logit_proba_empty_ui_true_issue_raw ~ normal(0, 5);
  logit_proba_none_ui_raw ~ normal(0, 5);
  
  mean_logit_item_issues_proba_true_issue ~ normal(0, 5);
  sd_logit_item_issues_proba_true_issue ~ normal(0, 5);
  for (i in 1:n_issues) {
    //logit_item_issues_proba_true_issue_raw[, i] ~ normal(0, 1); //non-centered
    logit_item_issues_proba_true_issue[, i] ~ normal(mean_logit_item_issues_proba_true_issue[i], sd_logit_item_issues_proba_true_issue[i]); //centered
  }
  
  
  for (i in 1:n) {
      if (report[i] == 0) { // none or other issue checked
           target += log1m(item_issues_proba_true_issue[items[i], issues[i]]) + log(proba_none_no_issue[i]);
      }
      if (report[i] == 1){  //empty
        target += log_sum_exp(log(item_issues_proba_true_issue[items[i], issues[i]]) + log(proba_empty_true_issue[i]),
                              log1m(item_issues_proba_true_issue[items[i], issues[i]]) + log(proba_empty_no_issue[i]));
        }
      if (report[i] == 2){ //checked
        target += log(item_issues_proba_true_issue[items[i], issues[i]]) + log(proba_checked_true_issue[i]);
      }
    }
}

