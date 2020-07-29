{
if (match($9, "block_start")) {sum_bstart++; str=", \"first_epoch_num\":}}"; gsub("}}", str, $0); gsub("}}", sum_bstart*4-4"}}", $0); gsub("}}", ", \"epoch_count\": 4}}", $0); print $0}
else if (match($9, "block_stop")) {sum_bstop++; str=", \"first_epoch_num\":}}"; gsub("}}", str, $0); gsub("}}", sum_bstop*4-4"}}", $0); gsub("}}", ", \"epoch_count\": 4}}", $0); print $0}
else if (match($9, "eval_start")) {sum_estart++; str=", \"epoch_num\":}}"; gsub("}}", str, $0); gsub("}}", sum_estart*4-1"}}", $0); print $0}
else if (match($9, "eval_accuracy")) {sum_eacc++; str=", \"epoch_num\":}}"; gsub("}}", str, $0); gsub("}}", sum_eacc*4-1"}}", $0); print $0}
else if (match($9, "eval_stop")) {sum_estop++; str=", \"epoch_num\":}}"; gsub("}}", str, $0); gsub("}}", sum_estop*4-1"}}", $0); print $0}
else if (match($9, "run_stop")) {str=", \"status\": \"success\"}}"; gsub("}}", str, $0); print $0}
else if (match($9, "eval_samples") && count_esample==0) {count_esample++; print $0}
else if (match($9, "eval_samples") && count_esample>0) {count_esample++; gsub("eval_samples", "duplicate_eval_samples", $0); print $0}
else if (match($9, "opt_name") && count_on==0) {count_on++; print $0}
else if (match($9, "opt_name") && count_on>0) {count_on++; gsub("opt_name", "duplicate_opt_name", $0); print $0}
else if (match($9, "lars_opt_base_learning_rate") && count_loblr==0) {count_loblr++; print $0}
else if (match($9, "lars_opt_base_learning_rate") && count_loblr>0) {count_loblr++; gsub("lars_opt_base_learning_rate", "duplicate_lars_opt_base_learning_rate", $0); print $0}
else if (match($9, "lars_opt_end_learning_rate") && count_loelr==0) {count_loelr++; print $0}
else if (match($9, "lars_opt_end_learning_rate") && count_loelr>0) {count_loelr++; gsub("lars_opt_end_learning_rate", "duplicate_lars_opt_end_learning_rate", $0); print $0}
else if (match($9, "lars_opt_learning_rate_decay_poly_power") && count_lolrdpp==0) {count_lolrdpp++; print $0}
else if (match($9, "lars_opt_learning_rate_decay_poly_power") && count_lolrdpp>0) {count_loelrdpp++; gsub("lars_opt_learning_rate_decay_poly_power", "duplicate_lars_opt_learning_rate_decay_poly_power", $0); print $0}
else if (match($9, "lars_opt_learning_rate_decay_poly_power") && count_lolrdpp==0) {count_lolrdpp++; print $0}
else if (match($9, "lars_opt_learning_rate_decay_poly_power") && count_lolrdpp>0) {count_loelrdpp++; gsub("lars_opt_learning_rate_decay_poly_power", "duplicate_lars_opt_learning_rate_decay_poly_power", $0); print $0}
else if (match($9, "lars_opt_learning_rate_decay_steps") && count_lolrds==0) {count_lolrds++; print $0}
else if (match($9, "lars_opt_learning_rate_decay_steps") && count_lolrds>0) {count_loelrds++; gsub("lars_opt_learning_rate_decay_steps", "duplicate_lars_opt_learning_rate_decay_steps", $0); print $0}
else if (match($9, "lars_epsilon") && count_le==0) {count_le++; print $0}
else if (match($9, "lars_epsilon") && count_le>0) {count_le++; gsub("lars_epsilon", "duplicate_lars_epsilon", $0); print $0}
else if (match($9, "lars_opt_learning_rate_warmup_epochs") && count_lolrwe==0) {count_lolrwe++; print $0}
else if (match($9, "lars_opt_learning_rate_warmup_epochs") && count_lolrwe>0) {count_lolrwe++; gsub("lars_opt_learning_rate_warmup_epochs", "duplicate_lars_opt_learning_rate_warmup_epochs", $0); print $0}
else if (match($9, "lars_opt_weight_decay") && count_lowd==0) {count_lowd++; print $0}
else if (match($9, "lars_opt_weight_decay") && count_lowd>0) {count_lowd++; gsub("lars_opt_weight_decay", "duplicate_lars_opt_weight_decay", $0); print $0}
else {print $0}
}
