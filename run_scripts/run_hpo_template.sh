#!/bin/bash
environment_name="tab_benchmark"
experiment_python_location="path/to/tab_benchmark/benchmark/tabular_experiment.py"

# Create a dictionary with argument names and values
declare -A args_dict=(
# base
["experiment_name"]=""
["models_nickname"]=""
["seeds_models"]=""
["n_jobs"]=""
["models_params"]=""
["fits_params"]=""
["error_score"]=""
["timeout_fit"]=""
["timeout_combination"]=""
["log_dir"]=""
["log_file_name"]=""
["work_root_dir"]=""
["save_root_dir"]=""
["mlflow_tracking_uri"]=""
["dask_cluster_type"]=""
["n_workers"]=""
["n_cores_per_worker"]=""
["n_processes_per_worker"]=""
["n_threads_per_worker"]=""
["n_cores_per_task"]=""
["n_processes_per_task"]=""
["n_threads_per_task"]=""
["dask_memory"]=""
["dask_job_extra_directives"]=""
["dask_address"]=""
["n_gpus_per_worker"]=""
["n_gpus_per_task"]=""
# hpo
["hpo_framework"]=""
["n_trials"]=""
["timeout_hpo"]=""
["timeout_trial"]=""
["max_concurrent_trials"]=""
["sampler"]=""
["pruner"]=""
["direction"]=""
["hpo_metric"]=""
# tabular_experiment
["datasets_names_or_ids"]=""
["seeds_datasets"]=""
["resample_strategy"]=""
["k_folds"]=""
["folds"]=""
["pct_test"]=""
["validation_resample_strategy"]=""
["pct_validation"]=""
["tasks_ids"]=""
["task_repeats"]=""
["task_samples"]=""
["task_folds"]=""
["max_time"]=""
)

declare -A bool_args_dict=(
# base
["create_validation_set"]=0
["do_not_clean_work_dir"]=0
["do_not_log_to_mlflow"]=0
["do_not_check_if_exists"]=0
["do_not_retry_on_oom"]=0
["raise_on_fit_error"]=0
)

# Construct the argument string
args_str=""
for key in "${!args_dict[@]}"; do
  if [ -n "${args_dict[$key]}" ]; then
    args_str="$args_str --$key ${args_dict[$key]}"
  fi
done

for key in "${!bool_args_dict[@]}"; do
  if [ "${bool_args_dict[$key]}" -eq 1 ]; then
    args_str="$args_str --$key"
  fi
done

# Activate the conda environment and run the experiment
eval "$(conda shell.bash hook)"
conda activate $environment_name
python $experiment_python_location $args_str
