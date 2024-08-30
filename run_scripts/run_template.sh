base_experiment_location="/home/users/belucci/tab_benchmark/tab_benchmark/benchmark/base_experiment.py"
experiment_name="xgboost_test"
log_dir="/home/users/belucci/tab_benchmark/results/logs"
dask_cluster_type="slurm"
slurm_config_name="slurm-single-core-process-cpu"
models_nickname="TabBenchmarkXGBClassifier"
n_jobs=1
mlflow_tracking_uri="http://clust1.ceremade.dauphine.lan:5002/"
task_folds="0 1"
tasks_ids="7592 31 24"
n_workers=3
conda activate tab_benchmark
python $base_experiment_location --experiment_name $experiment_name --log_dir $log_dir \
 --dask_cluster_type $dask_cluster_type --slurm_config_name $slurm_config_name --models_nickname $models_nickname \
--n_jobs $n_jobs --mlflow_tracking_uri $mlflow_tracking_uri --task_folds $task_folds --tasks_ids $tasks_ids \
--n_workers $n_workers