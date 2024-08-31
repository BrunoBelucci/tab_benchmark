base_experiment_location="/home/users/belucci/tab_benchmark/tab_benchmark/benchmark/base_experiment.py"
experiment_name="xgboost_test"
output_dir="/home/users/belucci/tab_benchmark/results"
log_dir="/home/users/belucci/tab_benchmark/results/logs"
dask_cluster_type="local"
n_workers=3
models_nickname="TabBenchmarkXGBClassifier"
n_jobs=1
mlflow_tracking_uri="http://clust1.ceremade.dauphine.lan:5002/"
task_folds="0 1"
tasks_ids="7592 31 24"
eval "$(conda shell.bash hook)"
conda activate tab_benchmark
python $base_experiment_location --experiment_name $experiment_name --log_dir $log_dir \
 --dask_cluster_type $dask_cluster_type --models_nickname $models_nickname \
--n_jobs $n_jobs --mlflow_tracking_uri $mlflow_tracking_uri --task_folds $task_folds --tasks_ids $tasks_ids \
--n_workers $n_workers --output_dir $output_dir
