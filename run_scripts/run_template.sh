base_experiment_location="/home/users/belucci/tab_benchmark/tab_benchmark/benchmark/base_experiment.py"
experiment_name="xgboost_test"
output_dir="/home/users/belucci/tab_benchmark/results"
log_dir="/home/users/belucci/tab_benchmark/results/logs"
models_nickname="TabBenchmarkXGBClassifier"
n_jobs=10
dask_cluster_type="slurm"
n_workers=5
n_processes=1
n_cores=10
n_gpus=0
mlflow_tracking_uri="http://clust1.ceremade.dauphine.lan:5002/"
task_folds="0 1"
tasks_ids="7592 31 24"
dask_memory="10Gb"
dask_job_extra_directives="--exclude clust12"
eval "$(conda shell.bash hook)"
conda activate tab_benchmark
python $base_experiment_location --experiment_name $experiment_name --log_dir $log_dir --output_dir $output_dir \
--models_nickname $models_nickname --n_jobs $n_jobs \
--mlflow_tracking_uri $mlflow_tracking_uri --task_folds $task_folds --tasks_ids $tasks_ids \
--dask_cluster_type $dask_cluster_type --n_processes $n_processes --n_cores $n_cores --n_workers $n_workers \
--dask_memory $dask_memory --dask_job_extra_directives "$dask_job_extra_directives" --n_gpus $n_gpus