import os
from inspect import cleandoc
from generate_db_script import generate_postgres_db_script
from tab_benchmark.datasets import datasets_characteristics_path
import pandas as pd
from pathlib import Path
from shutil import rmtree


def generate_experiment_scripts(
        models_nickname,
        seeds_models=None,
        models_params=None,
        fits_params=None,
        create_validation_set=False,
        # either tasks_ids or datasets_names_or_ids must be provided
        tasks_ids=None,
        datasets_names_or_ids=None,
        # task parameters
        task_repeats=None,
        task_folds=None,
        task_samples=None,
        # datasets parameters
        seeds_datasets=None,
        folds=None,
        resample_strategy='k-fold_cv',
        k_folds=10,
        pct_test=0.2,
        validation_resample_strategy='next_fold',
        pct_validation=0.1,
        n_jobs=1,
        scripts_dir=Path.cwd() / 'experiment_scripts',
        python_file_dir=Path(__file__).parent.parent / 'tab_benchmark' / 'benchmark',
        python_file_name='base_experiment.py',
        experiment_name='tab_benchmark_experiment',
        log_dir=Path(__file__).parent.parent / 'results' / 'logs',
        work_root_dir=Path('/tmp'),
        save_root_dir=Path(__file__).parent.parent / 'results' / 'outputs',
        mlflow_tracking_uri='sqlite:///' + str(
            (Path(__file__).parent.parent / 'results').resolve()) + '/tab_benchmark.db',
        dask_cluster_type=None,
        n_workers=1,
        n_processes=1,
        n_cores=1,
        n_gpus=0,
        dask_memory=None,
        dask_job_extra_directives=None,
        dask_address=None,
        conda_env='tab_benchmark',
        # sbatch parameters
        generate_sbatch=True,
        sbatch_c=None,
        sbatch_G=None,
        sbatch_gres_mps=None,
        sbatch_w=None,
        sbatch_exclude=None,
        sbatch_output=str(Path(__file__).parent.parent / 'results' / 'sbatch_outputs' / '%x.%J.out'),
        sbatch_error=str(Path(__file__).parent.parent / 'results' / 'sbatch_errors' / '%x.%J.out'),
        sbatch_time='364-23:59:59',
        **kwargs
):
    """Generate scripts for each individual combination of models, seeds, datasets, and folds."""
    # defaults for lists when not provided
    if seeds_models is None:
        seeds_models = [0]
    if folds is None:
        folds = [0]
    if seeds_datasets is None:
        seeds_datasets = [0]
    if task_repeats is None:
        task_repeats = [0]
    if task_folds is None:
        task_folds = [0]
    if task_samples is None:
        task_samples = [0]

    # create directory if it does not exist
    if isinstance(scripts_dir, str):
        file_dir = Path(scripts_dir)
    os.makedirs(scripts_dir, exist_ok=True)

    # create base content for shell script
    base_sh_content = cleandoc(f"""
    eval "$(conda shell.bash hook)"
    conda activate {conda_env}
    cd {python_file_dir}
    python {python_file_name}""")
    base_sh_content += (f" --experiment_name {experiment_name} --log_dir {log_dir} --work_root_dir {work_root_dir} "
                        f"--save_root_dir {save_root_dir} --mlflow_tracking_uri {mlflow_tracking_uri} "
                        f"--n_jobs {n_jobs}")
    if models_params is not None:
        base_sh_content += f" --models_params '{models_params}'"
    if fits_params is not None:
        base_sh_content += f" --fits_params '{fits_params}'"
    if dask_cluster_type is not None:
        base_sh_content += (f" --dask_cluster_type {dask_cluster_type} --n_workers {n_workers} "
                            f"--n_processes {n_processes} --n_cores {n_cores} --n_gpus {n_gpus}")
        if dask_memory is not None:
            base_sh_content += f" --dask_memory {dask_memory}"
        if dask_job_extra_directives is not None:
            base_sh_content += f" --dask_job_extra_directives {dask_job_extra_directives}"
    if dask_address is not None:
        base_sh_content += f" --dask_address {dask_address}"
    if create_validation_set:
        base_sh_content += ' --create_validation_set'
    if datasets_names_or_ids is not None and tasks_ids is None:
        base_sh_content += (f" --resample_strategy {resample_strategy} --k_folds {k_folds} "
                            f"--pct_test {pct_test} --validation_resample_strategy {validation_resample_strategy} "
                            f"--pct_validation {pct_validation}")
        using_own_resampling = True
    elif datasets_names_or_ids is None and tasks_ids is not None:
        using_own_resampling = False
    else:
        raise ValueError("You must provide either datasets_names_or_ids or tasks_ids, but not both.")
    if kwargs:
        base_sh_content += ' ' + ' '.join([f"--{key} {value}" for key, value in kwargs.items()])

    # append sbatch parameters if generating sbatch scripts
    if generate_sbatch:
        sbatch_content = "#!/bin/sh\n"
        if sbatch_c is not None:
            sbatch_content += f"#SBATCH -c {sbatch_c}\n"
        if sbatch_G is not None:
            sbatch_content += f"#SBATCH -G {sbatch_G}\n"
        if sbatch_gres_mps is not None:
            sbatch_content += f"#SBATCH --gres=mps:{sbatch_gres_mps}\n"
        if sbatch_w is not None:
            sbatch_content += f"#SBATCH -w {sbatch_w}\n"
        if sbatch_exclude is not None:
            sbatch_content += f"#SBATCH --exclude={sbatch_exclude}\n"
        sbatch_content += f"#SBATCH --output={sbatch_output}\n"
        sbatch_content += f"#SBATCH --error={sbatch_error}\n"
        sbatch_content += f"#SBATCH --time={sbatch_time}\n"
        base_file = sbatch_content + base_sh_content
        file_ext = '.sbatch'
    else:
        file_ext = '.sh'
        base_file = base_sh_content

    # generate scripts for each combination of models, seeds, datasets, and folds
    file_paths = []
    for model_nickname in models_nickname:
        for seed_model in seeds_models:
            if using_own_resampling:
                for dataset_name_or_id in datasets_names_or_ids:
                    for seed_dataset in seeds_datasets:
                        for fold in folds:
                            file_content = base_file
                            combination_name = (f"{model_nickname}_{seed_model}_{dataset_name_or_id}_"
                                                f"{seed_dataset}_{fold}")
                            if generate_sbatch:
                                # insert in the second line the combination name as job name
                                file_content = file_content.split('\n')
                                file_content.insert(1, f"#SBATCH --job-name={combination_name}")
                                file_content = '\n'.join(file_content)
                            file_content += (f" --models_nickname {model_nickname} --seeds_models {seed_model} "
                                             f"--datasets_names_or_ids {dataset_name_or_id} "
                                             f"--seeds_datasets {seed_dataset} --folds {fold} "
                                             f"--log_file_name {combination_name}")
                            file_path = scripts_dir / f"{combination_name}{file_ext}"
                            file_paths.append(file_path)
                            with open(file_path, 'w') as file:
                                file.write(file_content)
            else:
                for task_id in tasks_ids:
                    for task_repeat in task_repeats:
                        for task_fold in task_folds:
                            for task_sample in task_samples:
                                file_content = base_file
                                combination_name = (f"{model_nickname}_{seed_model}_{task_id}_"
                                                    f"{task_repeat}_{task_sample}_{task_fold}")
                                if generate_sbatch:
                                    # insert in the second line the combination name as job name
                                    file_content = file_content.split('\n')
                                    file_content.insert(1, f"#SBATCH --job-name={combination_name}")
                                    file_content = '\n'.join(file_content)
                                file_content += (f" --models_nickname {model_nickname} --seeds_models {seed_model} "
                                                 f"--tasks_ids {task_id} --task_repeats {task_repeat} "
                                                 f"--task_folds {task_fold} --task_samples {task_sample} "
                                                 f"--log_file_name {combination_name}")
                                file_path = scripts_dir / f"{combination_name}{file_ext}"
                                file_paths.append(file_path)
                                with open(file_path, 'w') as file:
                                    file.write(file_content)
    return file_paths


file_dir = Path() / 'scripts'
rmtree(file_dir, ignore_errors=True)
# create the database
file_name = 'start_db_adacap'
conda_env = 'tab_benchmark'
database_root_dir = '/home/users/belucci/adacap/results'
db_name = 'adacap'
db_port = 5001
mlflow_port = 5002
generate_sbatch = True
n_cores = 6
clust_name = 'clust10'
job_name = 'adacap_db'
output_job_file = '/home/users/belucci/adacap/results/sbatch_outputs/%x.%J.out'
error_job_file = '/home/users/belucci/adacap/results/sbatch_errors/%x.%J.err'
wall_time = '364-23:59:59'
db_file = generate_postgres_db_script(file_dir=file_dir, file_name=file_name, conda_env=conda_env,
                                      database_root_dir=database_root_dir, db_name=db_name, db_port=db_port,
                                      mlflow_port=mlflow_port, generate_sbatch=generate_sbatch, n_cores=n_cores,
                                      clust_name=clust_name, job_name=job_name, output_job_file=output_job_file,
                                      error_job_file=error_job_file, wall_time=wall_time)

# get datasets characteristics
datasets_characteristics = pd.read_csv(datasets_characteristics_path)

# create the experiment scripts
models_nickname = ['TabBenchmarkMLP_Adacap']
models_params = '{"n_jobs":0}'  #,"auto_reduce_batch_size":1}'
seeds_models = [0]  # 361097 after
tasks_ids = [361293, 361292, 361099, 361097, 362091, 359942, 361291, 361257, 362089, 4774, 362110, 362117, 362093,
             362094]
task_folds = [i for i in range(10)]
n_jobs = 1
create_validation_set = False
# scripts_dir = Path() / 'scripts'
python_file_dir = '/home/users/belucci/adacap'
python_file_name = '-m adacap.experiments.adacap_experiment'
experiment_name = 'permutation'
log_dir = '/home/users/belucci/adacap/results/logs'
work_root_dir = '/tmp'
save_root_dir = '/home/users/belucci/adacap/results/outputs'
mlflow_tracking_uri = f'http://{clust_name}.ceremade.dauphine.lan:{mlflow_port}/'
generate_sbatch = True
sbatch_c = 2
sbatch_G = 1
sbatch_gres_mps = None
sbatch_w = None
sbatch_output = '/home/users/belucci/adacap/results/sbatch_outputs/%x.%J.out'
sbatch_error = '/home/users/belucci/adacap/results/sbatch_errors/%x.%J.err'
sbatch_time = '364-23:59:59'
sbatch_exclude = 'clust1,clust2,clust11,clust12'
n_gpus = 1
kwargs = {'n_y_permuted': '1 2 5 10 20'}
generate_experiment_scripts(models_nickname=models_nickname, seeds_models=seeds_models, tasks_ids=tasks_ids,
                            task_folds=task_folds, n_jobs=n_jobs, scripts_dir=file_dir, python_file_dir=python_file_dir,
                            python_file_name=python_file_name,
                            experiment_name=experiment_name,
                            log_dir=log_dir, work_root_dir=work_root_dir, save_root_dir=save_root_dir,
                            mlflow_tracking_uri=mlflow_tracking_uri, generate_sbatch=generate_sbatch, sbatch_c=sbatch_c,
                            sbatch_w=sbatch_w, sbatch_output=sbatch_output, sbatch_error=sbatch_error,
                            sbatch_G=sbatch_G, sbatch_gres_mps=sbatch_gres_mps,
                            sbatch_time=sbatch_time, n_gpus=n_gpus, models_params=models_params,
                            sbatch_exclude=sbatch_exclude, create_validation_set=create_validation_set, **kwargs)
