from inspect import cleandoc
from pathlib import Path
import os


def generate_experiment_scripts(
        models_nickname,
        seeds_models=None,
        models_params=None,
        fits_params=None,
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
        file_dir=Path.cwd() / 'experiment_scripts',
        experiment_file=Path(__file__).parent.parent / 'tab_benchmark' / 'benchmark' / 'base_experiment.py',
        experiment_name='tab_benchmark_experiment',
        log_dir=Path(__file__).parent.parent / 'results' / 'logs',
        work_dir=Path('/tmp'),
        save_dir=Path(__file__).parent.parent / 'results' / 'outputs',
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
    if isinstance(file_dir, str):
        file_dir = Path(file_dir)
    os.makedirs(file_dir, exist_ok=True)

    # create base content for shell script
    base_sh_content = cleandoc(f"""
    eval "$(conda shell.bash hook)"
    conda activate {conda_env}
    python {experiment_file}""")
    base_sh_content += (f" --experiment_name {experiment_name} --log_dir {log_dir} --work_dir {work_dir} "
                        f"--save_dir {save_dir} --mlflow_tracking_uri {mlflow_tracking_uri} --n_jobs {n_jobs}")
    if models_params is not None:
        base_sh_content += f" --models_params {models_params}"
    if fits_params is not None:
        base_sh_content += f" --fits_params {fits_params}"
    if dask_cluster_type is not None:
        base_sh_content += (f" --dask_cluster_type {dask_cluster_type} --n_workers {n_workers} "
                            f"--n_processes {n_processes} --n_cores {n_cores} --n_gpus {n_gpus}")
        if dask_memory is not None:
            base_sh_content += f" --dask_memory {dask_memory}"
        if dask_job_extra_directives is not None:
            base_sh_content += f" --dask_job_extra_directives {dask_job_extra_directives}"
    if dask_address is not None:
        base_sh_content += f" --dask_address {dask_address}"
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
                            file_path = file_dir / f"{combination_name}{file_ext}"
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
                                file_path = file_dir / f"{combination_name}{file_ext}"
                                file_paths.append(file_path)
                                with open(file_path, 'w') as file:
                                    file.write(file_content)
    return file_paths
