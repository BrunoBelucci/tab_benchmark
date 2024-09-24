import argparse
import os
import time
from pathlib import Path
from random import SystemRandom
import dask
import ray
from distributed import get_worker
from ray import tune
from ray.tune import Tuner, randint, Callback
import mlflow
from tab_benchmark.benchmark.base_experiment import BaseExperiment, log_and_print_msg
from tab_benchmark.benchmark.utils import treat_mlflow, get_search_algorithm_tune_config_run_config
from tab_benchmark.models.dnn_model import DNNModel
from tab_benchmark.models.dnn_models import max_epochs_dnn
from tab_benchmark.models.xgboost import n_estimators_gbdt
from tab_benchmark.utils import get_git_revision_hash, flatten_dict, extends


@ray.remote
class LastMetricsActor:
    def __init__(self):
        self.last_reported_metrics = {}

    def add_metrics(self, trial_id, result):
        self.last_reported_metrics[trial_id] = result

    def get_metrics(self, trial_id):
        return self.last_reported_metrics.get(trial_id, [])


class LastMetricsActorCallback(Callback):
    def __init__(self, metrics_actor):
        self.metrics_actor = metrics_actor

    def on_trial_result(self, iteration, trials, trial, result, **info):
        trial_id = trial.trial_id
        # Store the metrics in the Ray Actor
        self.metrics_actor.add_metrics.remote(trial_id, result)


class HPOExperiment(BaseExperiment):
    @extends(BaseExperiment.__init__)
    def __init__(self, *args, search_algorithm='random_search', trial_scheduler=None, n_trials=30,
                 timeout_experiment=10 * 60 * 60,
                 timeout_trial=2 * 60 * 60, retrain_best_model=False, max_concurrent_trials=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_algorithm = search_algorithm
        self.trial_scheduler = trial_scheduler
        self.n_trials = n_trials
        self.timeout_experiment = timeout_experiment  # 10 hours
        self.timeout_trial = timeout_trial  # 2 hours
        self.retrain_best_model = retrain_best_model
        self.max_concurrent_trials = max_concurrent_trials
        self.log_dir_dask = None

    def add_arguments_to_parser(self):
        super().add_arguments_to_parser()
        self.parser.add_argument('--search_algorithm', type=str, default=self.search_algorithm)
        self.parser.add_argument('--trial_scheduler', type=str, default=self.trial_scheduler)
        self.parser.add_argument('--n_trials', type=int, default=self.n_trials)
        self.parser.add_argument('--timeout_experiment', type=int, default=self.timeout_experiment)
        self.parser.add_argument('--timeout_trial', type=int, default=self.timeout_trial)
        self.parser.add_argument('--retrain_best_model', action='store_true')
        self.parser.add_argument('--max_concurrent_trials', type=int, default=self.max_concurrent_trials)

    def unpack_parser(self):
        args = super().unpack_parser()
        self.search_algorithm = args.search_algorithm
        self.trial_scheduler = args.trial_scheduler
        self.n_trials = args.n_trials
        self.timeout_experiment = args.timeout_experiment
        self.timeout_trial = args.timeout_trial
        self.retrain_best_model = args.retrain_best_model
        self.max_concurrent_trials = args.max_concurrent_trials

    def get_model(self, model_params=None, n_jobs=1,
                  logging_to_mlflow=False, create_validation_set=False, output_dir=None, data_return=None, **kwargs):
        model_nickname = kwargs.get('model_nickname')
        seed_model = kwargs.get('seed_model')
        if data_return:
            data_params = data_return.get('data_params', None).copy()
        else:
            data_params = None
        if output_dir is None:
            # if logging to mlflow we use the mlflow artifact directory
            if logging_to_mlflow:
                # this is already unique
                output_dir = Path(mlflow.get_artifact_uri())
                unique_name = output_dir.parts[-2]
            else:
                # if we only save the model in the output_dir we will have some problems when running in parallel
                # because the workers will try to save the model in the same directory, so we must create a unique
                # directory for each combination model/dataset
                if data_params is not None:
                    unique_name = '_'.join([f'{k}={v}' for k, v in data_params.items()])
                else:
                    # not sure, but I think it will generate a true random number and even be thread safe
                    unique_name = f'{SystemRandom().randint(0, 1000000):06d}'
                output_dir = self.output_dir / f'{model_nickname}_{unique_name}'
            # if running on a worker, we use the worker's local directory
            try:
                worker = get_worker()
                output_dir = Path(worker.local_directory) / unique_name
            except ValueError:
                # if running on the main process, we use the output_dir
                output_dir = output_dir

            if ray.train._internal.session._get_session():
                # When doing HPO with ray we want the output_dir to be configured relative to the ray storage
                output_dir = Path.cwd() / unique_name
            os.makedirs(output_dir, exist_ok=True)
            # Else we use the default output_dir (artifact location if mlflow or self.output_dir)
        return super().get_model(model_params, n_jobs, logging_to_mlflow,
                                 create_validation_set, output_dir, data_return, **kwargs)

    def get_training_fn_for_hpo(self, is_openml=True, has_early_stopping=False):
        def training_fn(config, last_metrics_actor=None):
            # setup logger on ray worker
            config = config.copy()
            self.setup_logger(log_dir=self.log_dir_dask, filemode='a')
            parent_run_uuid = config.pop('parent_run_uuid', None)
            if is_openml:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('task_id'),
                        config.pop('task_fold'), config.pop('task_repeat'), config.pop('task_sample'))
            else:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('dataset_name_or_id'),
                        config.pop('seed_dataset'), config.pop('fold'))
            results = super(HPOExperiment, self).run_combination_with_mlflow(
                *args,
                create_validation_set=True, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)
            metrics_results = {metric: value for metric, value in results['evaluate_return'].items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
            if parent_run_uuid:
                mlflow.log_metrics(metrics_results, step=int(time.time_ns()), run_id=parent_run_uuid)
            metrics_results['was_evaluated'] = True
            if has_early_stopping:
                # We will repeat the last reported metrics, because the metric must be present for ray to work
                # this is not clean, but it is the only way I found to make it work
                train_context = ray.train.get_context()
                trial_id = train_context.get_trial_id()
                last_reported_metrics = ray.get(last_metrics_actor.get_metrics.remote(trial_id))
                last_reported_metrics_results = {metric: value for metric, value in last_reported_metrics.items()
                                                 if metric.startswith('validation_') or metric.startswith('test_')}
                metrics_results.update(last_reported_metrics_results)
            return metrics_results

        return training_fn

    def run_combination_hpo(self,
                            n_jobs=1, create_validation_set=True,
                            model_params=None, is_openml=True, logging_to_mlflow=False,
                            fit_params=None, return_results=False, retrain_best_model=False, **kwargs):
        num_cpus = n_jobs
        num_gpus = self.n_gpus / (self.n_cores / n_jobs)
        ray.init(address='local', num_cpus=num_cpus, num_gpus=num_gpus, ignore_reinit_error=True)
        model_nickname = kwargs.pop('model_nickname')
        seed_model = kwargs.pop('seed_model')
        models_dict = kwargs.pop('models_dict', self.models_dict)
        parent_run_uuid = kwargs.pop('parent_run_uuid', None)
        search_algorithm_str = kwargs.pop('search_algorithm', self.search_algorithm)
        trial_scheduler_str = kwargs.pop('trial_scheduler', self.trial_scheduler)
        n_trials = kwargs.pop('n_trials', self.n_trials)
        timeout_experiment = kwargs.pop('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.pop('timeout_trial', self.timeout_trial)
        max_concurrent_trials = kwargs.pop('max_concurrent_trials', self.max_concurrent_trials)
        if logging_to_mlflow:
            storage_path = Path(mlflow.get_artifact_uri())
        else:
            storage_path = self.output_dir
        model_cls = models_dict[model_nickname][0]
        # check if model_cls is a model that has early stopping
        if model_cls.has_early_stopping:
            # metric is reported inside the model
            metric = 'validation_default'
            trainable = self.get_training_fn_for_hpo(is_openml=is_openml, has_early_stopping=True)
            last_metrics_actor = LastMetricsActor.remote()
            callbacks = [LastMetricsActorCallback(last_metrics_actor)]
            with_parameters = {'last_metrics_actor': last_metrics_actor}
        else:
            # metric is reported by the training function
            metric = 'final_validation_default'
            trainable = self.get_training_fn_for_hpo(is_openml=is_openml, has_early_stopping=False)
            callbacks = []
            with_parameters = {}
        mode = 'min'
        search_space, default_values = model_cls.create_search_space()
        model_params = model_params if model_params is not None else {}
        search_space.update(model_params)
        fit_params = fit_params.copy() if fit_params is not None else {}
        fit_params['report_to_ray'] = True
        param_space = dict(
            seed_model=randint(0, 10000),  # seed for model, seed_model will be passed to the search algorithm
            n_jobs=n_jobs,
            model_params=search_space,
            parent_run_uuid=parent_run_uuid,
            fit_params=fit_params,
            model_nickname=model_nickname,
        )
        default_param_space = dict(
            seed_model=seed_model,
            model_params=default_values,
        )
        data_params = kwargs.copy()  # hopefully it only rest them
        param_space.update(**data_params)
        if issubclass(model_cls, DNNModel):
            max_t = max_epochs_dnn
        else:
            max_t = n_estimators_gbdt
        search_algorithm, tune_config, run_config = get_search_algorithm_tune_config_run_config(
            default_param_space, search_algorithm_str, trial_scheduler_str, max_t, n_trials, timeout_experiment,
            timeout_trial, storage_path, metric, mode, seed_model, max_concurrent_trials, callbacks)
        trainable_with_parameters = tune.with_parameters(trainable, **with_parameters)
        trainable_with_resources = tune.with_resources(trainable_with_parameters, {'cpu': num_cpus, 'gpu': num_gpus})
        tuner = Tuner(trainable_with_resources, param_space=param_space,
                      tune_config=tune_config, run_config=run_config)
        # to test the trainable function uncomment the following 3 lines
        # param_space['model_params'] = default_values
        # param_space['seed_model'] = 0
        # trainable(param_space)
        # raise
        results = tuner.fit()
        best_result = results.get_best_result()
        best_result_was_evaluated = best_result.metrics.get('was_evaluated', False)
        best_params_and_seed = best_result.metrics['config']['model_params'].copy()
        best_params_and_seed['seed_best_model'] = best_result.metrics['config']['seed_model']
        best_metric_results = {f'best_{metric}': value for metric, value in best_result.metrics.items()
                               if metric.startswith('final_validation_') or metric.startswith('final_test_')}
        if not best_result_was_evaluated:
            # if early stopping was used, the best model may not have been evaluated on the final validation and test
            # sets, because of early stopping managed by the tuner scheduler,
            # so we retrain them from scratch
            # OBS.: We could start from the last saved model but:
            # This may lead to minor differences between training the entire model from scratch
            # for example, for xgboost, when we load the best model, the random state is not preserved, so the model
            # will build trees differently than the original model would have built
            # And we need to save as many models as the number of trials, which can be a lot
            # now we retrain the best model starting from the best model file
            config = best_result.metrics['config']
            config['fit_params']['report_to_ray'] = False
            parent_run_uuid = config.pop('parent_run_uuid', None)
            if is_openml:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('task_id'),
                        config.pop('task_fold'), config.pop('task_repeat'), config.pop('task_sample'))
            else:
                args = (config.pop('model_nickname'), config.pop('seed_model'), config.pop('dataset_name_or_id'),
                        config.pop('seed_dataset'), config.pop('fold'))
            best_model_results = super(HPOExperiment, self).run_combination_with_mlflow(
                *args,
                create_validation_set=True, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)

            # change best_metric_results with new results
            best_metric_results = {f'best_{metric}': value for metric, value in best_model_results.items()
                                   if metric.startswith('final_validation_') or metric.startswith('final_test_')}

        if logging_to_mlflow:
            mlflow.log_params(best_params_and_seed)
            mlflow.log_metrics(best_metric_results)
        if retrain_best_model:
            # retrain without the validation set
            # for models that used early stopping, we will still create a validation set if auto_early_stopping is True,
            # so maybe it is not the best idea to retrain them
            # this is most useful for models that do not have early stopping and will benefit from a
            # training set with the validation set included
            config = best_result.metrics['config'].copy()
            config['fit_params']['report_to_ray'] = False
            results = super(HPOExperiment, self).run_combination_with_mlflow(
                create_validation_set=False, parent_run_uuid=parent_run_uuid, is_openml=is_openml, return_results=True,
                **config)
            metrics_results = {f'final_{metric}': value for metric, value in results.items()
                               if metric.startswith('validation_') or metric.startswith('test_')}
            if logging_to_mlflow:
                mlflow.log_metrics(metrics_results)
        if return_results:
            return results
        else:
            return True

    def run_combination_with_mlflow(self, *args,
                                    n_jobs=1, create_validation_set=False,
                                    model_params=None, logging_to_mlflow=False,
                                    fit_params=None, return_results=False, parent_run_uuid=None, **kwargs):
        # setup logger to local dir on dask worker (will not have effect if running from main)
        self.log_dir_dask = Path.cwd() / self.log_dir
        self.setup_logger(log_dir=self.log_dir_dask, filemode='w')
        experiment_name = kwargs.pop('experiment_name', self.experiment_name)
        mlflow_tracking_uri = kwargs.pop('mlflow_tracking_uri', self.mlflow_tracking_uri)
        check_if_exists = kwargs.pop('check_if_exists', self.check_if_exists)
        search_algorithm = kwargs.get('search_algorithm', self.search_algorithm)
        trial_scheduler = kwargs.get('trial_scheduler', self.trial_scheduler)
        n_trials = kwargs.get('n_trials', self.n_trials)
        timeout_experiment = kwargs.get('timeout_experiment', self.timeout_experiment)
        timeout_trial = kwargs.get('timeout_trial', self.timeout_trial)
        max_concurrent_trials = kwargs.get('max_concurrent_trials', self.max_concurrent_trials)
        retrain_best_model = kwargs.pop('retrain_best_model', self.retrain_best_model)
        unique_params = self.combination_args_to_kwargs(*args, **kwargs)
        model_nickname = unique_params.get('model_nickname')
        model_params = model_params if model_params else self.models_params.get(model_nickname, {}).copy()
        fit_params = fit_params if fit_params else self.fits_params.get(kwargs.get('model_nickname'), {}).copy()
        unique_params.update(model_params=model_params, fit_params=fit_params,
                             create_validation_set=create_validation_set,
                             search_algorithm=search_algorithm, trial_scheduler=trial_scheduler, n_trials=n_trials,
                             timeout_experiment=timeout_experiment, timeout_trial=timeout_trial,
                             max_concurrent_trials=max_concurrent_trials, retrain_best_model=retrain_best_model,
                             **kwargs)
        possible_existent_run, logging_to_mlflow = treat_mlflow(experiment_name, mlflow_tracking_uri, check_if_exists,
                                                                **unique_params)
        unique_params.pop('model_params')
        unique_params.pop('fit_params')
        unique_params.pop('create_validation_set')
        unique_params.pop('retrain_best_model')

        if possible_existent_run is not None:
            log_and_print_msg('Run already exists on MLflow. Skipping...')
            if return_results:
                return possible_existent_run.to_dict()
            else:
                return None

        if logging_to_mlflow:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name, artifact_location=str(self.output_dir))
            mlflow.set_experiment(experiment_name)
            run_name = '_'.join([f'{k}={v}' for k, v in unique_params.items()])
            if parent_run_uuid is not None:
                # check if parent run is active, if not start it
                possible_parent_run = mlflow.active_run()
                if possible_parent_run is not None:
                    if possible_parent_run.info.run_uuid != parent_run_uuid:
                        mlflow.start_run(run_id=parent_run_uuid, nested=True)
                else:
                    mlflow.start_run(parent_run_uuid)
                nested = True
            else:
                nested = False
            with mlflow.start_run(run_name=run_name, nested=nested) as run:
                mlflow.log_params(flatten_dict(unique_params))
                mlflow.log_params(flatten_dict(fit_params))
                mlflow.log_params(flatten_dict(model_params))
                model_nickname = unique_params['model_nickname']
                if model_nickname.find('TabBenchmark') != -1:
                    mlflow.log_param('model_name', model_nickname[len('TabBenchmark'):])
                mlflow.log_param('create_validation_set', create_validation_set)
                mlflow.log_param('git_hash', get_git_revision_hash())
                # slurm parameters
                mlflow.log_param('SLURM_JOB_ID', os.getenv('SLURM_JOB_ID', None))
                mlflow.log_param('SLURMD_NODENAME', os.getenv('SLURMD_NODENAME', None))
                # dask parameters
                mlflow.log_param('dask_cluster_type', self.dask_cluster_type)
                mlflow.log_param('n_workers', self.n_workers)
                mlflow.log_param('n_cores', self.n_cores)
                mlflow.log_param('n_processes', self.n_processes)
                mlflow.log_param('dask_memory', self.dask_memory)
                mlflow.log_param('dask_job_extra_directives', self.dask_job_extra_directives)
                mlflow.log_param('dask_address', self.dask_address)
                mlflow.log_param('n_gpus', self.n_gpus)
                # hpo parameters
                mlflow.log_param('search_algorithm', search_algorithm)
                mlflow.log_param('trial_scheduler', trial_scheduler)
                mlflow.log_param('n_trials', n_trials)
                mlflow.log_param('timeout_experiment', timeout_experiment)
                mlflow.log_param('timeout_trial', timeout_trial)
                mlflow.log_param('max_concurrent_trials', max_concurrent_trials)
                mlflow.log_param('retrain_best_model', retrain_best_model)
                parent_run_uuid = run.info.run_uuid
                return self.run_combination_hpo(n_jobs=n_jobs,
                                                create_validation_set=create_validation_set,
                                                model_params=model_params,
                                                parent_run_uuid=parent_run_uuid,
                                                logging_to_mlflow=logging_to_mlflow,
                                                retrain_best_model=retrain_best_model, return_results=return_results,
                                                **unique_params)
        else:
            return self.run_combination_hpo(n_jobs=n_jobs,
                                            create_validation_set=create_validation_set,
                                            model_params=model_params,
                                            parent_run_uuid=parent_run_uuid,
                                            logging_to_mlflow=logging_to_mlflow,
                                            retrain_best_model=retrain_best_model, return_results=return_results,
                                            **unique_params)

    def setup_dask(self, n_workers, cluster_type='local', address=None):
        # we need to increase number of threads and open files for ray to work,
        # it creates a lot of threads and open files, even though only some are active at the same time
        # if it is a local cluster this must be setup manually before running the program
        if cluster_type == 'slurm':
            job_script_prologue = dask.config.get(
                "jobqueue.slurm.job-script-prologue", []
            )
            job_script_prologue = job_script_prologue + [
                # https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html
                "ulimit -n 65535",
                "ulimit -u 65535",
            ]
            dask.config.set({
                'jobqueue.slurm.job-script-prologue': job_script_prologue
            })
        return super().setup_dask(n_workers, cluster_type, address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    experiment = HPOExperiment(parser=parser)
    experiment.run()
