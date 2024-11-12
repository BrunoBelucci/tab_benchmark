import mlflow
from sqlalchemy import create_engine
import pandas as pd
from pathlib import Path
from shutil import rmtree
import argparse


def clean_orphan_artifactory(tracking_url, artifactory_root_dir, use_sql=True):
    # get all experiments artifactories location
    if not use_sql:
        mlflow.set_tracking_uri(tracking_url)
        experiments = mlflow.search_experiments()
        experiment_artifactories = {Path(exp.artifact_location).resolve() for exp in experiments}
    else:
        engine = create_engine(tracking_url)
        query = 'SELECT experiments.artifact_location from experiments'
        experiment_artifactories = set(pd.read_sql(query, engine)['artifact_location'].apply(
            lambda x: Path(x).resolve()).tolist())
    # get all dir that should represent an artifactory location
    experiment_existent_artifactories = {Path(art_dir).resolve() for art_dir in artifactory_root_dir.iterdir()}
    # get what exists in the filesystem but not in the mlflow tracking server
    orphan_artifactories = experiment_existent_artifactories - experiment_artifactories
    # remove orphan artifactories
    for orphan_artifactory in orphan_artifactories:
        rmtree(orphan_artifactory)

    # now do the same for runs inside the experiments
    if not use_sql:
        runs = mlflow.search_runs(search_all_experiments=True)
        # get all runs artifactories location
        runs_artifactories = {Path(run.artifact_uri).parent.resolve() for run in runs.itertuples()}
    else:
        query = 'SELECT runs.artifact_uri from runs'
        runs_artifactories = set(pd.read_sql(query, engine)['artifact_uri'].apply(
            lambda x: Path(x).parent.resolve()).tolist())

    runs_existent_artifactories = {run_dir.resolve()
                                   for experiment_dir in artifactory_root_dir.iterdir()
                                   for run_dir in experiment_dir.iterdir()}
    orphan_runs_artifactories = runs_existent_artifactories - runs_artifactories
    for orphan_run_artifactory in orphan_runs_artifactories:
        rmtree(orphan_run_artifactory)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean orphan artifactories from mlflow tracking server')
    parser.add_argument('--mlflow_tracking_uri', type=str, required=True, help='mlflow tracking uri')
    parser.add_argument('--artifactory_root_dir', type=str, required=True, help='root dir of the artifactories')
    args = parser.parse_args()
    artifactory_root_dir = Path(args.artifactory_root_dir)
    clean_orphan_artifactory(args.mlflow_tracking_uri, artifactory_root_dir)
