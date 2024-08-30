from tab_benchmark.datasets import datasets_characteristics_path
import pandas as pd
import argparse


def get_tasks_ids(tasks_names):
    datasets = pd.read_csv(datasets_characteristics_path)
    tasks_ids = datasets.loc[datasets['task_name'].isin(tasks_names), 'task_id'].tolist()
    return tasks_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks_names', nargs='+', type=str)
    args = parser.parse_args()
    tasks_ids = get_tasks_ids(args.tasks_names)
    print(tasks_ids)
