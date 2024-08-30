import argparse
from pathlib import Path


def generate_postgres_db_script(
        file_dir=Path.cwd(),
        file_name='start_db',
        conda_env='tab_benchmark',
        database_dir=Path(__file__).parent.parent / 'results',
        db_name='tab_benchmark', db_port=5001, mlflow_port=5002, generate_sbatch=True,
        # sbatch parameters
        n_cores=6, clust_name='clust1', job_name='tab_benchmark_db',
        output_job_dir='/home/users/belucci/outputs/%x.%J.out',
        error_job_dir='/home/users/belucci/outputs/%x.%J.err',
        wall_time='364-23:59:59',
):
    log_file = database_dir / (db_name + '.log')
    sh_content = f"""
    if [ ! -d {str(database_dir.absolute())} ]; then
        conda run -n {conda_env} initdb -D {str(database_dir.absolute())}
        echo "host	all	all	samenet	trust" >> {str(database_dir.absolute())}/pg_hba.conf
    fi
    conda run -n {conda_env} pg_ctl -D {str(database_dir.absolute())} -l {str(log_file.absolute())} -o "-h 0.0.0.0 -p {db_port}" start
    conda run -n {conda_env} createdb {db_name} -p {db_port}
    conda run -n {conda_env} mlflow server --backend-store-uri postgresql://localhost:{db_port}/{db_name} -h 0.0.0.0 -p {mlflow_port}
    """
    if generate_sbatch:
        sbatch_content = f"""#!/bin/sh
        #SBATCH -c {n_cores}
        #SBATCH -w {clust_name}
        #SBATCH --job-name={job_name}
        #SBATCH --output={output_job_dir}
        #SBATCH --error={error_job_dir}
        #SBATCH --time={wall_time}
        """
        file_content = sbatch_content + sh_content
        file_ext = '.sbatch'
    else:
        file_content = sh_content
        file_ext = '.sh'
    file_path = file_dir / (file_name + file_ext)
    with open(file_path, 'w') as file:
        file.write(file_content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', type=str, default=Path.cwd())
    parser.add_argument('--file_name', type=str, default='start_db')
    parser.add_argument('--conda_env', type=str, default='tab_benchmark')
    parser.add_argument('--database_dir', type=str, default=Path(__file__).parent.parent / 'results')
    parser.add_argument('--db_name', type=str, default='tab_benchmark')
    parser.add_argument('--db_port', type=int, default=5001)
    parser.add_argument('--mlflow_port', type=int, default=5002)
    parser.add_argument('--generate_sbatch', type=bool, default=True)
    parser.add_argument('--n_cores', type=int, default=6)
    parser.add_argument('--clust_name', type=str, default='clust1')
    parser.add_argument('--job_name', type=str, default='tab_benchmark_db')
    parser.add_argument('--output_job_dir', type=str, default='/home/users/belucci/outputs/%x.%J.out')
    parser.add_argument('--error_job_dir', type=str, default='/home/users/belucci/outputs/%x.%J.err')
    parser.add_argument('--wall_time', type=str, default='364-23:59:59')
    args = parser.parse_args()
    generate_postgres_db_script(**vars(args))
