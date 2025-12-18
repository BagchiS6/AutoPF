
from matensemble.matflux import SuperFluxManager
import numpy as np
from typing import Dict, List, Union, Optional
import os

def automoose(mooseapp: str, 
                    params: Dict[str, Union[int, str, List, Optional[List[str]]]], 
                    write_restart_freq: int = 10,
                    buffer_time: float = 0.5,
                    adaptive_load_balance: bool = True) -> None:
    """
    Automatically sets up and runs high-throughput Phase Field simulations using MOOSE.

    Parameters:
    - mooseapp: The MOOSE application executable path.
    - params: A dictionary containing simulation parameters such as:
        - 'total_jobs': Size of the total number of MOOSE jobs.
        - 'base_input': Base input file for MOOSE simulations.
        - 'arg_list': list of moose job argument list (list of lists in case each job requires multiple args to be passed).
        - 'num_cores': Number of cores to use for each MOOSE job. 
        - 'directory_list': (optional) Directory list to store simulation outputs.
    """

    sim_cmd = f'{os.path.abspath(mooseapp)} -i {os.path.abspath(params["base_input"])}'
    sim_arg_list = params['arg_list']
    N_sims = params['total_jobs']
    n_cores = params['num_cores']
    sim_list = list(np.arange(N_sims))
    sim_dir_list = params.get('directory_list', None)

    sfm = SuperFluxManager(gen_task_list=sim_list,
                                gen_task_cmd=sim_cmd,
                                ml_task_cmd=None,
                                tasks_per_job=n_cores,
                                cores_per_task=1,
                                gpus_per_task=0,
                                write_restart_freq=write_restart_freq)
    sfm.poolexecutor(task_arg_list=sim_arg_list,
                     buffer_time=buffer_time,
                     task_dir_list=sim_dir_list,
                     adaptive=adaptive_load_balance)
    return

def get_latest_restart_file(directory: str = '.') -> Optional[str]:
    """
    Find the latest (highest numbered) restart file in the directory.

    Parameters:
    - directory: Directory to search for restart files (default: current directory)

    Returns:
    - Path to the latest restart file, or None if no restart files found.
    """
    import glob
    import re
    
    restart_files = glob.glob(os.path.join(directory, 'restart_*.dat'))
    
    if not restart_files:
        return None
    
    # Extract numbers from filenames and find the max
    max_num = -1
    latest_file = None
    
    for file in restart_files:
        match = re.search(r'restart_(\d+)\.dat', file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_file = file
    
    return latest_file

def read_job_stats(stat_file: Optional[str] = None, 
                   dir_list: Optional[List[str]] = None,
                   arg_list: Optional[List] = None) -> Dict:
    """
    Reads job execution statistics from a pickle file and maps them to directories/arguments.
    If no file is specified, automatically finds and reads the latest restart file.

    Parameters:
    - stat_file: Path to the pickle file containing job statistics. 
                 If None, searches for the latest restart_*.dat file.
    - dir_list: Optional list of directories corresponding to each job ID.
    - arg_list: Optional list of arguments corresponding to each job ID.

    Returns:
    - A dictionary with job execution statistics, enriched with directory and argument mappings.
    """
    import pickle
    
    if stat_file is None:
        stat_file = get_latest_restart_file()
        if stat_file is None:
            raise FileNotFoundError("No restart files found in current directory")
        print(f"Reading latest restart file: {stat_file}")
    
    with open(stat_file, 'rb') as f:
        job_stats = pickle.load(f)
    
    # Enrich job stats with directory and argument information
    if dir_list is not None or arg_list is not None:
        for category in ['Completed tasks', 'Running tasks', 'Pending tasks', 'Failed tasks']:
            if category in job_stats:
                job_stats[f'{category} details'] = []
                for job_id in job_stats[category]:
                    detail = {'job_id': job_id}
                    if dir_list is not None and job_id < len(dir_list):
                        detail['directory'] = dir_list[job_id]
                    if arg_list is not None and job_id < len(arg_list):
                        detail['arguments'] = arg_list[job_id]
                    job_stats[f'{category} details'].append(detail)
    
    return job_stats