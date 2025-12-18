import os
import numpy as np
from autopf.utils import automoose, read_job_stats


if __name__ == "__main__":
        mooseapp = 'allen_cahn-opt'  # Replace with actual MOOSE app path
        # create a paramter set for kappa (interfacial energy) and L (mobility) varying from 0.1 to 1.0
        kappa_values = np.linspace(0.1, 1.0, 10)
        L_values = np.linspace(0.1, 1.0, 10)

        arg_list = []
        dir_list = []
        for L in L_values:
            for kappa in kappa_values:
                arg_list.append([f'Materials/consts/prop_values={L}',
                                  f'Kernels/ACInterface/kappa_name={kappa}'])
                try:
                     os.makedirs(f'output', exist_ok=True)
                except:
                     pass
                dir_list.append(f'output/AC_L_{L}_kappa_{kappa}')

        params = {'total_jobs': 100,
                 'base_input': 'AC.i',
                 'arg_list': arg_list,  
                 'num_cores': 4,
                 'directory_list': dir_list}
        
        # bootstrap the allen-cahn simulations
        automoose(mooseapp, params)

        # Read stats with directory and argument mapping
        job_stats = read_job_stats(dir_list=dir_list, arg_list=arg_list)

        # Write job stats to file
        import json
        from datetime import datetime
        
        output_filename = f'job_stats_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_filename, 'w') as f:
            json.dump(job_stats, f, indent=2, default=str)
        
        # print(f"Job statistics written to: {output_filename}")
        
        # # Print summary
        # print(f"\nJob Summary:")
        # print(f"  Completed: {len(job_stats.get('Completed tasks', []))}")
        # print(f"  Running: {len(job_stats.get('Running tasks', []))}")
        # print(f"  Pending: {len(job_stats.get('Pending tasks', []))}")
        # print(f"  Failed: {len(job_stats.get('Failed tasks', []))}")
     


        


     


