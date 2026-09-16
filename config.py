import os

results_file_path = os.path.join(os.path.dirname(__file__), 'results/')
comments = 'stats'
ra_results_file_path = os.path.join(results_file_path, 'results_' + comments + '.csv')
ra_avg_results_file_path = os.path.join(results_file_path, 'avg_results_' + comments + '.csv')

config_obj_lambda = 0.9993
config_num_hops = 2