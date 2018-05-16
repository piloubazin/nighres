#!/usr/bin/env python
"""
"""

import sys
import pickle
from nipype import config, logging
from nighres.lesion_tool.lesion_pipeline import Lesion_extractor

wf_name = sys.argv[1]
base_dir = sys.argv[2]
input_dir = sys.argv[3]
sub_file = sys.argv[4]
atlas = sys.argv[5]
grid = sys.argv[6]

subjects = pickle.load(open(sub_file,"rb"))

wf = Lesion_extractor(wf_name=wf_name,
                      base_dir=base_dir,
                      input_dir=input_dir,
                      subjects=subjects,
                      #main=main,
                      #acc=acc,
                      atlas=atlas)

config.update_config({'logging': {'log_directory': wf.base_dir,'log_to_file': True}})
logging.update_logging(config)
config.set('execution','job_finished_timeout','20.0')
wf.config['execution'] = {'job_finished_timeout': '10.0'}
try:
    wf.run('SLURM', plugin_args={'sbatch_args': '-p ' + grid })
    #wf.write_graph(dotfilename='WAIMEA', graph2use='colored', format='png', simple_form=True)
except:
    print('Error! Pipeline exited with exception:')
    raise

