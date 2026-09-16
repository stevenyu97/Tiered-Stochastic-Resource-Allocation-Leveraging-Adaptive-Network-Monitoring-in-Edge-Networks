**Resource Allocator**

This package provides resource allocation methods for assiging tasks to compute resources

# Setup standalone Solver on local machine:
##### In ../ramite2-optimization directory.
```bash
#create a new conda environment standalone and install requirements.txt file on your local machine
conda create -y -n standalone python==3.10 pip3
conda activate standalone
pip3 install -r requirements.txt

#After environment setup run solver placement scenarios with the various parameters for resources, timeliness, images and accuracy requirements:
./run_batch.sh
#This outputs solver placements in the results folder



#if you have "ModuleNotFoundError: No module named 'config'" you need to set PYTHONPATH to your current directory:
#export PYTHONPATH=~/ramite/ramite2-optimization:$PYTHONPATH
```
##### Documentation
our default topology and resources are located in ramite/queries, its hierarchical & undirected graph, CHQ->APC->PLT->UGS (each node type indicates a stepdown in resources):
Here these 2 are the final processed query inputs that feed into the Solver:
* mean_band.csv: defines node topology and bandwidth for those links in bits per second
* resource_query.csv: each nodes compute resources-memory & cpu(millicores)


### MAIN file to run the RAMITE Solver: where the tasks and (topology/resouces) are passed to solver to be placed
* examples/AnglovaScenarioDAVC.py: 

### get loaded in the solver at initialization:
* workflows.csv: are discrete benchmarked workflows the solver ingests to place tasks (these will remain unchanged)


### Experimental generation to create additional random toplogies/resources for experimentation
* ramite/queries/network_generator/network_creator_csv.py

### The algorithm implementation of the SCIP MIP solver: variables and constraints
* ramite/solvers.py