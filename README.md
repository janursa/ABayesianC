
  

# Approximate Bayesian calculation (ABC)

This package conducts ABC on a given model and parameters. Basically, ABayesianC does the following:

- Sample uniformly from the n-dimensional space of the free parameters 

- Create a parameter set for each of sample set

- Run the given model for each parameter set and collect the error value

- Choose the best fits by the rejection algorithm 

  
## Getting started

### Quick start

`pip install --upgrade ABayesianC`

```py

# inside your script, e.g. test.py

from ABayesianC import tools

obj = tools.ABC(settings = settings, free_params = free_params)

obj.sample()

obj.run()

obj.postprocess()

```


### More on it

The module receives two inputs from users.  First, the free parameters' list that is a python dictionary containing the names and bounds (min and max) of each free parameter, as shown below:

```python
free_params = {
    'p_name_1': [1.1,4.3], # [min,max]/ Prior
    'p_name_2': [6.4,23.1]
}
```
Second, the settings variable that is another python dictionary containing:

```py
settings = {
    "MPI_flag": True, # whether to use MPI or not
    "sample_n": 10000,  # Sample number
    "top_n": 100, # Number of top selected samples, i.e. posterior
    "output_path": "outputs", # Relative output directory to save the results
    "replica_n":3 #  number of replica run for each param set
    "model": Model # the model that receives the parameter set and returns the error value
        
}
```
The provided `model` must:
- receive a parameter set as argument 
- has a function named `run` 
- the `run` function runs the model and returns back the error/fitness value

### Parallel run

To run the constructed script, e.g. `test.py`, in parallel, commain in terminal,

```py
mpiexec -n available_cpu_core python test.py
```
`available_cpu_core` is the CPU core number that user intend to allocate for this process. For more info, see [MPI for Python](https://mpi4py.readthedocs.io/en/stable/).

### Outputs

Among the library outputs are:
- `samples.txt`: the samples in the n-dimensional space of the free parameters
- `distances.txt`: the distances/errors/fitness values obtained for each parameter set
- `best_distances.txt`: the best n distances. n is defined in the settings 
- `posterior.json`: the posteriors extracted for each free parameter using top n best fit
- `medians.json`: the medians of the posteriors for each free parameter. These values can be considered as inferred values.

## Install

Using pip manager:

-  `pip install --upgrade ABayesianC`

Or, download the package and in the root folder, command:

-  `python3 setup.py install`

## Authors

- Jalil Nourisa

## Useful links

 [MPI for Python](https://mpi4py.readthedocs.io/en/stable/).

## Contributing to ABayesianC
In case of encountering a problem, pls report it as an issue or contant the author (jalil.nourisa@gmail.com)
