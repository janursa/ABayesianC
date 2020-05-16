import time
import numpy as np
import os
import json
from diversipy import lhd_matrix
from diversipy import transform_spread_out
from plots import box_plot
class clock:
    start_t = 0
    end_t = 0
    @staticmethod
    def start():
        clock.start_t = time.time()
    @staticmethod
    def end():
        clock.end_t = time.time()
        print('Elapsed time: ',clock.end_t - clock.start_t)

class ABC:
    settings = 0
    comm = 0
    rank = 0
    param_sets = 0
    def __init__(self,free_params,settings):
        self.settings = settings
        if self.settings["MPI_flag"]:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
        else:
            self.rank = 0

        if self.rank == 0:
            self.free_params = free_params
            self.free_params_keys = list(free_params.keys())
            self.free_params_bounds = list(free_params.values())
            

            try:
                os.makedirs(self.settings["output_path"])
            except OSError:
                print ("Creation of the directory %s failed" % self.settings["output_path"])
            else:
                print ("Successfully created the directory %s " % self.settings["output_path"])


    def sample(self):
        if self.rank == 0:
            # python version > 3.6
            non_scalled_samples = transform_spread_out(lhd_matrix(self.settings["sample_n"], len(self.free_params))).transpose()
            scaled_samples = []
            ii = 0
            for bounds in self.free_params_bounds:
                low = bounds[0]
                high = bounds[1]
                pre_samples_param = non_scalled_samples[ii]
                samples_param = list(map(lambda x:x*(high-low)+low ,pre_samples_param))
                scaled_samples.append(samples_param)
                ii+=1
            priors = {key:value for key,value in zip(self.free_params_keys,scaled_samples)}
            samples = np.array(scaled_samples).transpose()
            np.savetxt(self.settings["output_path"]+'/samples.txt', samples, fmt='%f')
            ##### create parameter sets
            param_sets = []
            for sample in samples:
                param_set = {}
                for i in range(len(sample)):
                    sample_p = sample[i]
                    key = self.free_params_keys[i]
                    param_set.update({key:sample_p})
                param_sets.append(param_set)
            with open(self.settings["output_path"]+'/param_sets.json','w') as file:
                file.write(json.dumps({"param_sets":param_sets}))

            self.param_sets = param_sets

    def run(self):
        
        if self.rank == 0:
            # reload
            with open(self.settings["output_path"]+'/param_sets.json') as file:
                self.param_sets = json.load(file)["param_sets"]
            CPU_n = self.comm.Get_size()
            shares = np.ones(CPU_n,dtype=int)*int(len(self.param_sets)/CPU_n)
            plus = len(self.param_sets)%CPU_n
            for i in range(plus):
                shares[i]+=1
            portions = []
            for i in range(CPU_n):
                start = i*shares[i-1]
                end = start + shares[i]
                portions.append([start,end])
            paramsets = self.param_sets

        else:
            portions = None
            paramsets = None

        portion = self.comm.scatter(portions,root = 0)    
        paramsets = self.comm.bcast(paramsets,root = 0) 

        def run_model(start,end):
            distances = []
            for i in range(start,end):
                distance = self.settings["run_func"](paramsets[i],self.settings["args"])
                distances.append(distance)
            return distances
        distances_perCore = run_model(portion[0],portion[1])
        

        distances_stacks = self.comm.gather(distances_perCore,root = 0)
        if self.rank == 0:
            distances = np.array([])
            for stack in distances_stacks:
                distances = np.concatenate([distances,stack],axis = 0)

            np.savetxt(self.settings["output_path"]+'/distances.txt',np.array(distances),fmt='%s')
    def postprocessing(self):
        if self.rank == 0:
            # reload 
            distances = []
            with open(self.settings["output_path"]+'/distances.txt') as file:
                for line in file:
                    line.strip()
                    try:
                        value = float(line)
                    except:
                        value = None
                    distances.append(value)
            samples = np.loadtxt(self.settings["output_path"]+'/samples.txt', dtype=float)

            # top fitnesses
            top_n = self.settings["top_n"]
            fitness_values = np.array([])
            for item in distances:
                if item == None:
                    fitness = 0
                else:
                    fitness = 1 - item
                fitness_values = np.append(fitness_values,fitness)
            top_ind = np.argpartition(fitness_values, -top_n)[-top_n:]
            top_fitess_values = fitness_values[top_ind]
            np.savetxt(self.settings["output_path"]+'/top_fitness.txt',top_fitess_values,fmt='%f')

            # extract posteriors
            top_fit_samples = samples[top_ind].transpose()
            try:
                posteriors = {key:list(value) for key,value in zip(self.free_params_keys,top_fit_samples)}
            except TypeError:
                posteriors = {self.free_params_keys[0]:list(top_fit_samples)}
            with open(self.settings["output_path"]+'/posterior.json', 'w') as file:
                 file.write(json.dumps({'posteriors': posteriors}))

            # box plot
            scalled_posteriors = {}
            for key,values in posteriors.items():
                min_v = self.free_params[key][0]
                max_v = self.free_params[key][1]
                scalled = list(map(lambda x: (x-min_v)/(max_v-min_v),values))
                scalled_posteriors.update({key:scalled})
            box_plot(scalled_posteriors,self.settings["output_path"])
