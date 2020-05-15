import time
import numpy as np
import os
import json
from diversipy import lhd_matrix
from diversipy import transform_spread_out

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
            np.savetxt('outputs/samples.txt', samples, fmt='%f')
            ##### create parameter sets
            param_sets = []
            for sample in samples:
                param_set = {}
                for i in range(len(sample)):
                    sample_p = sample[i]
                    key = self.free_params_keys[i]
                    param_set.update({key:sample_p})
                param_sets.append(param_set)
            with open("outputs/param_sets.json",'w') as file:
                file.write(json.dumps({"param_sets":param_sets}))

            self.param_sets = param_sets


    def run_model(self,start_end):
    #     import copy

    #     start_n = start_end[0]
    #     end_n = start_end[1]
    #     distances = np.array([])
    # #     bar.start()
    #     for i in range(start_n,end_n):
    #         param_set = self.param_sets[i]
    # #         print("iteration ",i)
    #         schemes_copy = copy.deepcopy(self.schemes)
    #         rep_distances = []
    #         for rep_i in range(self.replica_n):
    #             try:
    #                 sim_results_list = self.Model(param_set).run(schemes_copy)
    #             except ValueError:
    #                 # None value if the param set leads to invalid definition
    #                 distances = np.append(distances,None) 
    #                 break
    #             distance = self.distance_function(sim_results_list,self.expectations)
    #             rep_distances.append(distance)
    #         else:
    # #             print(rep_distances)
    #             distances = np.append(distances,np.mean(rep_distances))
    # #         bar.update(i+1)
    # #     bar.finish()
    #     return distances
        return

    def run(self):
        
        if self.rank == 0:
            CPU_n = self.comm.Get_size()
            share = int(len(self.param_sets)/CPU_n)
            plus =  len(self.param_sets)%CPU_n
            start = [i*share for i in range(CPU_n)]
            paramsets = self.param_sets
            
        else:
            start = None
            share = None
            paramsets = None
            
        start = self.comm.scatter(start,root = 0)  
        share = self.comm.bcast(share,root = 0)  
        paramsets = self.comm.bcast(paramsets,root = 0) 
        
        end = start+share
        for i in range(start,end):
            print("start: ",start," end: ",end)
            # self.settings["run_func"](paramsets[i],self.settings["args"])

        # distances = [] 
        # for result in results:
        #     distances = np.concatenate([distances,result],axis=0)

        # np.savetxt('outputs/distances.txt',np.array(distances),fmt='%s')
