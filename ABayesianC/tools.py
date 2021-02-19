"""This module contains ABC class
Author: Jalil Nourisa
"""
import time
import os
# from pprogress import ProgressBar
import json

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

    """ Contains essential function for ABC 
    
    Attributes:
        comm : MPI communication object
        rank (int): ID of each processor
        free_params (dict): Content of free parameteres including their tags and bounds
        free_params_bounds (narray): Bounds for each free parameter
        free_params_keys (array): Names of free parameters
        param_sets (list): The list of pararameter sets created during sampling
        settings (dict): Settings of the analysis
    """

    def __init__(self,free_params,settings):
        """Generates ABM object. Receives free paramatere lists and settings.
        
        Args:
            free_params (dict): Content of free parameteres including their tags and bounds
            settings (dict): Settings of the analysis
        """
        self.settings = settings

        if self.settings["MPI_flag"]:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
        else:
            self.rank = 0

        if self.rank == 0:
            print("Number of CPUs assigned: ",self.comm.Get_size())
            print("Sample number: ",settings['sample_n'])
            self.free_params = free_params
            self.free_params_keys = list(free_params.keys())
            self.free_params_bounds = list(free_params.values())
            print("The list of free parameters: ",self.free_params_keys)
            # try:
            #     os.makedirs(self.settings["output_path"])
            # except OSError:
            #     print("Creation of the directory %s failed" % self.settings["output_path"])
            # else:
            #     print("Successfully created the directory %s " % self.settings["output_path"])

    def sample(self):
        """Conducts
        - Uniform sampling from n-dimensional space of parameters within the bounds given as ABC.free_params.
        - Creates parameter sets and outputs them

        """
        if self.rank == 0:
            import numpy as np
            from diversipy import lhd_matrix
            from diversipy import transform_spread_out
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
            np.savetxt(self.settings["output_path"]+'/samples.txt', samples, fmt='%.12f')
            with open(self.settings["output_path"]+'/priors.json','w') as file:
                file.write(json.dumps(priors))
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
        """Runs the user given model for the parameter sets. 
        """
        if self.rank == 0:
            import numpy as np

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
                start = sum(shares[0:i])
                end = start + shares[i]
                portions.append([start,end])
            paramsets = self.param_sets

        else:
            portions = None
            paramsets = None

        portion = self.comm.scatter(portions,root = 0)    
        paramsets = self.comm.bcast(paramsets,root = 0) 

        def run_model(start,end):
            #print("\n start {} end {} rank {}".format(start,end,self.rank))
            # pb = ProgressBar(end-start)
            distances = []
            for i in range(start,end):
                replicas = []
                flag = True
                for j in range(self.settings["replica_n"]):
                    distance_replica = self.settings["model"](paramsets[i]).run()
                    if distance_replica is None:
                        distances.append(None)
                        flag = False
                        break
                    else:
                        replicas.append(distance_replica)
                if flag is False:
                    continue
                distance = sum(replicas)/len(replicas)
                distances.append(distance)
                # pb.update()
            # pb.done()
            return distances
        distances_perCore = run_model(portion[0],portion[1])
        distances_stacks = self.comm.gather(distances_perCore,root = 0)
        if self.rank == 0:
            import numpy as np
            distances = np.array([])
            for stack in distances_stacks:
                distances = np.concatenate([distances,stack],axis = 0)

            np.savetxt(self.settings["output_path"]+'/distances.txt',np.array(distances),fmt='%s')
    def postprocessing(self):
        """Conducts post processing tasks. Currently it extracts top fits and posteriors and also plots scaled posteriors.  
        """
        if self.rank == 0:
            # reload
            import numpy as np

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
            for i in range(len(distances)):
                if distances[i] == None:
                    distances[i] = 1000 # very high number

            # top_ind = np.argpartition(fitness_values, -top_n)[-top_n:]
            distances = np.array(distances)
            best_indices = distances.argsort()[:top_n]
            best_distances = distances[best_indices]
            np.savetxt(self.settings["output_path"]+'/best_distances.txt',best_distances,fmt='%f')
            np.savetxt(self.settings["output_path"]+'/top_ind.txt',best_indices,fmt='%d')

            # extract posteriors
            top_fit_samples = samples[best_indices].transpose()
            try:
                posteriors = {key:list(value) for key,value in zip(self.free_params_keys,top_fit_samples)}
            except TypeError:
                posteriors = {self.free_params_keys[0]:list(top_fit_samples)}
            with open(self.settings["output_path"]+'/posterior.json', 'w') as file:
                 file.write(json.dumps({'posteriors': posteriors}))
            # calculate median value
            from statistics import median
            medians = {}
            for (key,distribution) in posteriors.items():
                medians.update({key:median(distribution)})
            with open(self.settings["output_path"]+'/medians.json', 'w') as file:
                 file.write(json.dumps({'medians': medians}))



    def run_tests(self):
        if not self.settings["test"]:
            return
        if self.rank == 0:
            print("Running tests")
            import numpy as np
            top_ind = np.loadtxt(self.settings["output_path"]+'/top_ind.txt')
            top_ind = np.array(top_ind,int)
            # reload parameter sets generated during sampling
            with open(self.settings["output_path"]+'/param_sets.json') as file:
                self.param_sets = np.array(json.load(file)["param_sets"])
            # exctract the top parameter sets
            top_param_sets = self.param_sets[top_ind]
            top_param_sets_json = {'top_param_sets':list(top_param_sets)}
            with open(os.path.join(self.settings["output_path"],'top_param_sets.json'),'w') as file:
                file.write(json.dumps(top_param_sets_json,indent = 4))
            # get the CPU info and assign tasks for each
            CPU_n = self.comm.Get_size()
            shares = np.ones(CPU_n,dtype=int)*int(len(top_param_sets)/CPU_n)
            plus = len(top_param_sets)%CPU_n
            for i in range(plus):
                shares[i]+=1
            portions = []
            for i in range(CPU_n):
                start = i*shares[i-1]
                end = start + shares[i]
                portions.append([start,end])
            paramsets = top_param_sets

        else:
            portions = None
            paramsets = None

        portion = self.comm.scatter(portions,root = 0)
        paramsets = self.comm.bcast(paramsets,root = 0)

        def run_model(start,end):
            pb = ProgressBar(end-start)
            results_perCPU = []
            for i in range(start,end):
                results = self.settings["model"](paramsets[i]).test()
                results_perCPU.append(results)
                pb.update()
            pb.done()
            return results_perCPU
        top_results_perCore = run_model(portion[0],portion[1])
        # receive results of each CPU and stack them
        top_results_stacks = self.comm.gather(top_results_perCore,root = 0)
        if self.rank == 0:
            import numpy as np

            top_results = np.array([])
            for stack in top_results_stacks:
                top_results = np.concatenate([top_results,stack],axis = 0)
            # output the top results
            with open(os.path.join(self.settings["output_path"],'top_results.json'),'w') as file:
                        file.write(json.dumps({'top_results':list(top_results)},indent=4))