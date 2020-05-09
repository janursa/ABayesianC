import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from scipy.stats import norm, kurtosis
import json


"""@package params_joint_plot
This module reads accepted parameter values for each simulation and plots marginal plots for each combination of the parameters 

"""
class tools:
    def joint_dist(self,figs_dir):
        record = {}
        def CHECK_REPETITION(record,tag1,tag2):
            if tag1 in record:
                if record[tag1] == tag2 :
                    return True;
            if tag2 in record:
                if record[tag2] == tag1:
                    return True;
        
        for tag1 in self.data:
            for tag2 in self.data:
                if tag1==tag2 :
                    continue
                if CHECK_REPETITION(record,tag1,tag2):
                    continue
                plt.figure()
                h = sns.jointplot(x=self.data[tag1]["list"], y=self.data[tag2]["list"], kind='scatter', ratio=3,
                              xlim = self.data[tag1]["limits"], ylim = self.data[tag2]["limits"]
                              )
                h.set_axis_labels(tag1, tag2, fontsize=16)
                # sns.jointplot(x=self._param_dist_db[tag1], y=self._param_dist_db[tag2], kind='scatter', ratio=3)
                graph_save_name = figs_dir + "marginal_" + tag1 + "_" + tag2 + ".png"
                plt.savefig(graph_save_name,bbox_inches = 'tight')
                plt.close()

                record[tag1] = tag2
        return
    def single_dist(self,figs_dir,norm_order):
        for tag in self.data:
            plt.figure()
            if norm_order == True: # for parameters for calibration  
                sns.distplot(self.data[tag]["list"],fit=norm, kde=False,color="g",hist_kws={'linewidth':10},fit_kws={'linewidth':4})
                plt.xlabel('Parameter value', fontsize = 18)
            else: # for fitness
                sns.distplot(self.data[tag]["list"],kde=False,color="g",hist_kws={'linewidth':10},fit_kws={'linewidth':4})
                plt.xlabel('Fitness', fontsize = 18)
            graph_save_name = figs_dir + str(tag)+".png"
            
            plt.title(tag)
            plt.tick_params(axis="both",labelsize=18);
            plt.savefig(graph_save_name,bbox_inches = 'tight')
            plt.close()
        return
    def stats(self,_dir):
        stats = {}
        # print(self.data)
        for tag in self.data:
            dist = self.data[tag]["list"]
            kvalue =  kurtosis(dist)
            mean  = np.mean(dist)
            
            stats[tag] = []
            stats[tag].append({
                    'mean':mean,
                    'kurtosis':kvalue
                })
        file_name = _dir + "stats.json"
        with open(file_name,'w') as output:
            json.dump(stats,output,indent=4)
        return
    def read_parameter_file(self,file_name):
        with open(file_name) as json_file:
            self.data = json.load(json_file)

        # self._parameter_tags = []
        # self._param_dist_db = 0
        # fid = open(file_name, 'r')
        # content =[]
        # for line in fid:
        #     content.append(line)
        # ii = 0
        # db = {} # the content of dataframe
        # param_boundaries = {}
        # while True:
        #     # print(ii)
        #     try:
        #         header = content[ii].split()
        #     except:
        #         break
        #         # return
        #     tag = header[0]   # parameter name
        #     self._parameter_tags.append(tag)
        #     param_min_value = self.num(header[1]) #parameter minimum value
        #     param_max_value = self.num(header[2]) #parameter max value
        #     param_boundaries.update({tag:[param_min_value,param_max_value]})
        #     # print(param_boundaries)
        #     param_data = content[ii+1].split()
        #     param_data = [self.num(x) for x in param_data]
        #     db.update({tag:param_data})
        #     # param_content = {tag1:{}}
        #     ii+=2
        # self._param_dist_db = pd.DataFrame(data=db)
        # self._param_boundaries_db = pd.DataFrame(data=param_boundaries)

    def num(self,x):
        try:
            return int(x)
        except:
            return float(x)
    # _paremeters ={} # a dictionary that contains parameters by its tag
    # _parameter_tags=[]
    # _param_dist_db = 0 # pandas dataframe to contain parameters distribution values
    # _param_boundaries_db =0; #pandas dataframe to store parameters lower and upper 
    data =0

if __name__ == "__main__":
    params_file = sys.argv[1]
    to_dir = sys.argv[2]
    order = sys.argv[3]

    # params_file = "/Users/matin/Downloads/testProjs/ABM/build/outputs/ABC_outputs/posterior/post_dist";
    # to_dir = "/Users/matin/Downloads/testProjs/ABM/build/outputs/ABC_outputs/posterior/";
    # order = "stats"
    tools_ = tools()
    tools_.read_parameter_file(params_file);
    if order == "params_joint_distribution_plot":
        tools_.joint_dist(to_dir)
    if order == "params_single_distribution_plot":
        tools_.single_dist(to_dir,True)
    if order == "fitness_distribution_plot":
        tools_.single_dist(to_dir,False)
    if order == "stats":
        tools_.stats(to_dir)
        

