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
        params_count = len(self._parameter_tags)
        for i in range(params_count):
            for j in range(i+1,params_count):
                tag1 = self._parameter_tags[i]
                tag2 = self._parameter_tags[j]
                plt.figure()
                sns.jointplot(x=self._param_dist_db[tag1], y=self._param_dist_db[tag2], kind='scatter', ratio=3,
                              xlim = self._param_boundaries_db[tag1], ylim = self._param_boundaries_db[tag2]
                              )
                # sns.jointplot(x=self._param_dist_db[tag1], y=self._param_dist_db[tag2], kind='scatter', ratio=3)
                graph_save_name = figs_dir + "marginal_" + tag1 + "_" + tag2 + ".png"
                plt.savefig(graph_save_name)
                plt.close()
        return
    def single_dist(self,figs_dir):
        params_count = len(self._parameter_tags)
        for i in range(params_count):
            tag = self._parameter_tags[i]
            plt.figure()
            sns.distplot(self._param_dist_db[tag],fit=norm, kde=False,color="g",hist_kws={'linewidth':10},fit_kws={'linewidth':4})
            # sns.jointplot(x=self._param_dist_db[tag1], y=self._param_dist_db[tag2], kind='scatter', ratio=3)
            graph_save_name = figs_dir + tag+".png"
            plt.xlabel('Parameter value', fontsize = 18)
            plt.title(tag)
            plt.tick_params(axis="both",labelsize=18);
            plt.savefig(graph_save_name,bbox_inches = 'tight')
            plt.close()
        return
    def stats(self,_dir):
        stats = {}
        params_count = len(self._parameter_tags)
        for i in range(params_count):
            tag = self._parameter_tags[i]
            dist = self._param_dist_db[tag]
            
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
        self._parameter_tags = []
        self._param_dist_db = 0
        fid = open(file_name, 'r')
        content =[]
        for line in fid:
            content.append(line)
        ii = 0
        db = {} # the content of dataframe
        param_boundaries = {}
        while True:
            # print(ii)
            try:
                header = content[ii].split()
            except:
                break
                # return
            tag = header[0]   # parameter name
            self._parameter_tags.append(tag)
            param_min_value = self.num(header[1]) #parameter minimum value
            param_max_value = self.num(header[2]) #parameter max value
            param_boundaries.update({tag:[param_min_value,param_max_value]})
            # print(param_boundaries)
            param_data = content[ii+1].split()
            param_data = [self.num(x) for x in param_data]
            db.update({tag:param_data})
            # param_content = {tag1:{}}
            ii+=2
        self._param_dist_db = pd.DataFrame(data=db)
        self._param_boundaries_db = pd.DataFrame(data=param_boundaries)

    def num(self,x):
        try:
            return int(x)
        except:
            return float(x)
    # _paremeters ={} # a dictionary that contains parameters by its tag
    _parameter_tags=[]
    _param_dist_db = 0 # pandas dataframe to contain parameters distribution values
    _param_boundaries_db =0; #pandas dataframe to store parameters lower and upper bounds

if __name__ == "__main__":
    print("Marginal plot")
    # params_file = sys.argv[1]
    # to_dir = sys.argv[2]
    # order = sys.argv[3]

    params_file = "/Users/matin/Downloads/testProjs/ABM/build/outputs/ABC_outputs/posterior/post_dist";
    to_dir = "/Users/matin/Downloads/testProjs/ABM/build/outputs/ABC_outputs/posterior/";
    order = "stats"
    tools_ = tools()
    tools_.read_parameter_file(params_file);
    if order == "joint_plot":
        tools_.joint_dist(to_dir)
    if order == "single_plot":
        tools_.single_dist(to_dir)
    if order == "stats":
        tools_.stats(to_dir)
        

