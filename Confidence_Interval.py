# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:30:44 2019

@author: Amirh
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button
#%%
class ConInterval():
    """
    this algorithm takes in one parameter:
        
        data : which is a MxN numpy array, where M is number of samples and N
        is the number of features.
    """
    
    def __init__(self,data,verbose):    
        if type(data) is not np.ndarray:
            raise Exception("The input data should be of type numpy.ndarray")
        if data.ndim ==1:
            self.data = np.squeeze(data)
        else:
            self.data = data
        self.x_dim=self.data.shape[0]
        self.y_dim=self.data.shape[1]
        self.CI_value = 0.95
        self.verbose = verbose
    
    def calculate(self):
        self.CI= np.zeros((self.y_dim,2))
        self.mean= np.mean(self.data,axis=0) # across all samples
        
        for j in range(self.y_dim):
            self.CI[j,:]=np.array(st.t.interval(self.CI_value, len(self.data)-1, loc=np.mean(self.data[:,j]), scale=st.sem(self.data[:,j])))
        return self.CI
    
    def plot(self):
        
        fig,ax=plt.subplots(nrows=1,ncols=1)
        ax.plot(self.mean,'r',linewidth=1.5)
        ax.plot(self.CI[:,0],'b',linewidth=1)
        ax.plot(self.CI[:,1],'b',linewidth=1)
        ax.set_title(r"Confidence Interval Plot, Confidence Level %s"%(str(self.CI_value))[0:5])
        # turn the CI matrice to xy axis for shading the area
        CI_vert_UL = np.zeros((self.y_dim,2))
        CI_vert_LL = np.zeros((self.y_dim,2))
        for i in range(self.y_dim):
            CI_vert_LL[i]=np.array([i,self.CI[i,0]])
            CI_vert_UL[i]=np.array([i,self.CI[i,1]])
        self.CI_vert = np.concatenate((CI_vert_LL,CI_vert_UL[::-1]),axis=0)            
        poly = Polygon(self.CI_vert, facecolor='0.9', edgecolor='0.9')
        ax.add_patch(poly) 
        if self.verbose == True:                
            slider_ax = plt.axes([0.1, 0.03, 0.8, 0.03])
            CI_slider = Slider(slider_ax,'CI',0,1,0.95)
            
            def update(val):
                self.CI_value = CI_slider.val
                self.CI=self.calculate()
                ax.cla()
                ax.plot(self.mean,'r',linewidth=1.5)
                ax.plot(self.CI[:,0],'b',linewidth=1)
                ax.plot(self.CI[:,1],'b',linewidth=1)
                ax.set_title(r"Confidence Interval Plot, Confidence Level %s"%(str(self.CI_value))[0:5])
                # turn the CI matrice to xy axis for shading the area
                CI_vert_UL = np.zeros((self.y_dim,2))
                CI_vert_LL = np.zeros((self.y_dim,2))
                for i in range(self.y_dim):
                    CI_vert_LL[i]=np.array([i,self.CI[i,0]])
                    CI_vert_UL[i]=np.array([i,self.CI[i,1]])
                self.CI_vert = np.concatenate((CI_vert_LL,CI_vert_UL[::-1]),axis=0)            
                poly = Polygon(self.CI_vert, facecolor='0.9', edgecolor='0.9')
                ax.add_patch(poly) 
    
            CI_slider.on_changed(update) 

        return ax
    