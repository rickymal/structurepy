# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:26:23 2020

@author: Henrique Mauler
"""
import pandas as pd
import numpy as np
from numpy import array,arange,zeros,interp,nan
from scipy import integrate
from shapely.geometry import Polygon
import pdb
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


class Geometry:
    
    
    def check_polygon(self,poligono):
        c1 = len(poligono) > 3
        c2 = np.equal(poligono[0],poligono[-1])
        
    
    
    def cut(self,poligono,cut_axis):
        
        iter_ = range(len(poligono))
        poligono = poligono.astype(np.float16)
        new_pos_axis = list()
        for index in iter_[:-1]:
            inverted = False
            slc = slice(index,index + 2) #para selecionar uma linha
            vx = poligono[slc,0].copy() 
            vy = poligono[slc,1].copy()
            inv_ = False
            response = np.nan
            if vy[0] == cut_axis and vy[1] == cut_axis:
                response = np.mean(vx)
            else:
                if vy[0] > vy[1]:
                    response = interp(cut_axis,vy[::-1],vx[::-1],left = nan,right = nan)
                else:
                    response = interp(cut_axis,vy,vx,left = nan,right = nan)
                
            new_pos_axis.append(response)
        #new_pos_axis.append(0.0)
        
        imatrix = np.array(tuple(enumerate(new_pos_axis)))
        mask = ~np.isnan(imatrix[:,1])
        filled = imatrix[mask]
        values, index = filled[:,1], filled[:,0].astype(np.int) + 1
        result =  [(value,cut_axis) for value in values]
        
        
        # realizando a inserção dos novos pontos no poligono
        r = np.insert(poligono,index,result, axis = 0)
        
        # realizando o filtro dos dados 
        r_filtered = r[r[:, 1] <= cut_axis]
        _, ind = np.unique(r_filtered, axis=0, return_index=True)
        mask = np.sort(ind)
        r_filtered = r_filtered[mask]
        #r_filtered = np.insert(r_filtered, len(r_filtered), r_filtered[0], axis=0)
        r_filtered = np.insert(r_filtered,len(r_filtered),[0.,cut_axis],axis = 0)
        r_filtered = np.insert(r_filtered,len(r_filtered),[0.,min(poligono[:,1])],axis = 0)
        return r_filtered
        
        # realizar a parte da máscara
    
      
    def attributes(self,poligono):
        #Obtendo a área e o centroid
        P = Polygon(poligono)
        area = P.area
        centroid = P.centroid.coords
        centroid = tuple(*centroid)
        return area, centroid[0],centroid[1]


def hello():
    print("OLAR")

    