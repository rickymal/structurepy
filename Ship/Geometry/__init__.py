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
        
    
    def load_table(self,path = 'table_example_2.txt'):
        def _load_table(path):
            with open(path, 'r') as file:
                conteudo = file.read().split("-------------------------------")
                # # print(conteudo[0].split('\n'))
        
                balizas = list()
                all_knunck = list()
                positions = list()
                aplt = 3
                for x in conteudo[:-1]:
                    position = x.split('\n')[aplt - 3].split(':')[-1].replace(',', '.')
                    position = float(position)
                    x = x.split('\n')[aplt:]
                    positions.append(
                        position
                    )
                    aplt = 5
                    nList = list()
                    nKunckles = list()
                    for position in x[:-1]:
                        temp_2 = position.split()
                        temp = [0., 0.]
                        temp[0] = float(temp_2[0].replace(',', '.'))
                        temp[1] = float(temp_2[1].replace(',', '.'))
                        nList.append([temp[0:2]])
                        if len(temp) == 3:
                            nKunckles.append(True)
                        else:
                            nKunckles.append(False)
                    values = np.array(nList, dtype=np.float16)
                    nKunckles = np.array(nKunckles, dtype=np.float16)
                    balizas.append(
                        values.reshape(-1, 2)
                    )
                    all_knunck.append(nKunckles.reshape(-1, 1))
        
            return {
                'balizas': np.array(balizas, ),
                'knunck': np.array(all_knunck),
                'position': np.array(positions, dtype=np.float16),
            }
        
        r = _load_table(path = path)
        balizas = r['balizas']
        positions = r['position']
                
        
        return balizas,positions
    
    
    
    def basify(self,matrix):
        x_ = matrix
        minimus_x = tuple(min(x[:, 1]) for x in x_)
        minimus_x = min(minimus_x)
        for ind, x in enumerate(x_):
            x_[ind][:, 1] = x_[ind][:, 1] - minimus_x
        return x_
    
    
def hello():
    print("OLAR")

    

if __name__ == '__main__':
    print("Definindo os modelos")
    quadrado = [
        [0,0],
        [10,0],
        [10,10,],
        [0,10],
        [0,0],
        ]
    
    
    poligono =  [
        [0,0],
        [10,0],
        [12,2],
        [13,4],
        [15,6],
        [17,6],
        [17,10],
        [2,11],
        [0,11],
        ]
    
    specs = [0.55,]
    
    
    #draft = ship_drafts[11]
    geometry = Geometry()
    
    ship_drafts, ship_distances = geometry.load_table(path = 'table_example.txt')
    geometry.basify(ship_drafts) #inline
    
    
    
    #Test
    #ship_drafts = [np.array(quadrado) for _ in range(5)]
    
    #ship_frafts = quadrado
    #ship_distances = np.arange(0,5)
    
    
    #EndTest
    
    
    
    ship_draft_areas = list()
    ship_draft_centroids = list()
    for draft in ship_drafts:
        y_values = np.sort(draft[:,1].ravel())
        y_values = np.unique(y_values)
        plt.plot(*draft.T)
        areas = list()
        y_ = list()
        centroids = list()
        for y in y_values:
            x = geometry.cut(draft,y)
            area,centroid,_  = geometry.attributes(x)
            plt.plot(*x.T)
            areas.append(area)
            centroids.append(centroid)
            y_.append(y)
        
        plt.plot(y_,areas)
        plt.show()
            
        s_area = pd.Series(areas,index = y_)
        s_centroid = pd.Series(centroids,index = y_)
        ship_draft_areas.append(s_area)
        ship_draft_centroids.append(s_centroid)
    dataframe_area = pd.DataFrame(ship_draft_areas).T * 2
    dataframe_centroid = pd.DataFrame(ship_draft_centroids).T
    dataframe_area_interpolated = dataframe_area.interpolate(method = 'index')
    dataframe_centroid_interpolated = dataframe_centroid.interpolate(method = 'index')
    
    dai = dataframe_area_interpolated
    dci = dataframe_centroid_interpolated
    
    # Realizando calculo de volume
    from scipy.integrate import simps
    print(len(ship_distances))
    print(len(dai))
    serie_volum = dai.apply(simps,axis = 1, x = ship_distances)
    
    
    
    
    attributes_output = {
        'area' : dai,
        'centroid' : dci,
        'volum' : serie_volum,
        }
    
    
    
    dataframe_area.plot()
    
    

    
    
    """
    
    
    x = cut(poligono,3)
    
    
    
    
    for i in range(0,12):
        x = cut(poligono,i) 
        area,centroid = attributes(x)
        plt.plot(*x.T)
    
    """
    
    
    
    
    
    
    
    
    