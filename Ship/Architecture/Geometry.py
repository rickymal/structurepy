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
from shapely import geometry


def deprecated(function):
    pass

class Geometry:
    
    
    def check_polygon(self,poligono):
        c1 = len(poligono) > 3
        c2 = np.equal(poligono[0],poligono[-1])
        
    @deprecated
    @classmethod
    def cut_axis2(cls,poligono,cut_axis):
        
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
            if vx[0] == cut_axis and vx[1] == cut_axis:
                response = np.mean(vx)
            else:
                if vx[0] > vx[1]:
                    response = interp(cut_axis,vx[::-1],vy[::-1],left = nan,right = nan)
                else:
                    response = interp(cut_axis,vx,vy,left = nan,right = nan)
                
            new_pos_axis.append(response)
        #new_pos_axis.append(0.0)
        
        imatrix = np.array(tuple(enumerate(new_pos_axis)))
        mask = ~np.isnan(imatrix[:,0])
        filled = imatrix[mask]
        values, index = filled[:,1], filled[:,0].astype(np.int) + 1
        result =  [(cut_axis,value) for value in values]
        
        
        # realizando a inserção dos novos pontos no poligono
        r = np.insert(poligono,index,result, axis = 0)

        r = np.insert(r,len(r),[0.,max(poligono[:,1])],axis = 0)
        r = np.insert(r,len(r),[0.,min(poligono[:,1])],axis = 0)
        
        


        # realizando o filtro dos dados 
        r_filtered = r[r[:, 0] <= cut_axis]
        _, ind = np.unique(r_filtered, axis=0, return_index=True)
        mask = np.sort(ind)
        r_filtered = r_filtered[mask]
        #r_filtered = np.insert(r_filtered,len(r_filtered),[0.,max(poligono[:,1])],axis = 0)
        #r_filtered = np.insert(r_filtered,len(r_filtered),[0.,min(poligono[:,1])],axis = 0)
        return r_filtered
        
        # realizar a parte da máscara

    @deprecated
    @classmethod
    def cut(cls,poligono,cut_axis):
        
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
    
    @deprecated
    def attributes(self,poligono):
        #Obtendo a área e o centroid
        P = Polygon(poligono)
        area = P.area
        centroid = P.centroid.coords
        centroid = tuple(*centroid)
        return area, centroid[0],centroid[1]


    def _transversal_cut(self,polygon,cut_axis = 0.5,axis_to_cut ='horizontal'):
        _CUT_AXIS = cut_axis
        if _CUT_AXIS is None:
            _CUT_AXIS = max(polygon[:,1]) #não haverá corte


        suposted_polygon = polygon
        if suposted_polygon[-1,0] != 0.0:
            suposted_polygon = np.append(suposted_polygon,
                                        [0,suposted_polygon[-1,1]])\
                .reshape(-1,2)\
                    .astype(np.float16)
            
        ring_polygon = geometry.LinearRing(suposted_polygon)
        poligono = geometry.Polygon(ring_polygon)
        if axis_to_cut == 'horizontal':
            cut_line = geometry.LineString([
                [-1.,_CUT_AXIS],
                [max(suposted_polygon[:,0])+1.,_CUT_AXIS],
                ])
        elif axis_to_cut == 'vertical':
            pass
            cut_line = geometry.LineString([
                [_CUT_AXIS,-1.],
                [_CUT_AXIS,max(suposted_polygon[:,1])+1.]
                
                ])
        else:
            raise Exception("The string inputet isn't undestood")
        
        from shapely import ops 
        data = ops.split(poligono,cut_line)
        x,y = data[0].exterior.coords.xy
        matrix_cut_representation = np.stack([x,y],axis = 1)
        centroids = data[0].centroid.coords[0]
        
        return {
            'numeric' : matrix_cut_representation,
            'centroids' : centroids,
            'area' : data[0].area,
            'cut' : cut_axis,
            }
        

    def _superior_cut(self,listOf_polygons,distances,cut_axis):    
        lof = []
        _CUT_AXIS = cut_axis    
        sections = listOf_polygons
        distances = distances
        for section,distance in zip(sections,distances):
            temp = self._transversal_cut(section,cut_axis = _CUT_AXIS)
            max_value_x = temp['numeric'][:,0].max()
            mm = temp['numeric']
            mask = mm[:,1] == mm[:,1].max()
            value_c = mm[mask][:,0].max()
            lof.append([distance,value_c])
            
        numeric_lof = np.array(lof)
        
        if numeric_lof[-1,1] != 0.0:
            numeric_lof = np.append(numeric_lof,
                                                [numeric_lof[-1,0],0])\
                        .reshape(-1,2)\
                            .astype(np.float16)
        
        ring_polygon = geometry.LinearRing(numeric_lof)
        poligono = geometry.Polygon(ring_polygon)
        
        return {
            'numeric' : numeric_lof,
            'centroids' : poligono.centroid.coords[0],
            'area' : poligono.area,
            'cut' : cut_axis,
            }
        
    
    def _lateral_cut(self,listOf_sections,distances,cut_axis):
        lof = []
        _CUT_AXIS = cut_axis    
        for section,distance in zip(sections,distances):
            temp = self._transversal_cut(section,cut_axis = _CUT_AXIS,axis_to_cut = 'vertical')
            max_value_x = temp['numeric'][:,0].max()
            mm = temp['numeric']
            mask = mm[:,0] == mm[:,0].max()
            value_c = mm[mask][:,1].min()
            lof.append([distance,value_c])
            
        numeric_lof = np.array(lof)
        
        if numeric_lof[0,1] != 0.0:
            
            numeric_lof = np.insert(numeric_lof,
                                    0,
                                    [0.,numeric_lof[:,1].max()],)\
                .reshape(-1,2).astype(np.float16)
        
        ring_polygon = geometry.LinearRing(numeric_lof)
        poligono = geometry.Polygon(ring_polygon)
        
        return {
            'numeric' : numeric_lof,
            'centroids' : poligono.centroid.coords[0],
            'area' : poligono.area,
            'cut' : cut_axis,
            
            }


    

if __name__ == '__main__':


    def load_table(path = 'table_example.txt'):
        def _load_table(path):
            with open(path, 'r') as file:
                conteudo = file.read().split("-------------------------------")
                # # # print(conteudo[0].split('\n'))
        
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



    gem = Geometry()
    sections, distances = load_table()
    print("finished")
    response = gem._superior_cut(sections,distances = distances,cut_axis = 0.5)
    
    