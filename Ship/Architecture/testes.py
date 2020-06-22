# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.

"""


import numpy as np
import pandas as pd
import shapely
from shapely import geometry
import matplotlib.pyplot as plt


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



#%% testando se uma matrix é de fato poligono
x = load_table()[0][2]


def _transversal_cut(polygon,cut_axis = 0.5,axis_to_cut ='horizontal'):
    _CUT_AXIS = cut_axis
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
#%%
# fazendo a vista superior


sections, distances = load_table()

def _superior_cut(listOf_polygons,distances,cut_axis):    
    lof = []
    _CUT_AXIS = cut_axis    
    for section,distance in zip(sections,distances):
        temp = _transversal_cut(section,cut_axis = _CUT_AXIS)
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
    
x = _superior_cut(sections,distances, 0.6)


#%% lateral





def _lateral_cut(listOf_sections,distances,cut_axis):
    lof = []
    _CUT_AXIS = cut_axis    
    for section,distance in zip(sections,distances):
        temp = _transversal_cut(section,cut_axis = _CUT_AXIS,axis_to_cut = 'vertical')
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

resultado = _lateral_cut(sections,distances,0.6) 
