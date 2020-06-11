import Geometry 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

class Ship(Geometry.Geometry):
    def __init__(self,):
        pass
            
    def load_table(self,path = 'table_example_2.txt'):
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
    
    def basify(self,matrix):
        x_ = matrix
        minimus_x = tuple(min(x[:, 1]) for x in x_)
        minimus_x = min(minimus_x)
        for ind, x in enumerate(x_):
            x_[ind][:, 1] = x_[ind][:, 1] - minimus_x
        return x_
    
    def attr(self,ship_draft):
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
                x = self.cut(draft,y)
                area,centroid,_  = self.attributes(x)
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
        # print(len(ship_distances))
        # print(len(dai))
        serie_volum = dai.apply(simps,axis = 1, x = ship_distances)

        #calcular lcb...
        lof_lcb = list()
        for calado,x in dai.iterrows():
            #x = np.array([ship_distances,dai.iloc[10,:]]) #amostra
            # print(type(x))
            transp_matrix = np.transpose([ship_distances,x.values])
            try:
                poligono = Polygon(transp_matrix)
                poligono.centroid.coords #to break if isn't a polygon
            except:
                poligono = np.nan  
            if isinstance(poligono,np.float) or not len(poligono.centroid.coords):
                lcb = np.nan
                lof_lcb.append((lcb,lcb))
            else:
                centroid = poligono.centroid.coords
                centroid = tuple(centroid)[0]
                lcb = centroid
                lof_lcb.append(lcb)
        lof_lcb_array = np.array(lof_lcb)
        columns = ['longitudinal','altura']
        index = [f'lcb({draft})' \
                 for draft in dai.index]      
        index = dai.index  
        lcb_table = pd.DataFrame(data = lof_lcb_array,
                     columns = columns,
                    index = index)
        lcb_table.columns.name = "Perspectiva"
        lcb_table.index.name = "draft for lcb"
        
        return {
            'area' : dai,
            'centroid' : dci,
            'volum' : serie_volum,
            'lcb' : lcb_table
            }


## startscript
"""
boat = Ship(path = 'load.txt')

"""
## endscritps

if __name__ == '__main__':

    boat = Ship()
    import os
    # print('path'.center(80,'-'))
    ship_drafts, ship_distances = boat.load_table(path = 'table_example.txt')
    boat.attr(ship_drafts)
  
    
    
    
    
    
    
    