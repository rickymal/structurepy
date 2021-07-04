from operator import pos
import pandas as pd
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import geometry
from math import atan2, sin, cos, sqrt, pi, degrees
from deprecation import deprecated
from collections import namedtuple
from numpy import nan
from numpy import interp
import pdb

class TransversalSectionComposer:
    """
    Classe que representa uma lista, ou conjunto de poligonos 2D redistribuidos em seções em um terceiro plano
    """

    def setAttribute(self,name,attributeData):
        if hasattr(self,"primitive_geometry_attributes"):
            self.primitive_geometry_attributes[name] = attributeData
        else:
            raise Exception("[Internal error] attribute 'primitive_geometry_attributes' not found")

    @classmethod
    def isPolygon(cls,polygon):
        c1 = len(polygon) > 3
        c2 = np.equal(polygon[0],polygon[-1])
        return c2

    def __init__(self):
        self.representations = {}
        self.perspectives = {}
        self.primitive_geometry_attributes = {}
        self.attr = {}

    def _basify(self,numeric_value):
        print(" Método basify ainda não implementado")
        return numeric_value
        pass

    def _load_attributes(self):
        return self.primitive_geometry_attributes

    def get_axis_values_unified(self, axis : str or int = 'y'):
        if 'numeric' not in self.representations:
            raise Exception("The representations wasn't found.")
        if isinstance(axis,str):
            if axis == 'y':
                axis = 1
            elif axis == 'x':
                axis = 0
            else:
                raise Exception("The parameter axis inserted was not recognized. The value was: " + axis)
        else:
            if axis != 0 or axis != 1:
                raise Exception("The parameter axis inserted was not recognized. The value was: " + axis)

        axis_values_disposable = np.concatenate(self.representations['numeric'], axis = 0)[:,axis]
        axis_values_unique = np.unique(axis_values_disposable)
        return axis_values_unique
        

    def _create_superior_attributes(self, **kwargs):
        y_axis = self.get_axis_values_unified(axis = 'y')
        
        polygons = self.representations['numeric']
        attrs = dict(area = [], centroid = [], numeric = [])


        
        cut_superior = self._superior_cut(polygons,self.representations['distances'],cut_axis = None)
        print(cut_superior)
        pdb.set_trace()
        quit()
        

    def _create_transversal_attributes(self, **kwargs):
        distances = self.distances
        polygons = self.representations['numeric']

        attrs = dict(area = [],centroid = [],numeric = [])

        for i in range(len(polygons)):
            # calculo de área
            polygon = polygons[i]
            y_axis = np.sort(polygon[:,1])
            datas = [self._transversal_cut(polygon,cut_axis = y) for y in y_axis]

            # TEST
            #for y in y_axis:
            #    resultado = self._transversal_cut(polygon,cut_axis = y) 

            # END TEST
            data_area = [x['area'] for x in datas]
            data_centroid = [x['centroids'] for x in datas]
            data_numeric = [x['numeric'] for x in datas]

            serie_area = pd.Series(data = data_area,index = y_axis,dtype=np.float16)
            serie_centroid = pd.Series(data = data_centroid,index = y_axis, dtype = np.float16)
            #serie_numeric = pd.Series(data = data_numeric,index = y_axis,dtype = np.float16)

            attrs['area'].append(serie_area.drop_duplicates())
            attrs['centroid'].append(serie_centroid.drop_duplicates())
            #attrs['numeric'].append(serie_numeric)

        polygon_index = pd.Series(data = [i for i in distances],name = "Distances")

        dataframe_area = pd.concat(attrs['area'],axis = 1).interpolate(method = 'index',limit_area = 'inside')
        dataframe_area.columns = polygon_index

        dataframe_centroid = pd.concat(attrs['centroid'],axis = 1).interpolate(method = 'index',limit_area = 'inside')
        dataframe_centroid.columns = polygon_index

        dataframe_area = dataframe_area * 2

        dataframe_area.index = pd.Series(dataframe_area.index.values, name = "Distance at height (doubled)")
        dataframe_area.columns = pd.Series(dataframe_area.columns.values, name = "Distance in length")

        self.primitive_geometry_attributes['transversal area'] = dataframe_area
        self.primitive_geometry_attributes['transversal centroid'] = dataframe_centroid
        self.primitive_geometry_attributes['transversal numeric'] = attrs['numeric']

        

        # Calculo do volume
        """
        Existem alguns pontos a se analise no código do cálculo do volume. Eu retiro todos os dados inexistente de área os os deixo com valor zero?
        """
        if False:
            y_positions = dataframe_area.index
            lof_volums = list()
            for i in range(len(y_positions)):
                serie = dataframe_area.loc[y_positions[i],:]
                serie_dropped = serie.replace(np.nan,0.0)
                x = serie_dropped.index
                y = serie_dropped.values
                from scipy import integrate

                if 'integration_type' in kwargs:
                    integration_type = kwargs['integration_type']
                else:
                    integration_type = 'trapz'


                if integration_type == 'trapz':
                    volum = integrate.trapz(y = y,x = x)
                elif integration_type == 'simps':
                    volum = integrate.simps(y = y,x = x)

                lof_volums.append(volum)

            serie_volum = pd.Series(data = lof_volums,index = y_positions,name = "Volume deslocado")

            self.primitive_geometry_attributes["volum"] = serie_volum
        else:
            pass
        

    @classmethod
    def fromNumeric(cls,numeric_value, distances,perspective,**kwargs):
        new_instance = cls()
        import pdb

        # Obter a distancia
        


        # fim
        if 'basify' in kwargs and kwargs['basify']:
            numeric_value = new_instance._basify(numeric_value)

        new_instance.representations['numeric'] = numeric_value
        new_instance.perspective = perspective
        new_instance.distances = distances
        new_instance.representations['distances'] = distances


        if len(numeric_value[0][0,:]) == 3:
            new_instance.isMatrix = True
            raise NotImplementedError("Operations with 3D matrizes is not implemented yet")
        else:
            new_instance.isMatrix = False
        print("Criando uma nova instancia")


        new_instance._create_superior_attributes(**kwargs)
        new_instance._create_transversal_attributes(**kwargs)

        return new_instance
        pass

    #Composição de atributos

    def _attributes_transversal(self,):
        distances = self.distances
        polygons = self.representations['numeric']
        lof_n = {}
        for distance,polygon in zip(distances,polygons):
            lof_nn = {}
            for cut_y in polygon[:,1]:
                temp = self._transversal_cut(polygon = polygon,cut_axis = cut_y,axis_to_cut = 'horizontal')
                lof_nn[cut_y] = temp
            lof_n[distance] = lof_nn
        self.representations['transversal_view'] = lof_n
        

    def _attributes_superior(self,):
        distances = self.distances
        polygons = self.representations['numeric']
        y_values = [p[:,1] for p in polygons]
        y_values_unique = np.unique(np.concatenate(y_values))


        
        lof_n = {}
        for c in y_values_unique:
            r = self._superior_cut(polygons,distances = distances,
            cut_axis = c)
            lof_n[c] = r

        self.representations['superior_view'] = lof_n

    def _attributes_lateral(self,):
        raise NotImplementedError


    def _transversal_cut(self,polygon : np.ndarray ,cut_axis : float = 0.5  ,axis_to_cut : str ='horizontal'):
  
        # pdb.set_trace()
        _CUT_AXIS = cut_axis
        x_ = polygon[:,0]
        y_ = polygon[:,1]
        if _CUT_AXIS is None:
            _CUT_AXIS = max(polygon[:,1]) #it won't make cut

        suposted_polygon = polygon 
        # the name is "suposted_polygon" because it not known yet if this matrix is closed

        # The matrix is closed? If not, this piece of code will extends the line to 0 coordinate at 'x'
        if suposted_polygon[-1,0] != 0.0:
            suposted_polygon = np.append(suposted_polygon,
                                        [0,suposted_polygon[-1,1]]).reshape(-1,2).astype(np.float16)

        the_real_polygon = suposted_polygon
        ring_polygon = geometry.LinearRing(the_real_polygon)
        polygon_object = geometry.Polygon(ring_polygon)
        if axis_to_cut == 'horizontal':
            # cut_line = geometry.LineString([
            #     [-1.,_CUT_AXIS],
            #     [max(the_real_polygon[:,0]) + 1.,_CUT_AXIS],
            #     ])

            the_most_distance_x = the_real_polygon[:,0].max() + 1 
            # the + 1 is margin of security

            cut_line = geometry.LineString([
                (-1,_CUT_AXIS),
                (the_most_distance_x, _CUT_AXIS),
            ])
        elif axis_to_cut == 'vertical':
            pass
            cut_line = geometry.LineString([
                [_CUT_AXIS,-1.],
                [_CUT_AXIS,max(the_real_polygon[:,1])+1.]                
                ])

        else:
            raise Exception("The axis_to_cut parameter need to be 'vertical' or 'horizontal' ")
        
        from shapely import ops 
        data = ops.split(polygon_object,cut_line)
        i = 0
        if len(data) == 2:
            pass
            _,y1 = data[0].exterior.coords.xy
            _,y2 = data[1].exterior.coords.xy
            min_y1 = min(y1)
            min_y2 = min(y2)

            if min_y1 > min_y2:
                i = 1
            else:
                i = 0
        else:
            pass

        x,y = data[i].exterior.coords.xy
        matrix_cut_representation = np.stack([x,y],axis = 1)
        x,y = matrix_cut_representation[:,0],matrix_cut_representation[:,1]

        centroids = data[i].centroid.coords[0]
        
        """
        Se a região de corte estiver no ponto mínimo, o valor é posto por inteiro, logo deve-se fazer a correção
        """
        if not _CUT_AXIS == min(polygon[:,1]):
            centroid_result = centroids[1] # o indice '0' é para o eixo 'horizontal' e o indice '1' para o eixo 'vertical'
            area_result = data[i].area
        else:
            centroid_result = 0.0
            area_result = 0.0

        mm = matrix_cut_representation

        # find the initial point, needed to be applied in transversal_cut method
        mask = np.equal(mm,[0,0],).all(axis = 1)
        index_mask = int(np.where(mask)[0][0]) 
        # convert ndarray int 64 for int python type
        
        first_part = mm[index_mask:,:].copy()
        last_part = mm[:index_mask,:].copy()
        if not np.equal(last_part[-1,:],[0,0]).all():
            last_part = np.vstack([last_part,[0,0]])
        new_mm = np.vstack([first_part,last_part])

        return {
            'numeric' : matrix_cut_representation,
            'centroids' : centroid_result,
            'area' : area_result,
            'cut' : cut_axis,
            'numeric rotated' : new_mm
            }



    def extract_transversal_view_to_superior_view(self, list_of_pair_distance_and_max_value_of_x, distance, temp, position_of_slice = None):
        matrix_representation_from_polygon_cutted = temp['numeric']
        mask = matrix_representation_from_polygon_cutted[:,1] == matrix_representation_from_polygon_cutted[:,1].max()

        if position_of_slice == 'begin' or position_of_slice == 'end':
            value_c = matrix_representation_from_polygon_cutted[mask][:,0].min()
            
        else:
            value_c = matrix_representation_from_polygon_cutted[mask][:,0].max()

        value_c = np.unique(matrix_representation_from_polygon_cutted[mask], axis = 1)[:,0].max()
        list_of_pair_distance_and_max_value_of_x.append([distance,value_c]) 

    def _superior_cut(self,sections : np.ndarray ,distances : np.ndarray ,cut_axis : float or int):    
        import pdb
        list_of_pair_distance_and_max_value_of_x = []
        _CUT_AXIS = cut_axis    
        for ind, (section,distance) in enumerate(zip(sections,distances)):
            temp = self._transversal_cut(section,cut_axis = _CUT_AXIS)
            # max_value_x = temp['numeric'][:,0].max()
            
            if ind == 0:
                pass

            # this block of code extract the view 
            self.extract_transversal_view_to_superior_view(list_of_pair_distance_and_max_value_of_x, distance, temp)

            if ind == len(distances):
                pass
            
        pdb.set_trace()
        numeric_lof = np.array(list_of_pair_distance_and_max_value_of_x)
        
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
        sections = listOf_sections
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




    # Métodos para o cálculo de momento de inércia


    def area(self,pts):
        

        if np.equal(pts[0],pts[-1]).all():
            pts = pts + pts[:1]
        x = [ c[0] for c in pts ]
        y = [ c[1] for c in pts ]
        s = 0
        for i in range(len(pts) - 1):
            s += x[i]*y[i+1] - x[i+1]*y[i]
        return s/2


    def centroid(self,pts):
        'Localização do centroid.'
        
        if np.equal(pts[0],pts[-1]).all():
            pts = pts + pts[:1]
        x = [ c[0] for c in pts ]
        y = [ c[1] for c in pts ]
        sx = sy = 0
        a = self.area(pts)
        for i in range(len(pts) - 1):
            sx += (x[i] + x[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
            sy += (y[i] + y[i+1])*(x[i]*y[i+1] - x[i+1]*y[i])
        return sx/(6*a), sy/(6*a)


    def inertia(self,pts):
        'Momento e o produto de inércia.'
        
        if np.equal(pts[0],pts[-1]).all():
            pts = pts + pts[:1]
        x = [ c[0] for c in pts ]
        y = [ c[1] for c in pts ]
        sxx = syy = sxy = 0
        a = self.area(pts)
        cx, cy = self.centroid(pts)
        for i in range(len(pts) - 1):
            sxx += (y[i]**2 + y[i]*y[i+1] + y[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
            syy += (x[i]**2 + x[i]*x[i+1] + x[i+1]**2)*(x[i]*y[i+1] - x[i+1]*y[i])
            sxy += (x[i]*y[i+1] + 2*x[i]*y[i] + 2*x[i+1]*y[i+1] + x[i+1]*y[i])*(x[i]*y[i+1] - x[i+1]*y[i])
        return sxx/12 - a*cy**2, syy/12 - a*cx**2, sxy/24 - a*cx*cy





    #Método depreciados 



        
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



