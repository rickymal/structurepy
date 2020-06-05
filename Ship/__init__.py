import Geometry 

print("WORKING", Geometry.Geometry)

## Como eu quero escrever no __name__
boat = Ship(path = 'load.txt')

boat.area(draft = 2.3,)
boat.volum(draft = 2.3)
boat.centroid(draft = 2.4)
boat.attrs.lcf(draft = 3.5)
boat.area




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
    geometry = Geometry()    
    ship_drafts, ship_distances = geometry.load_table(path = 'table_example.txt')
    geometry.basify(ship_drafts) #inline
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
            area, centroid = geometry.attributes(x)
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
  
    
    
    
    
    
    
    