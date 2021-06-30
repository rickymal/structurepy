from Geometry import TransversalSectionComposer
import loader
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import numpy as np

class ShipArchitecture(TransversalSectionComposer):

    def __init__(self):
        super().__init__()
        self.ShipAttributes = {}

    def get_volume(self,area, distances):
        tt = pd.DataFrame({
            'area' : area,
            'distances' : distances,
        })

        tt_dropped = tt.dropna()


        return integrate.trapz(y = tt_dropped.loc[:,'area'].values,x = tt_dropped.loc[:,'distances'].values)

    def volume(self):
        dataframe_area : pd.DataFrame = self.primitive_geometry_attributes['area']
        distances = dataframe_area.columns
        # dataframe_volume = dataframe_area.apply(lambda area : self.get_volume(area, distances), axis = 1)
        print("Content".center(80,'-'))
   
        l = 100
        volume_content = list()
        import pdb
        for _, area in dataframe_area.iterrows():
            # pdb.set_trace()
            print("content")
            if len(area.dropna()) == 1:
                print("O conteudo não possui valores suficientes para integração numérica")
                volume_content.append(np.nan)
                continue
            tt = pd.DataFrame({
            'area' : area,
            'distances' : distances,
            })
            tt_dropped = tt.dropna()
            nl = integrate.trapz(y = tt_dropped.loc[:,'area'].values,x = tt_dropped.loc[:,'distances'].values)
            volume_content.append(nl)
            
            
            pass
        data_result = pd.Series(volume_content, index = dataframe_area.index)
        data_result.name = "Volume (meters)"
        self.primitive_geometry_attributes['volume'] = data_result.copy()
        return data_result
    def awl(self):
        pass

    def Tcm(self):
        pass

    def Cwl(self):
        pass

    def Lcf(self):
        pass

    def Lcb(self):
        pass

    def kb(self):
        pass

    def Il(self):
        pass

    def It(self):
        pass

    def Bmt(self):
        pass

    def Bml(self):
        pass

    def displacement(self):
        pass

    def Cp(self):
        pass

    def Cb(self):
        pass

    def MTcm(self):
        pass

if __name__ == "__main__":
    data = loader.fromDelfShipTable(path = 'square.txt')
    geometry = ShipArchitecture.fromNumeric(numeric_value = data[0], distances = data[1],perspective = 'transversal')
    # geometry.primitive_geometry_attributes['area'].plot()
    # plt.show()
    print(geometry.primitive_geometry_attributes['area'].dropna())
    print(geometry.volume())

    # geometry.volume().plot()
    # plt.show()