from Geometry import TransversalSectionComposer
import loader
import matplotlib.pyplot as plt

class ShipArchitecture(TransversalSectionComposer):

    def __init__(self):
        super().__init__()
        self.ShipAttributes = {}

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
    data = loader.fromDelfShipTable(path = 'table.txt')
    geometry = ShipArchitecture.fromNumeric(numeric_value = data[0], distances = data[1],perspective = 'transversal')
    geometry.primitive_geometry_attributes['area'].plot()
    plt.show()
    geometry.primitive_geometry_attributes["volum"].plot()
    plt.show()