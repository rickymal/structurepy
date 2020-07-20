from Geometry import TransversalSectionComposer as Gcomp
import loader

data = loader.fromDelfShipTable(path = 'table.txt')
geometry = Gcomp.fromNumeric(numeric_value = data[0], distances = data[1],perspective = 'transversal')
print("Iniciando testes".center(80,'-'))
print(geometry.primitive_geometry_attributes['area'])
print("Fim da execução da aplicação".center(80,'-'))

