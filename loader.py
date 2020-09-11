
import numpy as np

def fromDelfShipTable(path):
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

    return load_table(path = path)