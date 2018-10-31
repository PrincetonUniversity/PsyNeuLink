from psyneulink import *

dim_1 = 5
dim_2 = 5

p = TransferMechanism(size=[dim_1, dim_2], function=Gaussian)
m = ProcessingMechanism()

print(p.execute([[1,0,1,0,0],[1,0,0,0,0]]))

assert True