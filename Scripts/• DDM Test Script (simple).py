from Functions.Process import Process_Base
from Functions.Mechanisms.DDM import *
from Globals.Keywords import *

simple_ddm = Process_Base(params={kwConfiguration:[DDM]})
simple_ddm.execute()
