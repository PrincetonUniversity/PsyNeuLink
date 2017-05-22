
# Create a generator that simply returns the next mechanism from a list

class mechanismGenerator(object):

    # generator takes in a list of mechanisms
    def __init__(self, mechList):

        self.mechList = mechList

    # generator yields one mechanism from the list on each call
    def yield_mech(self):

        for i in self.mechList:
            yield(i)

# Test the generator
# for mech in mechanismGenerator([1,2,3,4,5]).yieldMech():
#     print(mech)


