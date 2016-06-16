# import matlab.engine
#test file
#get matlab handle
print("importing matlab...")
import matlab.engine
eng1=matlab.engine.start_matlab('-nojvm')
# eng1=matlab.engine.start_matlab()
print("matlab imported")

#run matlab function and print output
t=eng1.gcd(100.0, 80.0, nargout=3)
print(t)
#end

hello_world = 1
if hello_world:
    print("Hello World!")

exit()

