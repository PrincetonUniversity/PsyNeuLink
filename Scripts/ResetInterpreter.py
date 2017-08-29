import os,sys
from xml.etree import ElementTree as et

# if pycharm can give us the real name somehow
# that would be better
myname = 'ResetInterpreter'
myrunconfig = myname + '.xml'

# what to set interpreter to (nothing)
emptyString = ""

# assumes Working Dir is PsyNeuLink top level dir
# (set in run configuration)
targetDir = os.path.join('.idea','runConfigurations')
# ensure it is found
if not os.path.exists(targetDir):
    print('\n *** Please set ' + myname + ' working directory to PsyNeuLink top level ***')
    exit(1)

# go through all the XML files in runConfigurations
# and set SDK_HOME (if found) to emptyString
for root, dirs, files in os.walk(targetDir):
    for file in files:
        if file.endswith('.xml'):
            # what file we're on
            print('\n' + file)
            if file == myrunconfig:
                print('skipping myself!')
                continue
            relFile = os.path.join(targetDir, file)
            # use ElementTree to set the value
            tree = et.parse(relFile)
            # SDK_HOME
            # xpath syntax
            sdk_home = tree.find('.//option[@name="SDK_HOME"]')
            # if we found an SDK_HOME line in the file
            foundSomething = False
            if sdk_home is not None:
                print('\nbefore:')
                print(sdk_home.attrib)
                sdk_home.set('value', emptyString)
                foundSomething = True
                print('==>\nafter')
                print(sdk_home.attrib)
            module = tree.find('.//module')
            if module is not None:
                foundSomething = True
                print('\nbefore:')
                print(module.attrib)
                module.set('name', emptyString)
                print('==>\nafter')
                print(module.attrib)
            if foundSomething:
                # rewrite the file in-place
                tree.write(relFile)
            else:
                print('target attributes not found')