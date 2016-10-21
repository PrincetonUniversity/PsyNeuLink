#! /bin/bash

# replace this echo statement with a python test script
echo 'hello' > jenkins/test_output.txt

# compare reference output to test output
diff jenkins/test_output.txt jenkins/reference_output.txt

# return exit code of diff
if [ $? -eq 0 ]
	then
		exit 0
	else
		exit 1
fi