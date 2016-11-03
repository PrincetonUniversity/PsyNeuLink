#! /bin/bash
# you must run this script in the top level dir

# create the environment
conda create -n psyneulink_environment python=3.5

# activate the environment
source activate psyneulink_environment

# install all necessary dependencies
pip install .

# change dirs into scripts dir
cd Scripts

# run meta test script and save output in jenkins/testoutput
python META\ Test\ Script.py > ../jenkins/test_output.txt

# compare reference output to test output and save the exit code for later
diff jenkins/test_output.txt jenkins/reference_output.txt
DIFF_EXIT_CODE=$?

# deactivate the environment
source deactivate

conda remove --name psyneulink_environment --all

# return exit code of diff
if [ $DIFF_EXIT_CODE -eq 0 ]
	then
		exit 0
	else
		exit 1
fi