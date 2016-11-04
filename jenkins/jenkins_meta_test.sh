#! /bin/bash
# you must run this script in the top level dir

# setting the path
PATH=/Users/psyneulink/anaconda3/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

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

# change dirs back up to top level dir
cd ..

# compare reference output to test output and save the exit code for later
diff jenkins/test_output.txt ~/reference_output.txt
DIFF_EXIT_CODE=$?

# deactivate the environment
source deactivate

# fully remove the environment
conda remove --name psyneulink_environment --all

# return exit code of diff
if [ $DIFF_EXIT_CODE -eq 0 ]
	then
		exit 0
	else
		exit 1
fi