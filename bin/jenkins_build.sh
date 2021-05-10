#!/bin/bash
# you must run this script in the top level dir
source ~/.bashrc

while getopts ":e:" opt; do
	case $opt in
		e) env_name="$OPTARG"
		;;
		\?) echo "Invalid argument -$OPTARG" >&2
		;;
	esac
done

if [ -z $env_name ]; then
	echo "No python environment specified, set with -e" >&2
	exit 1
fi

PNL_DIR=$WORKSPACE
PYTEST_TEST_DIR=tests/

JUNIT_DIR=$PNL_DIR'/jenkins/junit-reports'
JUNIT_BACKUP_DIR=~/jenkins/junit-reports/$JOB_BASE_NAME

echo 'Copying junit reports to workspace from' $JUNIT_BACKUP_DIR
mkdir -p $JUNIT_DIR
cp -rp $JUNIT_BACKUP_DIR/*.xml $JUNIT_DIR/

cd $PNL_DIR

# echo 'Cleaning Working dir '$PNL_DIR
# git clean -xdf -e jenkins/junit-reports
# git remote prune origin

BRANCH='no_branch'
if [ $GIT_BRANCH ]; then
        BRANCH=$(echo $GIT_BRANCH | sed 's;/;_;g')
        echo 'On branch' $BRANCH
else
        echo 'No current branch'
fi

BUILD='no-build'

if [ $BUILD_NUMBER ]; then
        BUILD=$BUILD_NUMBER
        echo 'Running build' $BUILD
else
        echo 'No current build'
fi


echo "Building branch $BRANCH"

echo "Activating PsyNeuLink environment ($env_name)..."
# activate the environment
# source activate $env_name
pyenv activate $env_name

# If multiple jobs on this machine are running,
# the environment may fail to change due to a bug
# in pyenv
while [ $? -eq 1 ]; do
	sleep 1
	pyenv activate $env_name
done

echo 'Reinstalling environment...'
# install all necessary dependencies

# removing this below as the environments on this machine are messed up
# on other machines fresh installs work fine but..not here
# pip freeze | xargs pip uninstall -y

pip install --no-cache-dir --upgrade $PNL_DIR[dev]

if [ $? -eq 1 ]; then
    echo 'pip install failed'
    exit 1
fi

echo 'Running pytest on '$PYTEST_TEST_DIR
#gtimeout --foreground -k 6 10m python -m pytest --junit-xml=$JUNIT_DIR/$BRANCH-$BUILD.xml $PYTEST_TEST_DIR
python -m pytest -p no:logging --junit-xml=$JUNIT_DIR/$BRANCH-$BUILD.xml $PYTEST_TEST_DIR

PYTEST_EXIT_CODE=$?

if [ $PYTEST_EXIT_CODE -eq 124 ]; then
	echo
	echo 'pytest timed out'
fi

echo 'Backing up junit reports to' $JUNIT_BACKUP_DIR
cp -rp $JUNIT_DIR/*.xml $JUNIT_BACKUP_DIR/

echo 'Deactivating environment...'
# deactivate the environment
source deactivate

exit $PYTEST_EXIT_CODE

