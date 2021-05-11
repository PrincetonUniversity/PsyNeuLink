#!/bin/bash -l
# Make sure the above line is #!/bin/bash -l. If the -l is missing this script
# will not be executed as a login shell and /usr/local/bin will not be on the
# path. This will lead to issues with missing graphviz (dot) expected to be on
# the PATH by psyneulink.

cd $WORKSPACE
toxenvs=$(grep envlist tox.ini | sed "s;envlist = ;;")
USAGE="Usage: "$(basename "$0")" [$toxenvs]"

if [ $# -ne 1 ]; then
	echo "$USAGE"
	exit 0
fi

toxenv="$1"

export GIT_BRANCH_NO_SLASH="no_branch"
if [ $GIT_BRANCH ]
then
	GIT_BRANCH_NO_SLASH=`echo $GIT_BRANCH | sed 's;/;_;g'`
	echo 'On branch' $BRANCH
else
	echo 'No current branch'
fi

time tox -e "$toxenv"
