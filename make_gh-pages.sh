#!/usr/bin/env bash
# taken from 
# http://www.willmcginnis.com/2016/02/29/automating-documentation-workflow-with-sphinx-and-github-pages/
# adapted by Nate Wilson
# you must run this in /PsyNeuLink top level project directory

# get on the right branch
git checkout devel

# build the docs
cd docs
make clean
make html
cd ..

# commit and push
git add -A
git commit -m "building and pushing docs"
# note in our case we are pushing to devel
# instead of the more standard master
git push origin devel

# switch branches and pull the data we want
git checkout gh-pages
rm -rf .
touch .nojekyll
git checkout devel docs/build/html
mv ./docs/build/html/* ./
rm -rf ./docs
git add -A
git commit -m "publishing updated docs..."
git push origin gh-pages

# switch back
git checkout devel