#!/usr/bin/env python
# Script that will take results of Sphinx's make html
# and update github's gh-pages branch.
# Is really a bash script, but want to be able to
# leverage python environments and PyCharm configurations

import os
import sys
import atexit
import argparse
import glob
from subprocess import PIPE, run, CalledProcessError

gDefaultBranch = 'devel'  # default branch name that holds up to date docs subfolder
gDefaultOrg = 'princetonuniversity'  # really princetonuniversity but don't want to overwrite it


def get_current_branch():
    cb = run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    return cb


# whether by design or due to error, revert to
#  original branch when interpreter closes
def restore_original_branch(branch=""):
    if "" == branch:
        branch = gOrigBranch
    run_command(['git', 'checkout', branch])


def fatal(code):
    print('received fatal error, internal code ' + str(code))
    sys.exit(code)


def get_args(argv):
    parser = argparse.ArgumentParser(description="Use sphinx to build documents and update github gh-pages website")
    parser.add_argument('-b', '--branch', default=gDefaultBranch, help='branch to build docs in')
    parser.add_argument('-f', '--orgpush', action='store_true', help='update upstream organization site?')
    parser.add_argument('-r', '--org', default=gDefaultOrg, help='name of upstream organization')
    parser.add_argument('-d', '--docs', default='docs', help='top-level docs directory')
    parser.add_argument('-s', '--source', default='docs/source', help='source directory')
    parser.add_argument('-u', '--build', default='docs/build', help='build directory')
    parser.add_argument('-t', '--html', default='docs/build/html', help='html directory')

    args = parser.parse_args()
    print(args)
    return args


def run_command(command, printonly=False, exitOnError=True):
    print(' '.join(command))
    if printonly:
        return

    try:
        result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, check=True)
    except CalledProcessError as e:
        print('Error running ' + ' '.join(command))
        print(e.stdout)
        print(e.stderr)
        if exitOnError:
            fatal(2)
        else:
            print('continuing anyway')
    else:
        return result.stdout.rstrip()


def get_remote_url():
    #url = run_command(['git', 'remote', 'get-url', 'origin']) # too new for now (added Q4 2015 to git 2.7)
    url = run_command(['git', 'ls-remote', '--get-url', 'origin'])
    return url


def get_rootdir():
    rootdir = run_command(['git', 'rev-parse', '--show-toplevel'])
    return rootdir


def get_username(url=""):
    if "" == url:
        url = get_remote_url()
    username = url.split('/')[3]
    return username


def get_project(url=""):
    if "" == url:
        url = get_remote_url()
    project = url.split('/')[4].split('.')[0]
    return project


def get_url_updated(url=""):
    if "" == url:
        url = get_remote_url()
    username = get_username(url)
    project = get_project(url)
    return 'https://' + username + '.github.io/' + project


def get_temporary_dir():
    td = run_command(['mktemp', '-d'])
    atexit.register(run_command, ['rm', '-rf', td])
    return td


def sync_dirs(src, dst, delete=True, hidden=False):
    srcContents = src
    # rsync src/ contents, not src itself
    if not srcContents.endswith('/'): srcContents += '/'
    if delete:
        deleteStr = '--delete'
    else:
        deleteStr = ""

    if hidden:
        excludeStr = ""
    else:
        excludeStr = '--exclude=.*'

    run_command(['rsync', '-av', deleteStr, excludeStr, srcContents, dst])


def get_url_for_entity(entity):
    url = get_remote_url()
    project = get_project(url)
    entity_url = 'https://github.com/' + entity + '/' + project + '.git'
    return entity_url


def sync_ghpages(entity, html_dir):
    tmpdir = get_temporary_dir()
    atexit.register(run_command, command=['rm', '-rf', tmpdir], exitOnError=False)

    cwd = os.getcwd()
    try:
        task = 'copying ' + html_dir + ' to ' + entity + ' gh-pages'
        print(task)
        entity_url = get_url_for_entity(entity)
        run_command(['git', 'clone', '-b', 'gh-pages', '--single-branch', entity_url, tmpdir])
        sync_dirs(html_dir, tmpdir)
        os.chdir(tmpdir)
        run_command(['git', 'add', '-A'])
        run_command(['git', 'commit', '-a', '-m', task], exitOnError=False)
        run_command(['git', 'push', '-f'])
        os.chdir(cwd)
        print('pages built successfully, should be here within 5 - 60 secs:\n' + get_url_updated(entity_url))
    except:
        os.chdir(cwd)
        raise NameError('error trying to sync contents of ' + html_dir + ' to ' + tmpdir)


def main(argv):
    current_branch = get_current_branch()
    atexit.register(restore_original_branch, current_branch)

    args = get_args(argv)

    # ensure have committed any changes first
    print("If the following command fails you need to check in your changes first!")
    run_command(['git', 'status', '--porcelain'])

    # ensure are in sync
    print('Fetching current state of remote repository')
    run_command(['git', 'fetch'])

    # checkout designated branch
    print(
        'Checking out branch that has updated code/docs for web (' + args.branch + '; override with -r <branch_name>)')
    run_command(['git', 'checkout', args.branch])

    # check all required sphinx dirs
    for dir in [args.docs, args.source, args.build, args.html]:
        if not os.path.isdir(dir):
            print(' directory ' + dir + ' not found!')
            fatal()

    # run sphinx and build the html from the docs rst etc files, and push the update
    task = 'Building and pushing docs'
    print(task)
    try:
        from sphinx import cmdline
    except:
        raise NameError('Cannot find sphinx in selected interpreter.')
    # create the html pages from the rst docs in current branch
    # make clean -- sphinx-build will recreate the dirs
    print("removing any .doctrees toplevel dir, as well as all of the build, and html dirs")
    run_command(['rm', '-rf', '.doctrees'])
    run_command(['rm', '-rf', args.build])
    run_command(['rm', '-rf', args.html])
    # make html
    print("generating the HTML via sphinx-build...")
    cmdline.main(['sphinx-build', '-b', 'html', '-aE', args.source, args.html])
    # commit all changes to current branch + push changes
    print("adding and committing all changes to the docs")
    run_command(['git', 'add', '-A'])
    # Don't exit on error because if there's nothing to commit
    # (all up to date) an error is returned, but want to go to next step
    run_command(['git', 'commit', '-m', task], exitOnError=False)
    run_command(['git', 'push', 'origin', args.branch])

    # now checkout gh-pages, wipe it, put in all generated pages, and push
    task = 'Publishing updated docs'
    print(task)
    url = get_remote_url()
    username = get_username(url)
    entity = username

    try:
        sync_ghpages(entity, args.html)
        # can do same with upstream org if requested
        entity = args.org
        if args.orgpush: sync_ghpages(entity, args.html)
    except:
        print('error trying to push gh-pages for ' + entity)


if __name__ == "__main__":
    main(sys.argv[1:])
