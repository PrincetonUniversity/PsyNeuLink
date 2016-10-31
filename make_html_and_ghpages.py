#!/usr/bin/env python
# Script that will take results of Sphinx's make html
# and update github's gh-pages branch.
# Is really a bash script, but want to be able to
# leverage python environments and PyCharm configurations

# load input gDirs with command-line character arg and default value
gDirs = [
    ('c','docs','docs'),
    ('s','source','docs/source'),
    ('b','build','docs/build'),
    ('t','html','docs/build/html')
]
# positions for later indexing
Dir_docs = 0
Dir_source = 1
Dir_build = 2
Dir_html = 3
Tup_flag = 0
Tup_name = 1
Tup_path = 2

gShortOpts = ''.join([x[Tup_flag] for x in gDirs])
gLongOpts = [x[Tup_name] for x in gDirs]
# single-char command line opts (reserve h for help and d for debug)
# if debug ('-d') arg passed in,  will print informational messages
def debug_print(x): print(x)
def nodbg_print(x): pass
gDebug = False
dprint = nodbg_print

import sys

def usage():
    print("\nusage: " + sys.argv[0] + " [options]\n\noptions:\n")
    print("\t-h|--help\t\tprint this help")
    print("\t-d|--debug\t\tdebug (print info messages)")
    for idx,name in enumerate(gLongOpts):
       print('\t-' + gShortOpts[idx] + "|" + name + "==path\t\tlocation of " + name + " directory")
    print("\n")

def get_args(argv):
    import sys,getopt
    global gDebug, dprint

    keylist_with_equals = [x + '=' for x in gLongOpts]
    try:
       opts, args = getopt.getopt(argv, "hd" + gShortOpts, ["help","debug"] + keylist_with_equals)
    except getopt.GetoptError:
       usage()
       sys.exit(1)
    for opt, arg in opts:
       # deal with help and debug
       if opt in ("-h", "--help"):
           usage()
           sys.exit()
       if opt in ("-d", "--debug"):
           gDebug = True

       # custom settings
       for idx in range(len(gDirs)):
           if opt in ('-' + gShortOpts[idx], '--' + gLongOpts[idx]):
              gDirs[gLongOpts[idx]] = arg
              break

    if gDebug:
       dprint=debug_print
    else:
       dprint=nodbg_print

    dprint(gDirs)

def run_command(command, printonly=False, exitOnError=True):
    from subprocess import PIPE, run

    dprint(' '.join(command))
    if printonly:
       return

    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True)
    if 0 != result.returncode:
       print('Error running ' + ' '.join(command))
       print(result.stdout)
       print(result.stderr)
       if exitOnError: sys.exit(3)
    else:
       return result.stdout.rstrip()

def get_url_updated():
    url = run_command(['git', 'remote', 'get-url', 'origin'])
    username = url.split('/')[3]
    project = url.split('/')[4].split('.')[0]
    return "https://" + username + ".github.io/" + project

def main(argv):
    get_args(argv)

    import os,glob

    print("Detecting current branch ...")
    current_branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    print("... " + current_branch)
    for tups in gDirs:
       if not os.path.isdir(tups[Tup_path]):
           print(tups[Tup_name] + " directory " + tups[Tup_path] + " not found!")
           sys.exit(2)

    task = "Building and pushing docs"
    print(task)
    
    try:
       from sphinx import cmdline
    except:
       raise NameError("Cannot find sphinx in selected interpreter.")

    cmdline.main(["sphinx-build", "-b", "html", "-aE", gDirs[Dir_source][Tup_path], gDirs[Dir_build][Tup_path]])

    run_command(["git", "add", "-A"])
    # Don't exit on error because if there's nothing to commit
    # (all up to date) an error is returned, but want to go to next step
    run_command(["git", "commit", "-m", task],exitOnError=False)
    run_command(['git', 'push'])

    task = 'Publishing updated docs'
    print(task)
    run_command(['git', 'checkout', 'gh-pages'])
    run_command(['rm', '-rf'] + glob.glob('./*'))
    dprint(' '.join(os.listdir()))
    run_command(['touch', '.nojekyll'],exitOnError=False)
    run_command(['git', 'checkout', current_branch, gDirs[Dir_html][Tup_path]])
    run_command(['rsync', '-av', './' + gDirs[Dir_html][Tup_path] + '/', '.'])
    run_command(['rm', '-rf', gDirs[Dir_docs][Tup_path]])
    run_command(['git', 'add', '-A'])
    # an error here, if all up to date, would quit before
    # switching back to original branch, so allow continuing
    run_command(['git', 'commit', '-m', task],exitOnError=False)
    run_command(['git', 'push'])

    print('Switching back to ' + current_branch + ' branch')
    run_command(['git', 'checkout', current_branch])

    print("Your pages are published at " + get_url_updated())

if __name__ == "__main__":
    main(sys.argv[1:])
