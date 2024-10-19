#!/usr/bin/python

import os

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("."):
    for dirname in dirs:
        print(dirname)
        os.system(f"python eosfit.py {dirname}/{dirname.split('-')[1]}.txt")
        # print(dirname)
    break
        # doSomewthingWithDir(os.path.join(root, dirname))