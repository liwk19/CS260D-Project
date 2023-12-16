from os.path import abspath, dirname, exists
import os
import shutil

def get_cur_dir():
    return dirname(abspath(__file__))


for dir_ in os.walk(get_cur_dir()):
    if 'work_dir' in dir_[0] and exists(dir_[0]):
        print(dir_[0])
        # shutil.rmtree(dir_[0])
        os.system('rm -fr "%s"' % dir_[0])
