import time
import os, sys

# to import packages from src
# https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
testdir = os.path.dirname(__file__)
srcdir  = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from simpleLogger import SimpleLogger



def logger_test(n: int): 

    logger = SimpleLogger()

    for i in range(n): 
        logger.log(f"{i}'s log message")


logger_test(100)