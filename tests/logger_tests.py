import time
import os, sys
import pytest
import datetime

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



def test_log_message_is_written():
    logger = SimpleLogger()

    message = "Test log message"
    logger.log(message)

    with open(logger.filename, "r") as f:
        contents = f.read()
        assert message in contents


def test_log_format():
    logger = SimpleLogger()

    message = "Test log message"
    logger.log(message)

    expected_entry = f"{datetime.datetime.now().strftime('%H:%M:%S')}- {message}\n"
    with open(logger.filename, "r") as f:
        contents = f.read()
        assert expected_entry in contents


@pytest.fixture(autouse=True)
def cleanup_logs(request):
    yield
    # remove the log files created during the tests
    for filename in os.listdir("../logs/"):
        if filename.startswith("py_log_"):
            os.remove(os.path.join("../logs/", filename))