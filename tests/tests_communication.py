import time
import os, sys
import pytest

# to import packages from src
# https://stackoverflow.com/questions/1896918/running-unittest-with-typical-test-directory-structure
testdir = os.path.dirname(__file__)
srcdir  = "../src"
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from simpleLogger import SimpleLogger
from communication import Communicator
from metadata import Metadata

@pytest.fixture
def logger(tmp_path):
    # Using tmp_path fixture to create a temporary directory for testing
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    return SimpleLogger()

@pytest.fixture
def metadata(tmp_path, logger):
    fifo_states_name = str(tmp_path / "fifo_states")
    fifo_controls_name = str(tmp_path / "fifo_controls")

    # Create FIFO files
    os.mkfifo(fifo_states_name)
    os.mkfifo(fifo_controls_name)

    fd_states = os.open(fifo_states_name, os.O_RDWR | os.O_NONBLOCK)
    fd_controls = os.open(fifo_controls_name, os.O_RDWR | os.O_NONBLOCK)

    return Metadata(logger, fifo_states_name, fifo_controls_name, fd_states, fd_controls)

@pytest.fixture
def communicator(metadata):
    return Communicator(metadata)


def test_receive_from_pipe(communicator, tmp_path):
    message = "Test message"
    fifo_states_name = str(tmp_path / "fifo_states")
    fd_states = os.open(fifo_states_name, os.O_RDWR | os.O_NONBLOCK)

    # write a message 
    os.write(fd_states, message.encode('utf-8'))

    # read from the pipe and check if the received message is correct
    received_message = communicator.receive_from_pipe()
    assert received_message == message


#cleanup
@pytest.fixture(autouse=True)
def cleanup_pipes(request, tmp_path):
    yield
    # Remove the FIFO files created during the tests
    for filename in tmp_path.iterdir():
        if filename.is_fifo():
            filename.unlink()