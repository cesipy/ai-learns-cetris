import datetime

FILE_NAME = "../logs/py_log_"

class SimpleLogger:
    def __init__(self ):
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.filename = FILE_NAME + today + ".txt"

    def log(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp}- {message}\n"

        with open(self.filename, "a") as f:
            f.write(log_entry)


