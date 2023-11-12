import datetime


class SimpleLogger:
    def __init__(self ):
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        self.filename = "../logs/py_log_" + today

    def log(self, message: str):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"{timestamp} {message}\n"

        with open(self.filename, "a") as f:
            f.write(log_entry)


