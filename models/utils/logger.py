import logging
import datetime

class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey        = '\x1b[38;21m'
    blue        = '\x1b[38;5;39m'
    yellow      = '\x1b[38;5;226m'
    red         = '\x1b[38;5;196m'
    bold_red    = '\x1b[31;1m'
    green       = '\u001b[32m'
    magenta     = '\u001b[35m'
    cyan        = '\u001b[36m'
    reset       = '\x1b[0m'


    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.cyan + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Define format for logs
fmt = "%(asctime)s | %(filename)s | %(levelname)s | %(message)s" #[line:%(lineno)d]"

# Create stdout handler for logging to the console (logs all five levels)
today = datetime.date.today()
logging.basicConfig(
    level=logging.DEBUG,
    filename='Image_restorer_{}.log'.format(today.strftime('%Y_%m_%d')),
    filemode='a',
    format=fmt,
    force=True
)

stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(CustomFormatter(fmt))
logging.getLogger().addHandler(stdout_handler)