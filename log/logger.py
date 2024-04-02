import logging


class Logger(object):

    def __init__(self, path):
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        
        # create logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', "%Y-%m-%d %H:%M:%S")

        if not self.logger.handlers:
            # create file handler
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)

            # use StreamHandler for print
            print_handler = logging.StreamHandler()
            print_handler.setLevel(logging.DEBUG)
            print_handler.setFormatter(formatter)

            # add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(print_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)


# example implementation
if __name__ == "__main__":
    
    # TEST LOGGER
    logger = Logger('test.log')
    
    logger.info("info")
    logger.warning("warning")
    logger.error(" error")
    logger.critical("critical")
    logger.debug("debug")
