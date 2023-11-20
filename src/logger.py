"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import logging
from typing import Union

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    #format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    debug_format = "%(name)s - (%(filename)s:%(lineno)d) - %(levelname)s: %(message)s"
    info_format = "%(name)s - %(levelname)s: %(message)s"
    warn_format = "%(name)s - %(levelname)s: %(message)s"
    error_format = "%(name)s - %(levelname)s: %(message)s"
    critical_format = "%(name)s -%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + debug_format + reset,
        logging.INFO: grey + info_format + reset,
        logging.WARNING: yellow + warn_format + reset,
        logging.ERROR: red + error_format + reset,
        logging.CRITICAL: bold_red + critical_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def configure_logger(name : str, verbosity : Union[int,str]):
    # Create a custom logger
    logger = logging.getLogger(name)
    verbosity_map = {
        10: logging.DEBUG,
        20: logging.INFO,
        30: logging.WARNING,
        40: logging.ERROR,
        50: logging.CRITICAL,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{name}.log')
        c_handler.setLevel(verbosity_map[verbosity])
        f_handler.setLevel(verbosity_map[verbosity])
        # Create formatters and add it to handlers
        c_handler.setFormatter(CustomFormatter())
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s'))
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        # Set the verbolity
        logger.setLevel(verbosity_map[verbosity])
    return logger