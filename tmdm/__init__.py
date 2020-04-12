import sys
from loguru import logger
from dynaconf import settings

logger.remove()
logger.add(sys.stdout, level=settings['log_level'])
logger.add("./logs/debug.log", level='DEBUG', rotation='50MB', compression="zip")
logger.add("./logs/error.log", level='ERROR', rotation='50MB', compression="zip")
logger.add("./logs/info.log", level='INFO', rotation='50MB', compression="zip")
logger.debug('Set up logging.')
