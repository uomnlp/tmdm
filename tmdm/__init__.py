# import sys
# from loguru import logger
# from dynaconf import settings
#
# log_level = settings.get('log_level', "INFO")
#
# logger.remove()
# logger.add(sys.stdout, level=log_level)
# logger.add("./logs/debug.log", level='DEBUG', rotation='50MB', compression="zip")
# logger.add("./logs/error.log", level='ERROR', rotation='50MB', compression="zip")
# logger.add("./logs/info.log", level='INFO', rotation='50MB', compression="zip")
# logger.debug('Set up logging.')
#
from tmdm.model.extensions import Annotation
from tmdm.model.coref import Coreference
from tmdm.model.oie import Verb, Argument
from tmdm.model.ne import NamedEntity
from tmdm.model.rc import Relation
from tmdm.main import tmdm_pipeline
