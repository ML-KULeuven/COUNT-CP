import json
import numpy as np
import cpmpy
import glob
import csv
import learner
import pickle
import logging
import time
from multiprocessing import Pool
import argparse

from instance import Instance
from learn import learn, create_model, learn_propositional
import sys
from instances.type01 import model_type01
from instances.type02 import model_type02
from instances.type03 import model_type03
from instances.type04 import model_type04
from instances.type05 import model_type05
from instances.type06 import model_type06
from instances.type07 import model_type07
from instances.type08 import model_type08
from instances.type10 import model_type10
from instances.type11 import model_type11
from instances.type12 import model_type12
from instances.type13 import model_type13
from instances.type14 import model_type14
from instances.type15 import model_type15
from instances.type16 import model_type16
from instances.type20 import model_type20
from instances.type21 import model_type21
from instances.nurse_rostering import nurse_rostering_model

logger = logging.getLogger(__name__)



def true_model(t, instance):
    if t == 1:
        return model_type01(instance)
    elif t == 2:
        return model_type02(instance)
    elif t == 3:
        return model_type03(instance)
    elif t == 4:
        return model_type04(instance)
    elif t == 5:
        return model_type05(instance)
    elif t == 6:
        return model_type06(instance)
    elif t == 7:
        return model_type07(instance)
    elif t == 8:
        return model_type08(instance)
    elif t == 10:
        return model_type10(instance)
    elif t == 11:
        return model_type11(instance)
    elif t == 12:
        return model_type12(instance)
    elif t == 13:
        return model_type13(instance)
    elif t == 14:
        return model_type14(instance)
    elif t == 15:
        return model_type15(instance)
    elif t == 16:
        return model_type16(instance)
    elif t == 20:
        return model_type20(instance)
    elif t == 21:
        return model_type21(instance)
    elif t == 22:
        return nurse_rostering_model(instance)
