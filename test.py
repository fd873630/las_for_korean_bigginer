
import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch

import numpy as np
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev
from main import *
from models import EncoderRNN, DecoderRNN, Seq2seq


import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq
torch.set_printoptions(profile="full")

a = [1,2,3]
b = [4,5,6]
c = [7,8,9]
d = [10,11,12]


a_list = list(a,b,c,d)
print(a_list[-1])