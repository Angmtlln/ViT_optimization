import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from glob import glob
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
import copy

# Пути к обученным моделям (веса после дистилляции)
MODEL_WEIGHTS_DIR = Path('../../model_weights')

for i in MODEL_WEIGHTS_DIR:
    print(i)