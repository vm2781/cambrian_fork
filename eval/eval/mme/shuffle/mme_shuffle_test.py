import argparse
import os
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import shortuuid
from copy import deepcopy


from datasets import load_dataset
from cambrian.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from cambrian.conversation import conv_templates, SeparatorStyle
from cambrian.model.builder import load_pretrained_model
from cambrian.utils import disable_torch_init
from cambrian.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math



if __name__ == "__main__":

    questions = load_dataset("lmms-lab/MME", split="test")
    no_lines = []
    yes_lines = []
    for line in tqdm(questions):

        if line["answer"] == "No":
            no_lines.append(line)
        elif line["answer"] == "Yes":
            yes_lines.append(line)
        else:
            print("We have outlier")
            exit()