import argparse
import random
import numpy as np
import sys
import torch

import properties_loader

DIR="/home/qingyu/data/kevin_all/"
#parse arguments
parser = argparse.ArgumentParser(description="Experiemts for Coreference Resolution (by qyyin)\n")

parser.add_argument("-embedding_dir",default = DIR+"features/mention_data/word_vectors.npy", type=str, help="specify dir for embedding file")
parser.add_argument("-DIR",default = DIR, type=str, help="Home direction")
parser.add_argument("-DOCUMENT",default = DIR+"documents/", type=str, help="Document direction")
parser.add_argument("-language",default = "en", type=str, help="language")
parser.add_argument("-gpu",default = 3, type=int, help="GPU number")
parser.add_argument("-reduced",default = 0, type=int, help="GPU number")
parser.add_argument("-random_seed",default = 0, type=int, help="Random Seed")
parser.add_argument("-props",default = "./properties/probs.en", type=str, help="properties")


args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)

nnargs = properties_loader.read_pros(args.props)
for item in nnargs.items():
    print >> sys.stderr, item
