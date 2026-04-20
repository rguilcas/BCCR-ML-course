# This preprocessing has already been done for you in the data you've been provided.
# It filters out images which don't have three colour channels, and then splits the
# data into test, validation and training subsets.
# We do the split separately for each class to avoid class imbalance between the
# different subsets.
# We leave the image files as they are, but create metadata files which list which
# images are to be used in each subset.

import numpy as np
import os
from PIL import Image

np.random.seed(1)

from weather_helpers import filepath, metapath

os.makedirs(metapath, exist_ok=True)

classes = os.listdir(filepath)

# Find which images we want to use
# Remove those with too few or many channels
to_use = {}
for cls in classes:
    files = os.listdir(filepath+cls)
    to_use[cls] = []
    for ff in files:
        im = Image.open(filepath+cls+'/'+ff)
        im = np.array(im)
        shp = im.shape
        if (len(shp) < 3) or (shp[2] != 3):
            # Skip this image to make preprocessing easier
            continue
        else:
            to_use[cls].append(ff)


# Split the data into test, validation and training subsets
test_list, val_list, train_list = [], [], []
# For each class, split the data and add to the list
for cls in classes:
    N = len(to_use[cls])
    # 20% test, 20% val, 60% train
    test_boundary = N//5
    val_boundary = 2*test_boundary
    print(cls, N, test_boundary, val_boundary)
    indices = list(range(N))
    np.random.shuffle(indices)
    idx_test = indices[:test_boundary]
    idx_val = indices[test_boundary:val_boundary]
    idx_train = indices[val_boundary:]

    for subset_list, idx_list in [
            (test_list, idx_test), (val_list, idx_val), (train_list, idx_train)]:
        subset_list += [cls+'/'+to_use[cls][ii] for ii in idx_list]

# Save csv files listing which images make up which subsets
for ll, name in [(test_list, 'test'), (val_list, 'val'), (train_list, 'train')]:
    np.savetxt(metapath+name+'.csv', np.array(ll), fmt='%s')
