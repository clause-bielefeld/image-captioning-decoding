import pickle
import numpy as np
from nltk.corpus import stopwords
import string
import os
import sys
from tqdm.autonotebook import tqdm
import itertools
import argparse
import json
import pandas as pd


def get_coco_captions(path):
    """ get dataframe containing captions for coco train and val images """

    # load captions for train2014 images as dataframe
    with open(path+'annotations/captions_train2014.json') as file:
        captions = json.load(file)
        train_captions = pd.DataFrame(captions['annotations']).set_index('id')
        train_captions['coco_split'] = 'train'
    # load captions for val2014 images as dataframe
    with open(path+'annotations/captions_val2014.json') as file:
        captions = json.load(file)
        val_captions = pd.DataFrame(captions['annotations']).set_index('id')
        val_captions['coco_split'] = 'val'

    # merge train2014 and val2014 annotations into single dataframe
    captions = pd.concat([train_captions, val_captions])

    return captions


def coco_caption_splits(splits_path, captions_path, return_captions=True):
    """
        get image_ids and caption_ids for karpathy
        train, val, test, and restval splits
    """

    # set filepath and partition names
    filepath = os.path.join(splits_path, 'dataset_coco.json')
    partitions = ['test', 'restval', 'val', 'train']

    # return error and link to karpathy splits if file is not found
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            "File {path} doesn't exist".format(path=filepath)
            )
    # load karpathy splits as dataframe
    with open(filepath) as file:
        splits = json.load(file)
        splits = pd.DataFrame(splits['images'])
    # create dict with image_ids for each partition
    image_ids = {
        part: splits.loc[splits.split == part].cocoid.to_list()
        for part in partitions
    }
    # load coco captions, create dict with caption_ids for each partition
    captions = get_coco_captions(path=captions_path)
    caption_ids = {
        part: captions.loc[
                captions.image_id.isin(image_ids[part])
            ].index.to_list()
        for part in partitions
    }

    ids = {'image_ids': image_ids, 'caption_ids': caption_ids}

    if return_captions:
        return (captions, ids)

    return ids


def caps_to_set(caption):
    """ normalize captions, return as set of words """

    # lowercase, remove punctuation
    caption = caption.lower().translate(
        str.maketrans('', '', string.punctuation)
    )

    # whitespace + no stop words
    caption_words = [
        word for word in caption.split()
        if word not in stopwords.words('english')
    ]

    return set(caption_words)


def aggregate_caps_to_set(caps):
    """ aggregation function: aggregate captions, normalize, return as set """

    concat_caption = ' '.join(caps)

    return caps_to_set(concat_caption)


def caption_intersection(df):
    """
        return intersection between all captions

        :input: dataframe with caption column
        :output: intersection matrix, assignment of indices to image IDs
    """

    # get cartesian product of row/column number and entry id pairs
    entries = list(enumerate(df.index))
    cartesian_entries = list(itertools.product(entries, entries))

    # dict for assigning matrix column/row ids to image_ids
    idx2img = {i: index for i, index in entries}

    # initialize empty intersection matrix
    intersection_matrix = np.zeros((len(df), len(df)))

    # iterate through list of cartesian products
    # of row/column number and entry id
    for entry in tqdm(cartesian_entries):
        x, idx_x = entry[0]
        cap_x = df.caption[idx_x]

        y, idx_y = entry[1]
        cap_y = df.caption[idx_y]

        # image similarity: Jaccard coefficient of captions
        # (intersection of cap_x and cap_y / union of cap_x and cap_y)
        similarity = len(cap_x.intersection(cap_y)) / len(cap_x.union(cap_y))
        # assign overlap to intersection matrix
        intersection_matrix[x, y] = similarity

    return intersection_matrix, idx2img


def main(args):

    # Create model directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print('read data')

    # get captions and split ids
    captions, splits = coco_caption_splits(
        splits_path=args.splits_path, captions_path=args.coco_path
        )
    ids = splits['caption_ids'][args.split_name]
    data = captions.loc[ids]

    print('preprocess image captions')
    # concatenate all captions for each image, transform to set of words
    caps = data.groupby('image_id')\
        .agg({
            'caption': aggregate_caps_to_set,
            'coco_split': 'first'
        })

    # compute image clusters
    print('{name} set: compute intersection matrix'.format(
        name=args.split_name)
        )
    # compute intersection matrix,
    # get assignment of row/column indices to image IDs
    intersection_matrix, idx2img_id = caption_intersection(caps)

    # compute image clusters
    clusters = []
    images_in_cluster = args.images_per_cluster

    # iterate through intersection matrix rows
    print('{name} set: generate image clusters'.format(name=args.split_name))
    for i in tqdm(range(len(idx2img_id))):
        # get cluster of 10 most similar captions for each row

        target = idx2img_id[i]
        distractors = [
            idx2img_id[x] for x in np.argsort(
                -intersection_matrix[i]
            )[:images_in_cluster]
            ]
        if target in distractors:
            distractors.remove(target)
        cluster = [target] + distractors[:(images_in_cluster-1)]

        # append cluster to clusters list
        clusters.append(cluster)

    print('number of clusters:', len(clusters))

    # save image clusters to file
    file_path = os.path.join(
        args.out_dir,
        'image_clusters_{}_{}.pkl'.format(
                args.split_name, str(images_in_cluster)
            )
        )
    print('save image clusters to {path}'.format(path=file_path))
    pickle.dump(clusters, open(file_path, 'wb'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self',
                        help='To make it runnable in jupyter')
    parser.add_argument('--split_name', type=str,
                        default='val',
                        help='name of the split the clusters shall be created for'
                        )
    parser.add_argument('coco_path', type=str,
                        help='path to captions'
                        )
    parser.add_argument('splits_path', type=str,
                        help='path to karpathy split file')
    parser.add_argument('out_dir', type=str,
                        help='target directory'
                        )
    parser.add_argument('--images_per_cluster', type=int,
                        default=10, help='number of images in each cluster'
                        )

    args = parser.parse_args()

    main(args)
