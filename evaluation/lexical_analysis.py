import nltk
import os
from os.path import join
import json
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
import re
import string
from collections import Counter
import numpy as np
import argparse


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.unknown_token = '<unk>'

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unknown_token]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def tokenize_caption(c):
    # clean from left <start> <end> <unk> <pad> tokens
    c = re.sub(r'(<start>)|(<end>)|(<unk>)|(<pad>)', '', c)
    c = c.casefold()
    c = c.translate(str.maketrans('', '', string.punctuation))
    return nltk.word_tokenize(c)


def tokenize_caps_list(captions):
    """
    tokenize list of captions
    """
    tokens = []
    for c in captions:  # same procedure as in build_vocab script
        tokens += tokenize_caption(c)
    return tokens


def tokenize_caps_file(file_dir):
    """
    get tokens from caption file dir
    """
    with open(file_dir, 'rb') as f:
        data = json.load(f)

    captions = [d['caption'] for d in data]

    tokens = tokenize_caps_list(captions)

    return tokens


def get_token_counts(token_list):
    """
    get number of occurences for each token in token_list
    """

    token_freq = Counter(token_list)
    srt = sorted(token_freq, key=token_freq.get, reverse=True)
    counts = np.array([token_freq[w] for w in srt])

    return counts


def vocab_diff(tokens_list, ref_tokens_list):
    """
    get tokens which occur in tokens_list
    but not in ref_tokens_list
    """
    return [v for v in tokens_list if v not in ref_tokens_list]


def avg_rank(cleaned_model_tokens, train_tokens):
    """
    get average rank from tokens in cleaned_model_tokens
    with respect to train_tokens
    """
    count = Counter(train_tokens)
    sorted_count = sorted(count, key=count.get, reverse=True)

    ranks = [sorted_count.index(token) for token in cleaned_model_tokens]
    avg_rank = sum(ranks) / len(ranks)

    return avg_rank


def main(args):

    # read files
    with open(join(args.coco_ann_dir, 'captions_val2014.json')) as f:
        val_ann_df = pd.DataFrame(json.load(f)['annotations'])
    with open(join(args.coco_ann_dir, 'captions_train2014.json')) as f:
        train_ann_df = pd.DataFrame(json.load(f)['annotations'])
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    with open(args.cluster_path, 'rb') as file:
        image_clusters = pickle.load(file)

    ###############
    # Preparation #
    ###############

    print('input caption files')

    paths = glob(args.caps_dir + '*.json')
    files = [os.path.split(res_file)[-1].replace('_cleaned', '').replace('.json', '') for res_file in paths]
    files_paths = list(zip(files, paths))
    greedy_file = 'coco_test_greedy_d-na_l-na_r-na'

    print('create DataFrames containing annotated train/val/test captions')

    with open(join(args.splits_path, 'dataset_coco.json')) as f:
        split_df = pd.DataFrame(json.load(f)['images'])
    train_ids = pd.unique(
            split_df.loc[split_df.split.isin(['train', 'restval'])].cocoid
        ).tolist()
    test_ids = pd.unique(
            split_df.loc[split_df.split == 'test'].cocoid
        ).tolist()

    train_caps_df = pd.concat((
        train_ann_df.loc[train_ann_df.image_id.isin(train_ids)],
        val_ann_df.loc[val_ann_df.image_id.isin(train_ids)]
    ))
    test_caps_df = val_ann_df.loc[val_ann_df.image_id.isin(test_ids)]

    print('get tokens from speaker train/val/test captions')

    train_caps = train_caps_df.caption.values
    train_tokens = tokenize_caps_list(train_caps)

    # restrict train tokens to types in model vocab
    iv_train_tokens = [t for t in train_tokens if t in vocab.word2idx.keys()]

    # sample from annotated test captions: one caption per image
    test_caps = test_caps_df\
        .groupby('image_id')\
        .apply(lambda x: x.sample(1, random_state=args.random_seed))\
        .set_index('image_id')
    # normalize captions
    test_caps['caption'] = test_caps.caption.map(
            lambda x: ' '.join(tokenize_caption(x))
        )

    ann_fname = 'coco_test_annotations_d-na_l-na_r-na_t-na_p-na_k-na.json'

    if os.path.isfile(
            os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), ann_fname
                )
            ):
        print('load caption file for sampled human annotations')
        with open(ann_fname) as f:
            human_caps = json.load(f)
        test_caps = [c['caption'] for c in human_caps]

    else:
        raise Exception
        # output json file with sampled captions
        caption_dict = test_caps.reset_index()[['image_id', 'caption']]\
            .to_dict(orient='records')
        print('write sampled human captions to {}'.format(ann_fname))
        #with open(ann_fname, 'w') as f:
            #json.dump(caption_dict, f)
        test_caps = test_caps.caption.values

    test_tokens = tokenize_caps_list(test_caps)

    ###########################################
    # Gained / Lost Types; type / token ranks #
    ###########################################
    print('get gained/lost types; type/token ranks')

    # initialize captions dict

    captions = {
            name: {'file': file}
            for (name, file) in files_paths
            if name != 'coco_test_annotations_d-na_l-na_r-na_t-na_p-na_k-na'
        }

    # iterate through files and add to captions dict

    for c in tqdm(captions):
        print(c)
        # list of tokens
        captions[c]['tokens'] = tokenize_caps_file(captions[c]['file'])
        # list of oov tokens
        captions[c]['tokens_oov'] = vocab_diff(
                captions[c]['tokens'],
                iv_train_tokens
            )
        # list of iv tokens
        captions[c]['tokens_iv'] = [
                v for v in captions[c]['tokens']
                if v not in captions[c]['tokens_oov']
            ]
        # avg token rank
        captions[c]['rank_tokens'] = avg_rank(
                captions[c]['tokens_iv'],
                iv_train_tokens
            )
        # avg type rank
        captions[c]['rank_types'] = avg_rank(
                set(captions[c]['tokens_iv']),  # use set() to get types
                iv_train_tokens
            )

    # iterate through files and add tokens gained / lost with respect to greedy

    for c in tqdm(captions):
        captions[c]['tokens_gained'] = [
            t for t in vocab_diff(
                captions[c]['tokens'],  # tokens in the current token list
                captions[greedy_file]['tokens']  # but not in greedy tokens
            )
            if t in captions[c]['tokens_iv']  # restrict to iv tokens
        ]
        captions[c]['tokens_lost'] = [
            t for t in vocab_diff(
                captions[greedy_file]['tokens'],  # tokens in greedy tokens
                captions[c]['tokens']  # but not in the current token list
            )
            if t in captions[greedy_file]['tokens_iv']  # restrict to iv tokens
        ]

    print('get human baseline with annotated test captions')

    # oov tokens: not in train tokens
    human_tokens_oov = vocab_diff(test_tokens, train_tokens)
    # iv tokens: in train tokens
    human_iv_tokens = [
            t for t in test_tokens if t not in human_tokens_oov
        ]
    # restrict tokens to words in the model vocab
    cleaned_human_iv_tokens = [
            t for t in human_iv_tokens if t in vocab.word2idx.keys()
        ]

    ann_fname = ann_fname.replace('_cleaned', '').replace('.json', '')
    captions[ann_fname] = {
        'file': None,
        'tokens': test_tokens,
        'tokens_oov': human_tokens_oov,
        'tokens_iv': human_iv_tokens,
        'tokens_gained': [
            t for t in vocab_diff(
                cleaned_human_iv_tokens,  # tokens which are in cleaned_human_iv_tokens
                captions[greedy_file]['tokens']  # but not in greedy tokens
            )
        ],
        'tokens_lost': [
            t for t in vocab_diff(
                captions[greedy_file]['tokens'],  # tokens which are in in greedy tokens
                cleaned_human_iv_tokens  # but not in cleaned_human_iv_tokens
            )
            if t in captions[greedy_file]['tokens_iv']  # restrict to iv tokens
        ],
        'rank_tokens': avg_rank(
                cleaned_human_iv_tokens,
                iv_train_tokens
            ),
        'rank_types': avg_rank(
                set(cleaned_human_iv_tokens),  # use set() to get types
                iv_train_tokens
            )
    }

    print('dump raw caption data')
    with open('lexical_analysis_raw.pkl', 'wb') as f:
        pickle.dump(captions, f)

    print('create dataframe with results')

    results_df = pd.DataFrame()
    for c in captions:  # add data for every caption file to dataframe
        results_df = results_df.append(pd.Series(
            {
                '% OOV_tokens': (len(captions[c]['tokens_oov']) / len(captions[c]['tokens']))*100,
                '% OOV_types': (len(set(captions[c]['tokens_oov'])) / len(set(captions[c]['tokens'])))*100,  # use set() to get types
                'gained_types': len(set(captions[c]['tokens_gained'])),  # use set() to get types
                'lost_types': len(set(captions[c]['tokens_lost'])),  # use set() to get types
                'rank_tokens': captions[c]['rank_tokens'],
                'rank_types': captions[c]['rank_types']
            }, name=c
        ))

    fname = 'oov_gained_rank.csv'
    print('write results to {}'.format(fname))
    # save to file
    results_df.to_csv(fname)

    ##############################
    # Captions with gained types #
    ##############################

    print('get captions with gained types')

    disc_results = pd.read_csv(
        args.discriminativeness_path,
        converters={
            'all_ranks': lambda x: eval(x),
            'target_positions': lambda x: eval(x),
            'image_cluster': lambda x: eval(x),
        }
    )

    disc_results = disc_results.rename(columns={'Unnamed: 0': 'file'})
    res_df = disc_results.loc[disc_results.n_eval_dists == 2].set_index('file')

    assert not False in res_df.image_cluster.map(
            lambda x: x == image_clusters
        ).values

    retrieval_results = []
    for i, cluster in enumerate(image_clusters):
        r = res_df.target_positions.map(lambda x: x[i] == 0)
        retrieval_results.append(
            (cluster, r)
        )

    retrieval_df = pd.DataFrame()
    for image_ids, results in tqdm(retrieval_results):

        results = results.to_dict()
        results = pd.Series(results, name=image_ids[0])

        retrieval_df = retrieval_df.append(results)
    retrieval_df = retrieval_df.astype(bool)

    results = {}
    # iterate through files and names for captiosn
    for name, file in tqdm(files_paths):
        if name not in res_df.index:
            continue
        # get captions as df
        with open(file, 'rb') as f:
            df = pd.DataFrame(
                json.load(f)
            ).set_index('image_id')
        # prepare regex
        words = set(captions[name]['tokens_gained'])
        search = '({})'.format('|'.join(words))
        k = re.compile(r'(\b|^)%s(\b|$)' % search, re.I)
        # mark captions that contain gained tokens
        df['hit'] = df.apply(
            lambda x: k.findall(x.caption) if k.findall(x.caption) else False, axis=1
        )
        # get the indices for captions correctly guessed by the listener
        correct = retrieval_df.loc[retrieval_df[name] == True].index
        # get the indices for captions which include additional tokens
        includes_gained = df.loc[df.hit != False].index

        results[name] = {
            'correct': len(correct),
            'includes_gained': len(includes_gained),
            'correct_gained': len(set(correct).intersection(set(includes_gained))),
        }

    gained_caps_df = pd.DataFrame(results).T
    fname = 'gained_caps.csv'
    print('write results to {}'.format(fname))
    gained_caps_df.to_csv(join(args.out_path, fname))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--caps_dir',
        help='path to generated captions to be analyzed',
        default='../data/generated_captions'
    )

    parser.add_argument('--coco_ann_dir',
        help='path to COCO annotation directory')
    parser.add_argument('--splits_path',
        help='path to karpathy split file')

    parser.add_argument('--vocab_file',
        help='path to model vocab',
        default='../data/model/coco_vocab.pkl'
    )
    parser.add_argument('--discriminativeness_path',
        help='path to discriminativeness_results.csv previously generated'
    )
    parser.add_argument('--cluster_path',
        help='path to image cluster file',
        default='../data/image_clusters/image_clusters_test_3.pkl'
    )
    parser.add_argument('--out_path',
        help='path for storing results',
        default='../data/results')

    parser.add_argument('--random_seed',
        default=123
    )

    args = parser.parse_args()

    main(args)
