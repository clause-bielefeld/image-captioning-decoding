import os
import json
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter
from nltk.corpus import wordnet as wn
import argparse
import spacy


def get_pos_words(tokens, tags, pos_tag):
    matched = [list(zip(tokens[i], tags[i])) for i in tokens.keys()]
    flat = [item for sublist in matched for item in sublist]
    return [i for i, j in flat if j == pos_tag]


def wn_distance_from_root(word):
    """
    calculate distance from wn-root for a given word
    """
    synset = wn.synsets(word)[0]
    return min(
        [len(path) for path in synset.hypernym_paths()]
    )


def main(args):

    print('prepare spacy model & file paths')

    nlp = spacy.load("en_core_web_lg")
    # try en_core_web_sm if this doesn't work

    # input caption files
    paths = glob(args.caps_dir + '*.json')
    files = [
        os.path.split(res_file)[-1]\
               .replace('_cleaned', '')\
               .replace('.json', '')
        for res_file in paths
        ]
    files_paths = list(zip(files, paths))

    # initialize captions dict
    captions = {
        name: {'file': file}
        for (name, file)
        in files_paths
    }

    print('tokenize and tag captions')

    for c in tqdm(captions):

        with open(captions[c]['file'], 'rb') as f:
            data = json.load(f)

        caps = {d['image_id']: d['caption'] for d in data}
        captions[c]['captions'] = caps

        tokens = {}
        tags = {}
        detailed_tags = {}

        for imgid, cap in caps.items():

            doc = nlp(cap)
            tagged = [(token.text, token.pos_, token.tag_) for token in doc]

            tkns = [t[0] for t in tagged]
            tokens[imgid] = tkns

            ts = [t[1] for t in tagged]
            tags[imgid] = ts

            d_ts = [t[2] for t in tagged]
            detailed_tags[imgid] = d_ts

        captions[c]['tokens'] = tokens
        captions[c]['base_pos'] = tags
        captions[c]['detail_pos'] = detailed_tags

    fname = 'captions_tokens_tags.pkl'
    print('dump tokenized & tagged files to {}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(captions, f)

    print('get POS frequencies and WN distances for all generated tokens')
    df = pd.DataFrame()
    for i in sorted(captions.keys()):
        pos = captions[i]['base_pos']

        # POS frequencies

        all_pos = [item for sublist in pos.values() for item in sublist]
        pos_counts = dict(Counter(all_pos))

        tags = [i[0] for i in sorted(pos_counts.items())]
        counts = [i[1] for i in sorted(pos_counts.items())]
        #normalized_counts = [c/sum(counts) for c in counts]

        for key in pos_counts:
            pos_counts[key] = pos_counts[key] / sum(counts)

        # WN distances

        nouns = get_pos_words(captions[i]['tokens'], pos, 'NOUN')
        wn_distances = [wn_distance_from_root(n) for n in nouns if wn.synsets(n)]
        avg_wn_distance = sum(wn_distances) / len(wn_distances)

        pos_counts.update({'avg_wn_distance': avg_wn_distance})

        series = pd.Series(pos_counts, name=i)
        df = df.append(series)

    #df = df.fillna(0)
    fname = 'tokens_pos_distribution_wn_distance.csv'
    print('write results to {}'.format(fname))
    df.to_csv(fname)

    with open('lexical_analysis_raw.pkl', 'rb') as f:
        raw_data = pickle.load(f)

    # gained tokens
    print('get POS frequencies and WN distances for gained tokens (in comparison to greedy)')
    df = pd.DataFrame()

    for i in sorted(captions.keys()):
        raw = raw_data[i]
        gained = raw['tokens_gained']
        pos = captions[i]['base_pos']

        pos_tokens = []
        for k in captions[i]['tokens'].keys():
            zipped = list(zip(captions[i]['tokens'][k], captions[i]['base_pos'][k]))
            pos_tokens += [z for z in zipped if z[0] in raw['tokens_gained']]
        gained_pos = [pos for token, pos in pos_tokens]

        pos_counts = dict(Counter(gained_pos))

        tags = [i[0] for i in sorted(pos_counts.items())]
        counts = [i[1] for i in sorted(pos_counts.items())]
        #normalized_counts = [c/sum(counts) for c in counts]

        for key in pos_counts:
            pos_counts[key] = pos_counts[key] / sum(counts)

        nouns = get_pos_words(captions[i]['tokens'], captions[i]['base_pos'], 'NOUN')
        gained_nouns = [n for n in nouns if n in gained]
        if len(gained_nouns) > 0:
            wn_distances = [wn_distance_from_root(n) for n in gained_nouns if wn.synsets(n)]
            avg_wn_distance = sum(wn_distances) / len(wn_distances)
        else:
            avg_wn_distance = 0

        pos_counts.update({'avg_wn_distance': avg_wn_distance})

        series = pd.Series(pos_counts, name=i)
        df = df.append(series)

    #df = df.fillna(0)
    fname = 'gained_tokens_pos_distribution_wn_distance.csv'
    print('write results to {}'.format(fname))
    df.to_csv(fname)

    # gained types
    print('get POS frequencies and WN distances for gained types (in comparison to greedy)')
    df = pd.DataFrame()

    for i in sorted(captions.keys()):
        raw = raw_data[i]
        gained = raw['tokens_gained']
        pos = captions[i]['base_pos']

        pos_tokens = []
        for k in captions[i]['tokens'].keys():
            zipped = list(zip(captions[i]['tokens'][k], captions[i]['base_pos'][k]))
            zipped = set(zipped)
            pos_tokens += [z for z in zipped if z[0] in raw['tokens_gained']]
        gained_pos = [pos for token, pos in pos_tokens]

        pos_counts = dict(Counter(gained_pos))

        tags = [i[0] for i in sorted(pos_counts.items())]
        counts = [i[1] for i in sorted(pos_counts.items())]
        #normalized_counts = [c/sum(counts) for c in counts]

        for key in pos_counts:
            pos_counts[key] = pos_counts[key] / sum(counts)

        nouns = get_pos_words(captions[i]['tokens'], captions[i]['base_pos'], 'NOUN')
        nouns = set(nouns)
        gained_nouns = [n for n in nouns if n in gained]
        if len(gained_nouns) > 0:
            wn_distances = [wn_distance_from_root(n) for n in gained_nouns if wn.synsets(n)]
            avg_wn_distance = sum(wn_distances) / len(wn_distances)
        else:
            avg_wn_distance = 0

        pos_counts.update({'avg_wn_distance': avg_wn_distance})

        series = pd.Series(pos_counts, name=i)
        df = df.append(series)

    # df = df.fillna(0)
    fname = 'gained_types_pos_distribution_wn_distance.csv'
    print('write results to {}'.format(fname))
    df.to_csv(fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--caps_dir',
        help='path to generated captions to be analyzed',
        default='../data/generated_captions'
    )

    args = parser.parse_args()

    main(args)
