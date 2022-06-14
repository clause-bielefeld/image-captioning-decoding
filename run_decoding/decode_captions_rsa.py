import os
from os.path import join
import sys
import pickle
import torch
from torchvision import transforms
from tqdm.autonotebook import tqdm
import json
import argparse
from pprint import pprint

file_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.realpath(os.path.join(file_path, os.pardir))
sys.path.append(dir_path)
sys.path.append(join(dir_path, 'rsa'))
sys.path.append(join(dir_path, 'model'))

from bayesian_agents.joint_rsa import RSA
from predict_coco import rsa_decode_cluster
from model.build_vocab import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename_template = 'coco_{split}_{method}_d-{distractors}_l-{lambda_}_r-{rationality}.json'


def main(args):

    # load vocab
    with open(args.coco_vocab, 'rb') as f:
        vocab = pickle.load(f)

    # get image clusters
    with open(args.coco_cluster, 'rb') as f:
        image_clusters = pickle.load(f)

    n_dist = len(image_clusters[0])-1
    print(n_dist, 'distractors per target')

    # define image transformation parameters
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    # initialize speakers using the model parameters
    speaker_model = RSA(vocabulary=vocab)
    speaker_model.initialize_speakers(model_path=args.coco_model)
    # the rationality of the S1
    rat = [args.speaker_rat]

    caps = []

    for i in tqdm(range(len(image_clusters))):
        cluster = image_clusters[i]
        target = cluster[0]
        distractors = cluster[1:]
        c = rsa_decode_cluster(
            i, speaker_model, rat, image_clusters,
            transform, args.image_dir, beam=args.do_beam,
            mixed=args.do_mixed, separator=args.separator,
            beam_width=args.beam_width, max_len=args.max_len
            )
        res = {
            'image_id': target,
            'distractors': distractors,
            'caption': c['caption'][:-6]
            }
        print(res)
        caps.append(res)

    filename = filename_template.format(
        split=args.split,
        method='rsa',
        distractors=str(n_dist),
        lambda_='na',
        rationality=str(args.speaker_rat).replace('.', '-'),
    )
    filename = join(args.out_dir, filename)
    with open(filename, 'w') as f:
        json.dump(caps, f)
    print('saved to', filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', default='self', help='To make it runnable in jupyter')

    parser.add_argument('--coco_model', type=str,
                        default='../data/model/best.pkl')
    parser.add_argument('--coco_vocab', type=str,
                        default='../data/model/coco_vocab.pkl')
    parser.add_argument('--out_dir', type=str,
                        default='../data/generated_captions/')
    parser.add_argument('--image_dir', type=str, required=True)

    parser.add_argument('--coco_cluster', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')

    parser.add_argument('--do_beam', type=bool, default=True)
    parser.add_argument('--do_mixed', type=bool, default=False)
    parser.add_argument('--speaker_rat', type=float, default=1.0)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--separator', type=str, default=' ')

    args = parser.parse_args()
    args.separator = '' if args.separator in ['', 'char'] else ' '

    pprint(vars(args))

    main(args)
