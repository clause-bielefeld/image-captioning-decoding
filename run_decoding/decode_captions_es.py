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

from decoding import img_from_file, prepare_img, extended_discriminative_beam_search
from predict_coco import get_img_dir
from model.build_vocab import Vocabulary
from model.adaptive import Encoder2Decoder

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
    #print(n_dist)

    # define image transformation parameters
    transform = transforms.Compose([
        transforms.Resize((args.crop_size, args.crop_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    model = Encoder2Decoder(256, len(vocab), 512)

    model.load_state_dict(torch.load(args.coco_model, map_location='cpu'))
    model.to(device)
    model.eval()

    disc_pred_fuse_beam_caps = list()

    for i in tqdm(range(len(image_clusters))):

        cluster = image_clusters[i]

        target = cluster[0]
        distractors = cluster[1:]
        res_temp = {'image_id': target, 'distractors': distractors}

        images = [img_from_file(get_img_dir(i, args.image_dir)) for i in cluster]
        p_images = [prepare_img(i, transform) for i in images]
        t = p_images[0]
        ds = p_images[1:]

        if args.do_pred_fuse:
            # discriminative beam search

            prob, ids = extended_discriminative_beam_search(
                model, t, ds,
                lambda_=args.lambda_,
                beam_width=args.beam_width,
                max_len=args.max_len
                )

            words = [vocab.idx2word[ix] for ix in ids[:-1]]
            disc_pred_fuse_beam_res = {**res_temp}
            disc_pred_fuse_beam_res.update({
                'caption': ' '.join(words)
            })
            disc_pred_fuse_beam_caps.append(disc_pred_fuse_beam_res)
            #print('pred_fuse:', disc_pred_fuse_beam_res)

    if args.do_pred_fuse:
        filename = filename_template.format(
            split=args.split,
            method='predfuse_es',
            distractors=str(n_dist),
            lambda_=str(args.lambda_).replace('.', '-'),
            rationality='na',
        )
        filename = join(args.out_dir, filename)
        with open(filename, 'w') as f:
            json.dump(disc_pred_fuse_beam_caps, f)
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

    parser.add_argument('--do_pred_fuse', type=bool, default=True)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--separator', type=str, default=' ')

    args = parser.parse_args()
    args.separator = '' if args.separator in ['', 'char'] else ' '

    pprint(vars(args))

    main(args)
