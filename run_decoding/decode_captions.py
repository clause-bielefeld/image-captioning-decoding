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

from decoding import img_from_file, prepare_img, greedy, beam_search, top_p_sampling, top_k_sampling
from predict_coco import get_img_dir
from model.build_vocab import Vocabulary
from model.adaptive import Encoder2Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename_template = 'coco_{split}_{method}_d-{distractors}_l-{lambda_}_r-{rationality}_t-{temperature}_p-{top_p}_k-{top_k}.json'

print('Device:', device)

def main(args):

    # load vocab
    with open(args.coco_vocab, 'rb') as f:
        vocab = pickle.load(f)

    # get image clusters
    with open(args.coco_cluster, 'rb') as f:
        image_clusters = pickle.load(f)

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

    greedy_caps, beam_caps, nucleus_caps, topk_caps = list(), list(), list(), list()

    for i in tqdm(range(len(image_clusters))):

        cluster = image_clusters[i]

        target = cluster[0]
        res_temp = {'image_id': target}

        target_img = img_from_file(get_img_dir(target, args.image_dir))
        t = prepare_img(target_img, transform)

        if args.do_greedy:
            # greedy

            ids, _, _ = greedy(
                model, t,
                max_len=args.max_len,
                end_id=vocab.word2idx['<end>']
                )

            words = [vocab.idx2word[ix] for ix in ids[:-1]]
            greedy_res = {**res_temp}
            greedy_res.update({
                'caption': args.separator.join(words)
            })
            greedy_caps.append(greedy_res)
            #print('greedy:', greedy_res)

        if args.do_beam:
            # beam search

            _, ids = beam_search(
                model, t,
                beam_width=args.beam_width,
                max_len=args.max_len
                )

            words = [vocab.idx2word[ix] for ix in ids[:-1]]
            beam_res = {**res_temp}
            beam_res.update({
                'caption': args.separator.join(words)
            })
            beam_caps.append(beam_res)
            #print('beam:', beam_res)

        if args.do_nucleus:
            # nucleus sampling

            ids, _, _ = top_p_sampling(
                model, t,
                max_len=args.max_len,
                end_id=vocab.word2idx['<end>'],
                top_p=args.top_p,
                temperature=args.temperature
                )

            words = [vocab.idx2word[ix] for ix in ids[:-1]]
            nucleus_res = {**res_temp}
            nucleus_res.update({
                'caption': args.separator.join(words)
            })
            nucleus_caps.append(nucleus_res)
            #print('nucleus:', nucleus_res)

        if args.do_topk:
            # top-k sampling

            ids, _, _ = top_k_sampling(
                model, t,
                max_len=args.max_len,
                end_id=vocab.word2idx['<end>'],
                top_k=args.top_k,
                temperature=args.temperature
                )

            words = [vocab.idx2word[ix] for ix in ids[:-1]]
            topk_res = {**res_temp}
            topk_res.update({
                'caption': args.separator.join(words)
            })
            topk_caps.append(topk_res)
            #print('top-k sampling:', topk_caps)

    # save files

    if args.do_greedy:
        filename = filename_template.format(
            split=args.split,
            method='greedy',
            distractors='na',
            lambda_='na',
            rationality='na',
            temperature='na',
            top_p='na',
            top_k='na'
        )
        filename = join(args.out_dir, filename)
        with open(filename, 'w') as f:
            json.dump(greedy_caps, f)
            print('saved to', filename)

    if args.do_beam:
        filename = filename_template.format(
            split=args.split,
            method='beam',
            distractors='na',
            lambda_='na',
            rationality='na',
            temperature='na',
            top_p='na',
            top_k='na'
        )
        filename = join(args.out_dir, filename)
        with open(filename, 'w') as f:
            json.dump(beam_caps, f)
        print('saved to', filename)

    if args.do_nucleus:
        filename = filename_template.format(
            split=args.split,
            method='nucleus',
            distractors='na',
            lambda_='na',
            rationality='na',
            temperature=str(args.temperature).replace('.', '-'),
            top_p=str(args.top_p).replace('.', '-'),
            top_k='na'
        )
        filename = join(args.out_dir, filename)
        with open(filename, 'w') as f:
            json.dump(nucleus_caps, f)
        print('saved to', filename)

    if args.do_topk:
        filename = filename_template.format(
            split=args.split,
            method='topk',
            distractors='na',
            lambda_='na',
            rationality='na',
            temperature=str(args.temperature).replace('.', '-'),
            top_p='na',
            top_k=str(args.top_k)
        )
        filename = join(args.out_dir, filename)
        with open(filename, 'w') as f:
            json.dump(topk_caps, f)
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

    parser.add_argument('--do_greedy', action='store_true')
    parser.add_argument('--do_beam', action='store_true')
    parser.add_argument('--do_nucleus', action='store_true')
    parser.add_argument('--do_topk', action='store_true')
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_width', type=int, default=5)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--separator', type=str, default=' ')

    args = parser.parse_args()
    args.separator = '' if args.separator in ['', 'char'] else ' '

    pprint(vars(args))

    main(args)
