import json
import os
import pandas as pd
import argparse

def main(args):

    assert args.res_file.endswith('.json'), '{}: bad input file'.format(
        args.res_file)

    with open(args.template_file) as f:
        template_data = json.load(f)

    with open(args.res_file) as f:
        res_data = json.load(f)

    ann_data = pd.DataFrame(res_data)\
        .reset_index()\
        .rename(columns={
                args.imgid_column: 'image_id',
                args.caps_column: 'caption',
                'index': 'id'
                })\
        .to_dict(orient='records')

    if args.sort_caps:
        ann_data = sorted(ann_data, key=lambda k: k['image_id'])

    out_data = dict()

    for key in ['info', 'images', 'licenses']:
        out_data[key] = template_data[key]
    out_data['type'] = 'captions'
    out_data['annotations'] = ann_data

    if len(args.output_file) > 0:
        outfile = args.output_file
    else:
        outfile = args.res_file.replace('.json', '_reformat.json')

    with open(outfile, 'w') as f:
        json.dump(out_data, f)
        print('saved to file', outfile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--template_file", type=str,
                        help="path to template file")
    parser.add_argument("--res_file", type=str,
                        help="path to res file")
    parser.add_argument("--res_dir", type=str,
                        default="",
                        help="directory containing res files")
    parser.add_argument("--imgid_column", type=str,
                        default='image_id',
                        help="column name for image id")
    parser.add_argument("--caps_column", type=str,
                        default='caption',
                        help="column name for captions")
    parser.add_argument("--output_file", type=str,
                        default='',
                        help='output file')
    parser.add_argument("--sort_caps", action="store_true")

    args = parser.parse_args()

    if args.res_dir != "":
        args.output_file = ""
        files = os.listdir(args.res_dir)
        for file in files:
            args.res_file = os.path.join(args.res_dir, file)
            main(args)
    else:
        main(args)
