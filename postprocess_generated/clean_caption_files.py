import argh
import json
from os.path import join, split
from glob import glob
from copy import deepcopy
import re


def main(source_dir, out_dir):
    files = glob(join(source_dir, '*.json'))

    for file in files:

        filename = split(file)[-1]

        with open(file, 'r') as f:
            data = json.load(f)

        clean_data = deepcopy(data)
        for i in range(len(clean_data)):
            # get current caption
            c = clean_data[i]['caption']
            # replace tokens
            c = re.sub(r'(<start>)|(<end>)|(<unk>)|(<pad>)', '', c)
            c = re.sub(r'  ', ' ', c)
            c = c.rstrip()
            # store cleaned caption
            clean_data[i]['caption'] = c

        out_path = join(
            out_dir, filename.replace('.json', '_cleaned.json')
        )

        with open(out_path, 'w') as f:
            json.dump(clean_data, f)


if __name__ == '__main__':

    argh.dispatch_command(main)
