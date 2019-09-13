import sys
import json
import pickle
import argparse

DATASET_DIR='datasets'

def find_match(dataset_file_name):
    dataset = {}
    with open(f'{DATASET_DIR}/{dataset_file_name}','rb') as infile:
        dataset = pickle.load(infile)

    print(json.dumps(dataset, indent=4))
    
    
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='dataset_file_name', type=str, help='Dataset File Name (ex: sample-catalog)')
    # parser.add_argument('-s', dest='space_id', type=str, help='space id', required=True)
    # parser.add_argument('-d', dest='dataset_id', type=str, help='dataset id', required=True)

    params = parser.parse_args(argv)
    find_match(params.dataset_file_name)

if __name__ == '__main__':
    main(sys.argv[1:])