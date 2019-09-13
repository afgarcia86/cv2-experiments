import os
import sys
import json
import shutil
import pickle
import zipfile
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TMP_DIR='tmp'
DATASET_DIR='datasets'

def unzip_and_process_catalog(file_name):
    zippedfile = zipfile.ZipFile(f'{file_name}.zip', 'r')
    outfile = zippedfile.extractall(TMP_DIR)
    zippedfile.close()

    src_images = []
    src_data = None
    for f in glob(f'{TMP_DIR}/{file_name}/*'):
        if f.endswith('.csv'):
            dtype = {k: 'str' for k in open(f).readline().replace('\n', '').split(',')}
            src_data = pd.read_csv(f, error_bad_lines=False, dtype=dtype, keep_default_na=False, index_col=False)
        elif any(f.endswith(ext) for ext in ['.png', '.jpg']):
            src_images.append(f)

    if not len(src_images):
        raise Exception('at least one or more images are required')
    elif src_data.empty:
        raise Exception('a csv with data for the images is required')
    
    return src_images, src_data

def cleanup():
    for f in os.listdir(TMP_DIR):
        shutil.rmtree(f'{TMP_DIR}/{f}')

def build_dataset(file_name):
    try:
        # Process Catalog
        src_images, src_data = unzip_and_process_catalog(file_name)

        out_data = {}
        for item in src_data.to_dict('records'):
            filename = item.pop('filename', None)
            item['_id'] = item.pop('item_id', None)
            if not out_data.get(filename, None):
                out_data[filename] = {'items': [item]}
            else:
                out_data[filename]['items'].append(item)

        # Save Dataset
        for k, item in out_data.items():
            print(json.dumps({k: item}, indent=4))

        
        # Prepare Images
        gen = ImageDataGenerator(rescale=1./255)
        generator = gen.flow_from_dataframe(
            src_data.drop_duplicates(subset='filename'),
            directory=f'{TMP_DIR}/{file_name}',
            x_col='filename',
            y_col='item_id',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='sparse',
            batch_size=1,
            shuffle=False
        )

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path='ReferenceImageDetectionModel.tflite')
        interpreter.allocate_tensors()

        # Get input and output tensors.
        total_classes = len(generator.classes)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        filenames = generator.filenames

        # Process Images
        for i in range(0, total_classes):
            x, y = generator[i]
            filename = filenames[int(y[0])]
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            out_data[filename]['features'] = prediction.tolist()
        
        # Save Dataset
        for k, item in out_data.items():
            print(json.dumps({k: item}, indent=4))

        with open(f'{DATASET_DIR}/{file_name}','wb') as output:
            pickle.dump(out_data, output)

        # Cleanup
        cleanup()
 
    except Exception as e:
        sys.exit(f'ERROR: {e}')

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='file_name', type=str, help='Catalog File Name (ex: sample-catalog)')

    params = parser.parse_args(argv)
    build_dataset(params.file_name)

if __name__ == '__main__':
    main(sys.argv[1:])