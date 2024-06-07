"""
Generate scenarios from a folder contains multiple WAYMO data records
Author: Detian Guo
Date: 04/11/2022
"""
import argparse
import json
from pathlib import Path
import re
import time
import traceback
import multiprocessing  

from rich.progress import track
import tensorflow as tf
from helpers.create_rect_from_file import features_description, get_parsed_carla_data
from logger.logger import *
from scenario_miner import ScenarioMiner
from tags_generator import TagsGenerator
from warnings import simplefilter
simplefilter('error')
import os

def dir_path(path):
    if os.path.isdir(path):
        return path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Absolute path to the data directory')

# working directory 
ROOT = Path(__file__).parents[1]
#ROOT = Path(parser.parse_args().data_dir.rstrip("/training"))

# modify the following two lines to your own data and result directory
DATA_DIR = Path(parser.parse_args().data_dir)

DATA_DIR_WALK = [x for x in DATA_DIR.iterdir()]

def process_file(DATA_PATH):
    FILE = DATA_PATH.name
    FILENUM = re.search(r"-(\d{5})-", FILE)
    if FILENUM is not None:
        FILENUM = FILENUM.group()[1:-1]
        print(f"Processing file: {FILE}.")
    else:
        print(f"File name error: {FILE}.")
    result_dict = {}
    try:
        fileprefix = 'Waymo'
        dataset = tf.data.TFRecordDataset(DATA_DIR / FILE, compression_type='')
        for data in dataset.as_numpy_iterator():
            parsed = tf.io.parse_single_example(data, features_description)
            scene_id = parsed['scenario/id'].numpy().item().decode("utf-8")
            print(f"Processing scene: {scene_id}.")
            result_filename = f'{fileprefix}_{FILENUM}_{scene_id}_tag.json'
            #   tagging
            tags_generator = TagsGenerator()
            general_info, \
            inter_actor_relation, \
            actors_activity, \
            actors_environment_element_intersection, \
                ego_vehicle_ID = tags_generator.tagging(parsed,FILE)
            print(f"ego_vehicleID in runner.py: {ego_vehicle_ID}")
            result_dict = {
                'general_info': general_info,
                'inter_actor_relation': inter_actor_relation,
                'actors_activity': actors_activity,
                'actors_environment_element_intersection': actors_environment_element_intersection,
                'ego_vehicle_ID': ego_vehicle_ID
              }
            print(f"result_dict[\"ego_vehicle_ID\"]: {result_dict['ego_vehicle_ID']}")
            with open(RESULT_DIR / result_filename, 'w') as f:
              print(f"Saving tags.")
              json.dump(result_dict, f)
    except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"FILE:{FILENUM}.\nTag generation error:{e}")
            logger.error(f"trace:{trace}")
            
    return True

if __name__ == '__main__':
    RESULT_TIME = time.strftime("%Y-%m-%d-%H_%M", time.localtime())
    RESULT_DIR = ROOT / "results" / RESULT_TIME
    if not RESULT_DIR.exists():
        RESULT_DIR.mkdir(exist_ok=True, parents=True)
    time_start = time.perf_counter()
    
    pool = multiprocessing.Pool(processes = os.cpu_count())
    
    results = pool.map(process_file, DATA_DIR_WALK)
    
    pool.close()
    
    pool.join()
    
    time_end = time.perf_counter()
    print(f"Time cost: {time_end - time_start:.2f}s.RESULT_DIR: {RESULT_DIR}")
    logger.info(f"Time cost: {time_end - time_start:.2f}s.RESULT_DIR: {RESULT_DIR}")