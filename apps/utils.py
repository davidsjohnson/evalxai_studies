import os
import random
import time
from datetime import datetime
from pathlib import Path
import shutil
import re
import hashlib

import pandas as pd

import owncloud

from dotenv import load_dotenv
load_dotenv();

HASH_INTROS = {'training': 'a1c', 'validation_nodiag':'b55', 'validation_noxai': 'z81'}

# random.seed(0)

class Example():
  
  def __init__(self, 
               id: str,
               disp_id: str,
               image_path: Path,
               true: int,
               pred: int
              ):
    
    self.id: str = id
    self.disp_id: str = disp_id 
    self.image_path: Path = image_path
    self.true: int = true
    self.pred: int = pred

def init(oc_path, tmp_folder, num_samples):
    # get and process file
    image_folder = download_examples(oc_path, tmp_folder)
    # ensure order before shuffle for replicability
    image_paths = list(image_folder.rglob('*.png'))
    image_paths.sort()
    random.shuffle(image_paths)

    assert len(image_paths) == num_samples, f'Images Not loaded from {image_folder} - should be {num_samples} images, but there are {len(image_paths)}'

    return image_paths

# avoid time collisions
def create_tmp_folder(stage):
    created = False
    while not created:
        tmp_folder = Path('temp') / datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3] / stage
        try:
            os.makedirs(tmp_folder, exist_ok=False)
            created = True
        except OSError as e:
            time.sleep(0.001) # adding a sleep so loop doesn't take the whole processsing
    return tmp_folder


disp_ids = []

def setup_examples(image_paths):
  examples = []
  for path in image_paths:

    # get image data
    id, true, pred, _ = path.stem.split( '_')
    
    disp_id = random.randint(10000, 99999)
    while disp_id in disp_ids:
      disp_id = random.randint(10000, 99999)
    disp_ids.append(disp_id)
       
    
    true = int(true.split('=')[-1])
    pred = int(pred.split('=')[-1])

    examples.append(Example(id, disp_id, path, true, pred))

  return examples

def register_example(results, example, init_select):

  results[example.id] = dict(
    true = example.true,
    pred = example.pred,
    select = init_select
  )

  return results

def save_results(results, id, oc_path, tmp_folder, stage):

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'id'})

    results_df.to_csv(tmp_folder / f'results_{stage}_{id}.csv', index=False)

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    outfile = f'results/results_{stage}_{id}.csv'
    oc.put_file(str(oc_path / outfile), tmp_folder / f'results_{stage}_{id}.csv')

    oc.logout()

    return results_df

def results_exist(id, oc_path, tmp_path, stage):

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    try:
        oc.get_file(str(oc_path / f'results/results_{stage}_{id}.csv'), str(tmp_path / f'results_{stage}_{id}.csv'))
        return True
    except owncloud.HTTPResponseError as e:
        return False
    finally:
        oc.logout()


def hash_prof_id(prof_id, stage, len=15):
  hash_text = prof_id + f'supersecret_stage-{stage}'
  full_hash = hashlib.sha256(hash_text.encode()).hexdigest()
  return HASH_INTROS[stage] + full_hash[:len-3] # truncate

# update before final study
def id_valid(id_str, id_len=5):
  pattern = fr'^[a-zA-Z0-9]{{{id_len}}}$'
  return bool(re.match(pattern, id_str))


def download_examples(oc_path, tmp_folder):

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    zipfile = 'xai_samples.zip'

    oc.get_file(str(oc_path / zipfile), tmp_folder / zipfile)

    oc.logout()

    shutil.unpack_archive(tmp_folder / zipfile, tmp_folder, format='zip')

    return tmp_folder / Path(zipfile).stem


### Unit Tests

def test_init():
  stage = 'training'
  oc_path = Path(f'1. Research/1. HCXAI/1. Projects/evalxai_studies/example_validation_study/{stage}')
  tmp_folder = create_tmp_folder(stage)
  num_samples = 10
  paths = init(oc_path, tmp_folder, num_samples)
  
  assert len(paths) == num_samples, f'Images Not loaded from {oc_path} - should be {num_samples} images, but there are {len(paths)}'

def test_results_exist():
  prof_id = '12345'
  stage = 'training'
  tmp_folder = create_tmp_folder(stage)
  oc_basedir = Path(f'1. Research/1. HCXAI/1. Projects/evalxai_studies/example_validation_study/{stage}')
  res = results_exist(prof_id, oc_basedir, tmp_folder, stage)
  
  assert res != True, 'File Exists. Should return true'

def test_hash_prof_id():
  prof_id = '12345'
  hash_len = 15
  hash = hash_prof_id(prof_id, stage=0, len=hash_len)

  assert len(hash) == hash_len, 'Hash is not correct length.'
  assert hash == HASH_INTROS[0]+'fe8a11050dcb', f'Incorrect hash value for input. Expected: {HASH_INTROS[0]+"fe8a11050dcb"} - Received: {hash}'

def test():
  # test_init()
  # test_results_exist()
  test_hash_prof_id()

if __name__ == "__main__":
   test()