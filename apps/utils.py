import os
import random
from pathlib import Path
import shutil

import pandas as pd

import owncloud

# random.seed(0)

class Example():
  
  def __init__(self, 
               id: str,
               image_path: Path,
               true: int,
               pred: int
              ):
    
    self.id: str = id
    self.image_path: Path = image_path
    self.true: int = true
    self.pred: int = pred

def init(oc_path, tmp_folder):
    # get and process file
    image_folder = download_examples(oc_path, tmp_folder)
    # ensure order before shuffle for replicability
    image_paths = list(image_folder.rglob('*.png'))
    image_paths.sort()
    random.shuffle(image_paths)

    assert len(image_paths) == 36, f'Images Not loaded from {image_folder} - should be 30 images, but there are {len(image_paths)}'

    return image_paths

def setup_examples(image_paths):
  examples = []
  for path in image_paths:

    # get image data
    id, true, pred, _ = path.stem.split( '_')
    true = int(true.split('=')[-1])
    pred = int(pred.split('=')[-1])

    examples.append(Example(id, path, true, pred))

  return examples

def register_example(results, example, init_select):

  results[example.id] = dict(
    true = example.true,
    pred = example.pred,
    select = init_select
  )

  return results

def save_results(results, id, oc_path):

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'id'})

    results_df.to_csv(f'results_{id}.csv', index=False)

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    oc.put_file(str(oc_path / f'results/results_{id}.csv'), 
           f'results_{id}.csv')

    oc.logout()

    return results_df

def results_exist(id, oc_path):

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    try:
        oc.get_file(str(oc_path / f'results_{id}.csv'), f'results_{id}.csv')
        return True
    except owncloud.HTTPResponseError as e:
        return False
    finally:
        oc.logout()


def download_examples(oc_path, tmp_folder):

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    oc.get_file(str(oc_path / 'xai_samples.zip'), tmp_folder / 'xai_samples.zip')

    oc.logout()

    shutil.unpack_archive(tmp_folder /'xai_samples.zip', tmp_folder, format='zip')

    return tmp_folder / 'xai_samples'