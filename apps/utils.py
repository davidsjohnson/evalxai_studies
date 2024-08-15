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

def init(oc_path, tmp_folder, num_samples, training=False):
    # get and process file
    image_folder = download_examples(oc_path, tmp_folder, training)
    # ensure order before shuffle for replicability
    image_paths = list(image_folder.rglob('*.png'))
    image_paths.sort()
    random.shuffle(image_paths)

    assert len(image_paths) == num_samples, f'Images Not loaded from {image_folder} - should be {num_samples} images, but there are {len(image_paths)}'

    return image_paths


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

def save_results(results, id, oc_path, training=False):

    results_df = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': 'id'})

    results_df.to_csv(f'results_{id}.csv', index=False)

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    outfile = f'results/results_{id}.csv' if not training else f'results_training/results_training_{id}.csv'
    oc.put_file(str(oc_path / outfile), 
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


def download_examples(oc_path, tmp_folder, training=False):

    oc = owncloud.Client('https://uni-bielefeld.sciebo.de')
    oc.login(os.getenv('OC_USER'), os.getenv('OC_SECRET'))

    zipfile = 'xai_samples.zip' if not training else 'training_samples.zip'

    oc.get_file(str(oc_path / zipfile), tmp_folder / zipfile)

    oc.logout()

    shutil.unpack_archive(tmp_folder / zipfile, tmp_folder, format='zip')

    return tmp_folder / Path(zipfile).stem