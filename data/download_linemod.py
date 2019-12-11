import os
from multiprocessing import Pool
import urllib.request

object_names = ['ape', 'benchviseblue', 'bowl', 'cam', 'can', 'cat',
                'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher',
                'iron', 'lamp', 'phone']
base_url = 'http://campar.in.tum.de/personal/hinterst/index/downloads!09384230443!/{}.zip'
target_dir = 'data/temp'

def download_and_unzip(object_name):
    url = base_url.format(object_name)
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, 'linemod_{}.zip'.format(object_name))
    urllib.request.urlretrieve(url, target_file)
    os.makedirs('data/linemod', exist_ok=True)
    os.system('unzip {} -d data/linemod/original_dataset'.format(target_file))

pool = Pool()
pool.map(download_and_unzip, object_names)
