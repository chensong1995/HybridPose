import os
from multiprocessing import Pool
import urllib.request

url = 'https://cloudstore.zih.tu-dresden.de/index.php/s/a65ec05fedd4890ae8ced82dfcf92ad8/download'
target_dir = 'data/temp'

def download_and_unzip():
    os.makedirs(target_dir, exist_ok=True)
    target_file = os.path.join(target_dir, 'occlusion.zip')
    urllib.request.urlretrieve(url, target_file)
    os.system('unzip {} -d data'.format(target_file))
    os.system('mv data/OcclusionChallengeICCV2015 data/occlusion_linemod')
    # to lower case
    os.system('for i in $( ls data/occlusion_linemod/models | grep [A-Z] ); do mv -i data/occlusion_linemod/models/"$i" data/occlusion_linemod/models/"`echo $i | tr \'A-Z\' \'a-z\'`"; done')
    os.system('for i in $( ls data/occlusion_linemod/poses | grep [A-Z] ); do mv -i data/occlusion_linemod/poses/"$i" data/occlusion_linemod/poses/"`echo $i | tr \'A-Z\' \'a-z\'`"; done')

download_and_unzip()
