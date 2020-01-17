import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        #self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
        #self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))

        BDD_root = './data/bdd100k/images/100k/'
        #A: Cityscapes
        #B: BDD_day
        #with open('/home/hhsu22/bdd100k/labels/ImageSets/day%s.txt'%'train', 'r') as f:
        #  inds = f.readlines()
        with open('./data/bdd100k/labels/ImageSets/day%s.txt'%'val', 'r') as f:
          inds = f.readlines()
                
        self.files_A = sorted(glob.glob('./data/CityScapes/leftImg8bit/%s/*/*.*'%'train') + glob.glob('./data/CityScapes/leftImg8bit/%s/*/*.*'%'val'))
        
        self.files_B  = sorted([os.path.join(BDD_root, i.strip()) for i in inds])

        print(mode)
        print(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B, 'A_path': self.files_A[index % len(self.files_A)], 'B_path': self.files_B[index % len(self.files_B)]}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
