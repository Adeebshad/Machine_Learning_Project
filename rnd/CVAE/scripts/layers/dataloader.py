import torch,os,cv2,csv
import pandas as pd
import numpy as np
from skimage import io, transform
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF



class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        
        self.attributes_frame = pd.read_csv(csv_file,delimiter=' ')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.attributes_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.attributes_frame.iloc[idx, 0]+".jpg")
        image = io.imread(img_name)

        attributes = self.attributes_frame.iloc[idx, 1:]
        attributes = np.array([attributes])
        attributes = attributes.astype('float').reshape(-1, 1)
        sample = {'image': image, 'attributes': attributes}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ToTensor(object):
    def __call__(self, sample):
        image, attributes = sample['image'], sample['attributes']
        norm_image = cv2.normalize(image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #image = image.transpose((2, 0, 1))
#         attributes = TF.normalize(attributes, mean, var)
        im = torch.from_numpy(norm_image).float()
        ln = torch.from_numpy(attributes).float()
        return {'image':im,
                'attributes':ln }
    
    
# face_dataset = CustomDataset(csv_file=attrfile,
#                                     root_dir=imgpath)

# transformed_dataset = CustomDataset(csv_file=attrfile,
#                                            root_dir=imgpath,
#                                            transform=transforms.Compose([
#                                                ToTensor()
#                                            ]))

# train_iterator = DataLoader(transformed_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_iterator = DataLoader(transformed_dataset, batch_size=BATCH_SIZE)
    

