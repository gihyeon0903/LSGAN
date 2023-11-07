import cv2
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class CelebA_Datasets(Dataset):
      def __init__(self, data_type):
            # data_type= [Smile, NonSmile, Male, Female] -> [0, 1, 2, 3]
            celebA_attr_path = '../DCGAN/celebA_data/atrributes/list_attr_celeba.csv'
            celebA_attr = pd.read_csv(celebA_attr_path)
            
            filtered_data = self.data_extract_from_csv(data_type, celebA_attr)
            
            self.pwd_folder = '../DCGAN/CelebA_data/images/'
            self.img_path = filtered_data[['File_name']].to_numpy().reshape(-1)

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((64, 64), antialias=True),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
  

      def data_extract_from_csv(self, data_type, celebA_attr):
            if data_type == 'Man':
                  filtered_data = celebA_attr.loc[celebA_attr.Male == 1, ]
            elif data_type == 'Woman':
                  filtered_data = celebA_attr.loc[celebA_attr.Male == -1, ]
            return filtered_data


      def __getitem__(self, idx):
            img_path = self.pwd_folder + self.img_path[idx]
            img = cv2.imread(img_path.replace('jpg', 'png'))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.transform(img)

      def __len__(self):
            return len(self.img_path)