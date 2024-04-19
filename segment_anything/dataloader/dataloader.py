import cv2
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MedicalDataset(Dataset):
    def __init__(self,
                 root: str = 'dataset', 
                 mod1: str = 'CT', 
                 mod2: str = 'MR-T2'):
        super().__init__()
        self.root = root
        self.classes = os.listdir(root)
        self.dataset = []
        self.label = []

        for label in tqdm(os.listdir(root)):
            label_path = os.path.join(root, label)
            for instance in os.listdir(label_path):
                instance_path = os.path.join(label_path, instance)
                instance = {}
                for mode in os.listdir(instance_path):
                    instance.setdefault(mode, [])
                    mode_path = os.path.join(instance_path, mode)
                    for file_name in os.listdir(mode_path):

                        file_path = os.path.join(mode_path, file_name)
                        cap = cv2.VideoCapture(file_path)
        
                        frame_list = []
                        # 读取视频的每一帧并添加到帧列表中
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_list.append(frame)

                        # 关闭视频文件
                        cap.release()
                        instance[mode].append(frame_list)
                    instance['label'] = label
                self.dataset.append(instance)

        self.data = []
        self.label_map = {}
        for idx, instance in enumerate(self.dataset):
            instance_mod1 = instance.get(mod1, None)
            instance_mod2 = instance.get(mod2, None)
            if not (instance_mod1 and instance_mod2):
                continue
            assert instance.get('label', None), 'Failed to find label'
            if not self.label_map.get(instance['label'], None):
                self.label_map[instance['label']] = len(self.label_map)
            # make combination of mode from the same class
            for i in range(len(instance_mod1)):
                for j in range(len(instance_mod2)):
                    self.data.extend((x, y) for x, y in zip(instance_mod1[i], instance_mod2[j]))
                    self.label.extend((self.label_map[instance['label']], self.label_map[instance['label']]) for _ in instance_mod1[i])

            #TODO: make combination of mode from different class
        print('Catagory: ', self.label_map)
        self.num_classes = len(self.label_map)

    def __getitem__(self, index):
        modal_images = self.data[index]
        labels = self.label[index]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256))
        ])
        modal_images = [transform(img) for img in modal_images]
        return modal_images, labels
    
    def __len__(self):
        return len(self.label)