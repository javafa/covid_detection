import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image

def get_transform(resize=112, method=Image.BILINEAR):
    transform_list = []
    if resize > 0:
        size = [resize, resize]
        transform_list.append(transforms.Resize(size, method))

    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transform_list)

    # return transforms.Compose([
    #     transforms.Resize(size=(resize, resize), interpolation=method),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])

class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train', repeat_num = 10, transform=None,):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.repeat_num = repeat_num

        self.labels = {}
        
        self.label_path = os.path.join(root, self.phase, self.phase+'_label_COVID.txt')

        with open(self.label_path, 'r') as f:
            file_list = []
            label_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])
                if self.phase != 'test' :
                    label_list.append(v[1])
                    for i in range(33) :
                        temp = v[0].split('.')
                        aug = f'{temp[0]}_{str(i)}.png'
                        file_list.append(aug)
                        label_list.append(v[1])
                    
        self.labels['file'] = list(file_list)
        self.labels['label'] = list(label_list)

    def __getitem__(self, index):
        #if self.phase == 'train':
        if self.labels['file'][index].find('_') == -1 :
            image_path = os.path.join(self.root, self.phase, self.labels['file'][index])
        else :
            image_path = os.path.join(self.root, 'train_aug', self.labels['file'][index])

        if self.phase != 'test' :
            is_label = self.labels['label'][index]
            is_label = torch.tensor(int(is_label))

        transform = get_transform()
        image = Image.open(image_path).convert('RGB')
        image = transform(image)

        if self.phase != 'test' :
            return (self.labels['file'][index], image, is_label)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['file'][index], image, dummy)


    def __len__(self):
        return len(self.labels['file'])

    def get_label_file(self):
        return self.label_path


def data_loader(root, phase='train', batch_size=16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()