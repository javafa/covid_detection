import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

from dataloader import data_loader
from evaluation import evaluation_metrics
from model import Vgg19, BinaryClassifier

import random
import augment

'''
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
** 컨테이너 내 기본 제공 폴더
- /datasets : read only 폴더 (각 태스크를 위한 데이터셋 제공)
- /tf/notebooks :  read/write 폴더 (참가자가 Wirte 용도로 사용할 폴더)
1. 참가자는 /datasets 폴더에 주어진 데이터셋을 적절한 폴더(/tf/notebooks) 내에 복사/압축해제 등을 진행한 뒤 사용해야합니다.
   예시> Jpyter Notebook 환경에서 압축 해제 예시 : !bash -c "unzip /datasets/objstrgzip/10_classification_COVID.zip -d /tf/notebooks/"
   예시> Terminal(Vs Code) 환경에서 압축 해제 예시 : bash -c "unzip /datasets/objstrgzip/10_classification_COVID.zip -d /tf/notebooks/10_covid/data"
   
2. 참가자는 각 문제별로 데이터를 로드하기 위해 적절한 path를 코드에 입력해야합니다. (main.py 참조)
3. 참가자는 모델의 결과 파일(Ex> prediction.txt)을 write가 가능한 폴더에 저장되도록 적절 한 path를 입력해야합니다. (main.py 참조)
4. 세션/컨테이너 등 재시작시 위에 명시된 폴더(datasets, notebooks) 외에는 삭제될 수 있으니 
   참가자는 적절한 폴더에 Dataset, Source code, 결과 파일 등을 저장한 뒤 활용해야합니다.
   
!!!!!!!!!!!!!!!!!!!!! 필독!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''
# try:
#     from nipa import nipa_data
#     DATASET_PATH = nipa_data.get_data_root('COVID')
# except:
DATASET_PATH = os.path.join('./data')


def _infer(model, cuda, data_loader):
    res_fc = None
    res_id = None
    for index, (image_name, image, _) in enumerate(data_loader):
        if cuda :
            image = image.cuda()
        fc = model(image)
        fc = fc.detach().cpu().numpy()

        if index == 0:
            res_fc = fc
            res_id = image_name
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)
            res_id = res_id + image_name

    res_cls = np.argmax(res_fc, axis=1)
    #print('res_cls{}\n{}'.format(res_cls.shape, res_cls))

    return [res_id, res_cls]


def feed_infer(output_file, infer_func):
    prediction_name, prediction_class = infer_func()

    print('write output')
    predictions_str = []
    for index, name in enumerate(prediction_name):
        test_str = name + ' ' + str(prediction_class[index])
        predictions_str.append(test_str) 
    with open(output_file, 'w') as file_writer:
        file_writer.write("\n".join(predictions_str))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def validate(prediction_file, model, validate_dataloader, validate_label_file, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=validate_dataloader))
    metric_result = evaluation_metrics(prediction_file, validate_label_file)
    print("-------------------------------------------------")
    print('Eval result: {:.4f}'.format(metric_result))
    print("-------------------------------------------------")
    return metric_result


def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(epochname, dir_name, model, optimizer, metric_result, train_loss):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        # 'scheduler': scheduler.state_dict()
    }
    modelname = f'{dir_name}_{epochname}_{metric_result}_{train_loss}.pth'
    SAVE_PATH = os.path.join('./pth')
    torch.save(state,  os.path.join(SAVE_PATH, modelname))
    print('model saved: ', modelname)


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def setRandomSeed(seed) :
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

if __name__ == '__main__':

    setRandomSeed(77)

    img_root = "./data/train/"
    aug_dir = "./data/train_aug/"

    if not os.path.exists(aug_dir):
        augment.run(img_root, aug_dir)
    
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=2)
    args.add_argument("--lr", type=float, default=0.0001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=50)
    args.add_argument("--print_iter", type=int, default=600)
    args.add_argument("--dir_name", type=str, default="10_covid") 
    args.add_argument("--model_name", type=str, default="pth/10_covid_4_0.81_0.10653.pth") 
    args.add_argument("--prediction_file", type=str, default="tr_prediction.txt")
    args.add_argument("--batch", type=int, default=4)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    dir_name = config.dir_name
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode

    # create model
    # model = BinaryClassifier()
    model = Vgg19(num_classes=num_classes)

    # 전이학습으로 대체시 위에 주석
    # load_model(model_name, model)

    if cuda:
        model = model.cuda()

    # test in each epoch
    # test_dataloader, _ = data_loader(root=DATASET_PATH, phase='test', batch_size=1)

    if mode == 'train':
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        if cuda:
            loss_fn = loss_fn.cuda()

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)
        # learning decay
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

        # get data loader
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        validate_dataloader, validate_label_file = data_loader(root=DATASET_PATH, phase='validate', batch_size=batch)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)
        #print("num batches : ", num_batches)

        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ",total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :",trainable_params)
        print("------------------------------------------------------------")

        # train
        for epoch in range(num_epochs):
            model.train()
            
            train_losses = 0
            for iter_, data in enumerate(train_dataloader):
                # fetch train data
                _, image, is_label = data 
                if cuda:
                    image = image.cuda()
                    is_label = is_label.cuda() 

                # update weight
                pred = model(image)
                loss = loss_fn(pred, is_label)
                
                optimizer.zero_grad()
                loss.backward()
                train_losses += loss.item()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()

            # scheduler update
            scheduler.step()

        
            # validate
            metric_result = validate(prediction_file, model, validate_dataloader, validate_label_file, cuda)

             # save model 
            train_loss = round(train_losses/len(train_dataloader) , 5)
            print(f'[loss avg. {train_loss}]')
            metric_result = round(metric_result, 5)
            save_model(str(epoch + 1),dir_name, model, optimizer, metric_result, train_loss)

            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

