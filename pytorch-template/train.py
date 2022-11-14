import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility # 재현성
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True # deterministic 연산만 수행
torch.backends.cudnn.benchmark = False # True면 cudnn이 자동튜너로 하드웨어 사용할 최적 알고리즘을 찾는다.
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train') # logging
    #train이 적히면 train level에서 log데이터를 얻고, test면 test level에서 log data를 얻는다
    #verbosity: 0=warning, 1=info, 2=debug

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data) 
    # config에서 'data_loader' 설정을 받아와서, moudledata(data_loader.data_loaders)에 넣어서 설정
    valid_data_loader = data_loader.split_validation() # valid dataloader 설정 --> base_dataloader에서

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    # config에서 'data_loader' 설정을 받아와서 model.arch(model.model)에 넣어서 설정
    logger.info(model) # logger의 목표를 model로

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # config와 이름이 같은 model 파일에 정의되어있는 loss와 metric을 설정
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters()) #requires_grad = True인 파라미터들 가져오기
    # config와 이름이 같은 torch.optim에 정의되어있는 optimizer, lr_scheduler를 설정
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    
    # 위의 설정들을 사용하여 실제로 훈련하는 trainer 정의
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train() # 아직 abstract


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template') # 호출 인자값 파싱
    # 인자값 등록
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)') # 설정 파일 주소
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)') # 마지막 checkpoint 주소(이전 훈련 이어서 시작?)
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)') # GPU, CPU 선ㅌ택

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options) # 인자(args)와 option에 대해서 config 객체 생성
    main(config)
