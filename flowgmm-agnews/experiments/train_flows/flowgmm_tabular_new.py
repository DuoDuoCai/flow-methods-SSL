import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.model_trainers.piModel import PiModel
from oil.model_trainers.vat import Vat
from oil.datasetup.datasets import CIFAR10
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice, imap
from oil.tuning.study import Study, train_trial
from flow_ssl.data.nlp_datasets import AG_News,YAHOO
from flow_ssl.data import GAS, HEPMASS, MINIBOONE
from torchvision import transforms
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from oil.utils.utils import Expression,export,Named
from collections import defaultdict
from oil.model_trainers.piModel import PiModel
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam,AdamW
from oil.utils.utils import LoaderTo, cosLr, islice, dmap, FixedNumpySeed
from oil.tuning.study import train_trial
from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.tuning.args import argupdated_config
from functools import partial
from train_semisup_text_baselines import SmallNN
from oil.tuning.args import argupdated_config
import copy
#import flow_ssl.data.nlp_datasets as nlp_datasets
import flow_ssl.data as tabular_datasets
import train_semisup_flowgmm_tabular as flows
import train_semisup_text_baselines as archs
import oil.model_trainers as trainers
import sys
import inspect
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset


def see_dataloader(loader):
    batch = next(iter(loader))
    if isinstance(batch, (tuple, list)):
        for i, item in enumerate(batch):
            print(f"Part {i}: shape {getattr(item, 'shape', 'unknown')}")
    else:
        print(f"Single tensor: shape {batch.shape}")



def makeTrainer(*,dataset=HEPMASS,network=SmallNN,num_epochs=15,
                bs=5000,lr=1e-3,optim=AdamW,device='cuda',trainer=Classifier,
                split={'train': 1000, 'val': 16},net_config={},opt_config={'weight_decay':1e-5},
                trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
                save=False):
    dataset = AG_News

    with FixedNumpySeed(0):
        original_data = dataset()
        print("original len:", len(original_data))
        datasets = split_dataset(original_data,splits={'train':2000 , 'val': 10, '_unlab_w_label': 8000})
        datasets['_unlab'] = dmap(lambda mb: mb[0], datasets['_unlab_w_label'])
        datasets['test'] = dataset(train=False)

    device = torch.device("cpu")
    print("network var:", network)  # RealNVPTabularWPrior
    # model is flow_ssl.realnvp.realnvp.RealNVPTabular
    model = network(num_classes=datasets['train'].num_classes,dim_in=datasets['train'].dim,**net_config).to(device)
    print(type(model), "is type of model var")    
    print(type(model.prior))
    print((model.prior))    
    print("made the trainer:", type(model), "number of classes:", datasets['train'], "dim_in:", datasets['train'].dim, "other configs:", net_config)
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(datasets[k])),shuffle=(k=='train'), num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['pure_train'] = dataloaders['train']

    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)#lambda e:1

    print("my trainer configs:")
    print("opt_constr: ", opt_constr) #  functools.partial(<class 'torch.optim.adamw.AdamW'>, lr=0.0003, weight_decay=1e-05)
    print("lr_sched: ", lr_sched)   #  <function cosLr.<locals>.lrSched at 0x00000289BEF955E0>
    print("other configs:", trainer_config) # {'log_dir': 'C:\\Users\\Hardy/tb-experiments/UCI/', 'log_args': {'minPeriod': 0.1, 'timeFrac': 0.3}, 'unlab_weight': 0.6}
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)


if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg = argupdated_config(defaults,namespace=(tabular_datasets,flows,archs,trainers))
    trainer = makeTrainer(**cfg)
    trainer.train(cfg['num_epochs'])

    print("----------------------training acc--------------------------")
    print(trainer.labeled_train_accs)
    print(trainer.unlabeled_train_accs)
    print("----------------------validation acc--------------------------")
    print(trainer.val_accs)

    epochs = range(len(trainer.val_accs))  # X-axis: epochs or steps

    plt.plot(epochs, trainer.labeled_train_accs, label='Labeled Train Accuracy')
    plt.plot(epochs, trainer.unlabeled_train_accs, label='Unlabeled Train Accuracy')
    plt.plot(epochs, trainer.val_accs, label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Time')
    plt.legend()
    plt.grid(True)
    plt.show()