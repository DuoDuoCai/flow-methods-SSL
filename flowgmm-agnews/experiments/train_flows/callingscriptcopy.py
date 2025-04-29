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

'''
made the trainer: <class 'flow_ssl.realnvp.realnvp.RealNVPTabular'> number of classes: <flow_ssl.data.nlp_datasets.AG_News object at 0x000001D386A8F4C0> dim_in: 768 other configs: {'k': 1024, 'coupling_layers': 7, 'nperlayer': 1}
my trainer configs: 
train:   0%|                                                                                                                                    | 0/100 [00:00<?, ?it/s]   Minibatch_Loss
'''

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

    # print("Defined in module:", dataset().__module__)
    # print(inspect.getsource(dataset()))
    # sys.exit()
    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(0):
        # print("---------------------------------------------")
        # print(type(dataset()))  # <class 'flow_ssl.data.nlp_datasets.AG_News'>
        # print(dataset)
        # print("---------------------------------------------")
        original_data = dataset()
        # print("enter make trainer, the dataset:", type(dataset))    # flow_ssl.data.nlp_datasets.AG_News object
        # print(original_data)    # <flow_ssl.data.nlp_datasets.AG_News object at 0x0000022B620CF400>
        # print("original data length:", len(original_data))

        # TODO here to split the dataset
        print("original len:", len(original_data))
        datasets = split_dataset(original_data,splits={'train':2000 , 'val': 10, '_unlab_w_label': 10000})
        
        # print("type of original dataset:", type(original_data))     # flow_ssl.data.nlp_datasets.AG_News
        # print("type of  datasets:", type(datasets))     # 'dict'
        # print("type of value of  datasets:", type(datasets['val']))     # oil.datasetup.datasets.IndexedDataset
        # sys.exit()
        # calling the original dataset again
        # my_data= dataset()

        # datasets['Train'] = ConcatDataset([datasets['train'], datasets['_unlab_w_label']])  # not used for now

        datasets['_unlab'] = dmap(lambda mb: mb[0], datasets['_unlab_w_label'])
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(len(datasets['train']))
        # print(len(datasets['val']))
        # print(len(datasets['_unlab']))
        # print(len(datasets['Train']))
        # print("some _unlab data:",datasets['_unlab'] ,type(datasets['_unlab'] ))
        # see_dataloader(dataloaders['_unlab'])
        # for i, minibatch in enumerate(datasets['_unlab']):
        #     print(minibatch)
        #     break
        # print("the first unlabeled data:", datasets['_unlab'][0])
        # print("the 101th unlabeled data:", datasets['_unlab'][100])
        # print("the whole:", datasets['_unlab'][700:900])
        # sys.exit()

        datasets['test'] = dataset(train=False)
        #print(datasets['test'][0])

    device = torch.device("cpu")
    print("network var:", network)  # RealNVPTabularWPrior
    # model is flow_ssl.realnvp.realnvp.RealNVPTabular
    model = network(num_classes=datasets['train'].num_classes,dim_in=datasets['train'].dim,**net_config).to(device)
    print(type(model), "is type of model var")
    
    # dim_in: 768. {'k': 1024, 'coupling_layers': 7, 'nperlayer': 1}

    print(type(model.prior))
    print((model.prior))
    
    # sys.exit()
    print("made the trainer:", type(model), "number of classes:", datasets['train'], "dim_in:", datasets['train'].dim, "other configs:", net_config)
    
    # print("the dataloader is like")
    # train, val, _unlab, test
    # for k,v in datasets.items():
    #    print(k)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,len(datasets[k])),shuffle=(k=='train'), num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['pure_train'] = dataloaders['train']



    # print("some training data:")
    
    # see_dataloader(dataloaders['train'])
        
        
    # print("some validation data:", dataloaders['val'])
    # # see_dataloader(dataloaders['val'])
    # for i, minibatch in enumerate(dataloaders['val']):
    #     print(minibatch)
    #     break

    # print("some _unlab data:", dataloaders['_unlab'])
    # see_dataloader(dataloaders['_unlab'])
    # for i, minibatch in enumerate(dataloaders['_unlab']):
    #     print(minibatch)
    #     break
        

    # print("some test data:", dataloaders['test'])
    # see_dataloader(dataloaders['test'])
    # sys.exit()





    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)#lambda e:1

    print("my trainer configs:")
    print("opt_constr: ", opt_constr) #  functools.partial(<class 'torch.optim.adamw.AdamW'>, lr=0.0003, weight_decay=1e-05)
    print("lr_sched: ", lr_sched)   #  <function cosLr.<locals>.lrSched at 0x00000289BEF955E0>
    print("other configs:", trainer_config) # {'log_dir': 'C:\\Users\\Hardy/tb-experiments/UCI/', 'log_args': {'minPeriod': 0.1, 'timeFrac': 0.3}, 'unlab_weight': 0.6}

    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

# tabularTrial = train_trial(makeTrainer)

if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    cfg = argupdated_config(defaults,namespace=(tabular_datasets,flows,archs,trainers))
    # cfg.pop('local_rank')
    # print("configs")
    # print(cfg)
    '''
    {'dataset': <class 'flow_ssl.data.nlp_datasets.AG_News'>, 'network': <function RealNVPTabularWPrior at 0x000001D835F471F0>, 'num_epochs': 1, 'bs': 5000, 'lr': 0.0003, 'optim': <class 'torch.optim.adamw.AdamW'>, 'device': 'cuda', 'trainer': SemiFlow, 'split': {'train': 200, 'val': 5000}, 'net_config': {'k': 1024, 'coupling_layers': 7, 'nperlayer': 1}, 'opt_config': {'weight_decay': 1e-05}, 'trainer_config': {'log_dir': 'C:\\Users\\Hardy/tb-experiments/UCI/', 'log_args': {'minPeriod': 0.1, 'timeFrac': 0.3}, 'unlab_weight': 0.6}, 'save': False}
    '''
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
    


'''
some training data:
Part 0: shape torch.Size([1000, 768])
Part 1: shape torch.Size([1000])
some validation data: <torch.utils.data.dataloader.DataLoader object at 0x000001A495599430>
Part 0: shape torch.Size([16, 768])
Part 1: shape torch.Size([16])
some _unlab data: <torch.utils.data.dataloader.DataLoader object at 0x000001A4955995E0>
Single tensor: shape torch.Size([1016, 768])
some test data: <torch.utils.data.dataloader.DataLoader object at 0x000001A495599730>
Part 0: shape torch.Size([24, 768])
Part 1: shape torch.Size([24])

Part 0: shape torch.Size([1000, 768])
Part 1: shape torch.Size([1000])
some validation data: <torch.utils.data.dataloader.DataLoader object at 0x000001E85B2E9460>
Part 0: shape torch.Size([16, 768])
Part 1: shape torch.Size([16])
some _unlab data: <torch.utils.data.dataloader.DataLoader object at 0x000001E85B2E9610>
Single tensor: shape torch.Size([1016, 768])
some test data: <torch.utils.data.dataloader.DataLoader object at 0x000001E85B2E9760>
Part 0: shape torch.Size([24, 768])
Part 1: shape torch.Size([24])
'''