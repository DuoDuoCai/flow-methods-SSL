import torch
torch.set_printoptions(sci_mode=False)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap, export
#from .schedules import sigmoidConsRamp
import flow_ssl
import utils
from flow_ssl import FlowLoss
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.distributions import SSLGaussMixture
from scipy.spatial.distance import cdist
import sys
import functools
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import matplotlib.patches as mpatches
import plotly.graph_objs as go
import plotly.io as pio
# def plot_clusters(X, y_true, y_pred, uniq):
#     tsne = TSNE(n_components=3, learning_rate='auto', random_state=42)
#     X = X.detach().cpu().numpy()
#     y_true = y_true.detach().cpu().numpy()
#     y_pred = y_pred.detach().cpu().numpy()

#     X3 = tsne.fit_transform(X)  # perform t-SNE!

#     true_colors = np.array(['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
#     pred_colors = np.array(['navy', 'darkorange', 'darkgreen', 'darkred'])

#     label_names = ['world', 'sport', 'business', 'sci/tech']  # <--- your class names

#     facecolors = true_colors[y_true]  # center = ground truth
#     edgecolors = pred_colors[y_pred]  # edge = prediction

#     fig = plt.figure(figsize=(10,8))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.scatter(
#         X3[:,0], X3[:,1], X3[:,2],
#         c=facecolors,
#         edgecolors=edgecolors,
#         s=30,
#         linewidths=1.5,
#         alpha=1
#     )

#     ax.set_title(f'3D t-SNE: True vs Predicted on {len(y_true)} Samples')
#     ax.set_xlabel('t-SNE 1')
#     ax.set_ylabel('t-SNE 2')
#     ax.set_zlabel('t-SNE 3')

#     # Build custom legends with label names
#     true_patches = [mpatches.Patch(color=color, label=name) for color, name in zip(true_colors, label_names)]
#     # pred_patches = [mpatches.Patch(edgecolor=color, facecolor='white', label=f'Pred {i}', linewidth=2) for i, color in enumerate(pred_colors)]
#     pred_patches = [mpatches.Patch(edgecolor=color, facecolor='white', label='Pred ' + name, linewidth=2) for color, name in zip(pred_colors, label_names)]

#     legend1 = ax.legend(handles=true_patches, title="Ground Truth", loc='upper left', bbox_to_anchor=(1.05, 1))
#     legend2 = ax.legend(handles=pred_patches, title="Prediction", loc='lower left', bbox_to_anchor=(1.05, 0))
#     ax.add_artist(legend1)  # Add first legend manually

#     plt.tight_layout()
#     plt.savefig("C:\\Users\\Hardy\\Desktop\\flowgmm-public\\plots\\"+uniq+".png")

def plot_clusters_interactive(X, y_true, y_pred, uniq):
    # 1) run t-SNE to 3 dims
    X_np      = X.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    X3        = TSNE(n_components=3, learning_rate='auto', random_state=42) \
                  .fit_transform(X_np)

    # 2) define your colors & labels
    # true_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    true_colors = ['blue', 'orange', 'green', 'red']
    pred_colors = ['navy',      'darkorange',  'darkgreen',  'darkred']
    label_names = ['world', 'sport', 'business', 'sci/tech']

    # 3) build one Plotly trace per true class
    traces = []
    for i, name in enumerate(label_names):
        idxs = np.where(y_true_np == i)[0]
        traces.append(
            go.Scatter3d(
                x = X3[idxs,0],
                y = X3[idxs,1],
                z = X3[idxs,2],
                mode = 'markers',
                name = name,                 # legend entry
                marker = dict(
                    size        = 4,
                    color       = true_colors[i],      # fill
                    line        = dict(
                        color = pred_colors,           # this will broadcast per-trace
                        width = 1.5
                    ),
                    opacity     = 1.0
                ),
                # to get per-point edge colors for predictions, you could
                # instead build per-(true,pred) traces or use `marker.line.color`
                # as an array—but this simpler version shows true-class grouping.
            )
        )

    # 4) layout & combine
    layout = go.Layout(
        title = f'3D t-SNE: True vs Predicted ({len(y_true_np)} samples)',
        scene = dict(
            xaxis = dict(title='t-SNE 1'),
            yaxis = dict(title='t-SNE 2'),
            zaxis = dict(title='t-SNE 3'),
        ),
        margin = dict(l=0, r=0, b=0, t=50),
        legend = dict(x=1.05, y=1)
    )

    fig = go.Figure(data=traces, layout=layout)

    # 5) write to a standalone HTML file
    out_path = f"C:/Users/Hardy/Desktop/flowgmm-public/plots/{uniq}.html"
    pio.write_html(fig, file=out_path, auto_open=False)
    print(f"Interactive plot saved to {out_path}")
def acc_func(trainer,uniq,mb):
    predictions =trainer.model.prior.classify(trainer.model(mb[0])).type_as(mb[1]) 
    # plot_clusters_interactive(mb[0], mb[1], predictions, uniq)
    return predictions.eq(mb[1]).cpu().data.numpy().mean()
    
'''
tensor([-0.8179, -0.6017, -0.8693,  0.6344,  0.7255, -0.3551,  0.1519,  0.4408,
        -0.6616, -1.0000, -0.4618,  0.9513,  0.9892,  0.3473,  0.8191, -0.3826,
        -0.2501, -0.4836,  0.4754,  0.4596,  0.6990,  1.0000,  0.0543,  0.3161,
         0.4839,  0.9784, -0.6684,  0.8960,  0.9510,  0.7237, -0.4327,  0.4682,


tensor([    -0.5853,     -0.3808,     -0.6979,      0.2947,      0.4324,
            -0.1369,      0.0371,      0.3992,     -0.3322,     -0.9995,
            -0.1395,      0.6087,      0.9590,      0.2859,      0.6735,
            -0.1552,      0.3537,     -0.4794,      0.2584,      0.6816,
             0.5624,      0.9999,      0.3386,      0.3465,      0.3788,
'''
@export
def RealNVPTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=.8,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cpu')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = RealNVPTabular(num_coupling_layers=coupling_layers,in_dim=dim_in,hidden_dim=k,num_layers=1,dropout=True)#*np.sqrt(1000/dim_in)/3
    #dist_scaling = np.sqrt(-8*np.log(1-acc))#np.sqrt(4*np.log(20)/dim_in)#np.sqrt(1000/dim_in)
    if num_classes ==2:
        means = utils.get_means('random',r=means_r,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        #means = torch.zeros(2,dim_in,device=device)
        #means[0,1] = 3.75
        dist = 2*(means[0]**2).sum().sqrt()
        means[0] *= 7.5/dist
        means[1] = -means[0]
        # means[0] /= means[0].norm()
        # means[0] *= dist_scaling/2
        # means[1] = - means[0]
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    else:
        means = utils.get_means('random',r=means_r*.7,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))
    return model

def ResidualTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=1.,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cpu')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = TabularResidualFlow(in_dim=dim_in,hidden_dim=k,num_per_block=coupling_layers)#*np.sqrt(1000/dim_in)/3
    dist_scaling = np.sqrt(-8*np.log(1-acc))
    means = utils.get_means('random',r=means_r*dist_scaling,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
    means[0] /= means[0].norm()
    means[0] *= dist_scaling/2
    means[1] = - means[0]
    model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
    means_np = means.cpu().numpy()
    #print("Pairwise dists:", cdist(means_np, means_np))
    return model

@export
class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,cons_weight=3.,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight,'cons_weight':cons_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])
        self.val_accs=[]
        self.labeled_train_accs=[]
        self.unlabeled_train_accs=[]
        self.test_accs=[]


    def loss(self, minibatch):
        # print("in computing loss, minibatch")
        # x_lab: [1000, 768], y_lab: [1000], x_unlab: [1016, 768]
        (x_lab, y_lab), x_unlab = minibatch

        # in the second setup, select 1000 from unlab, and remember to change the argument to "NLL"
        # print("second argument:", x_unlab.shape[0])
        idxes = torch.randint(0, x_unlab.shape[0], (1000,))   
        x_unlab=x_unlab[idxes]
        # x_lab_unlab=x_lab.clone()

        a = float(self.hypers['unlab_weight'])
        b = float(self.hypers['cons_weight'])

        # create a copy of 
        # not sure about this
        # print("the type of self.model for the traininer:", type(self.model))
        # cyrrebt tghe best is 
        flow_loss = self.model.nll(x_lab,y_lab).mean() + a*self.model.nll(x_unlab).mean()
        # +0.5*self.model.nll(x_unlab).mean()
        # print("after the loss computation")
        # sys.exit()
        
        # perturb the x_lab
        # sigma=0.01
        # noise=torch.rand_like(x_lab)* sigma
        # x_lab_noise= x_lab+noise
        
        # with torch.no_grad():
        #     predict_label = self.model.prior.classify(self.model(x_lab)).detach()
        # cons_loss = self.model.nll(x_lab_noise,predict_label).mean()
        return flow_loss#+b*cons_loss
    
    def step(self, minibatch):
        self.optimizer.zero_grad()
        loss = self.loss(minibatch)
        loss.backward()
        utils.clip_grad_norm(self.optimizer, 100)
        self.optimizer.step()
        return loss
        

    def logStuff(self, step, minibatch=None):
        bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
        # acc_func = lambda mb: self.model.prior.classify(self.model(mb[0])).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model), torch.no_grad():
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
            # metrics['val_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['val']),bpd_func)
            # metrics['labeled_train_acc'] = self.evalAverageMetrics(self.dataloaders['pure_train'],acc_func)
            metrics['labeled_train_acc'] = self.evalAverageMetrics(
                    self.dataloaders['pure_train'],
                    functools.partial(acc_func, self,"labeled_train_at_"+str(step))
            )
            # metrics['unlabeled_train_acc'] = self.evalAverageMetrics(self.dataloaders['_unlab_w_label'],acc_func)
            metrics['unlabeled_train_acc'] = self.evalAverageMetrics(
                    self.dataloaders['_unlab_w_label'],
                    functools.partial(acc_func, self,"unlabeled_train_at_"+str(step))
            )
            
            # metrics['val_acc'] = self.evalAverageMetrics(self.dataloaders['val'],acc_func)
            # metrics['val_acc'] = self.evalAverageMetrics(
            #         self.dataloaders['val'],
            #         functools.partial(acc_func, self,"validation_at_"+str(step))
            # )
            # metrics['test_acc'] = self.evalAverageMetrics(self.dataloaders['test'],acc_func)
            metrics['test_acc'] = self.evalAverageMetrics(
                    self.dataloaders['test'],
                    functools.partial(acc_func, self,"validation_at_"+str(step))
            )
            if minibatch:
                metrics['Unlab_loss(mb)']=self.model.nll(minibatch[1]).mean().cpu().data.numpy()
            
            self.val_accs.append(metrics['test_acc'])
            self.labeled_train_accs.append(metrics['labeled_train_acc'])
            self.unlabeled_train_accs.append(metrics['unlabeled_train_acc'])

        self.logger.add_scalars('metrics',metrics,step)
        super().logStuff(step, minibatch)



# from oil.tuning.study import Study, train_trialng 
import collections
import os
import utils
import copy

#from train_semisup_text_baselines import makeTabularTrainer
#from flowgmm_tabular_new import tabularTrial

from flow_ssl.data.nlp_datasets import AG_News
from flow_ssl.data import GAS, HEPMASS, MINIBOONE

# if __name__=="__main__":
#     trial(uci_hepmass_flowgmm_cfg)
    # thestudy = Study(trial,uci_hepmass_flowgmm_cfg,study_name='uci_flowgmm_hypers222_m__m_m')
    # thestudy.run(1,ordered=False)
    # covars = thestudy.covariates()
    # covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
    # covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
    #print(covars.drop(['log_suffix','saved_at'],axis=1))
    # print(thestudy.covariates())
    # print(thestudy.outcomes)

