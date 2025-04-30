# Flow Gaussian Mixture Model (FlowGMM)

This project is a clone of https://github.com/izmailovpavel/flowgmm/tree/public , a PyTorch implementation of the Flow Gaussian Mixture Model (FlowGMM) model for paper:

[Semi-Supervised Learning with Normalizing Flows ](https://arxiv.org/abs/1912.13025)

by Pavel Izmailov, Polina Kirichenko, Marc Finzi and Andrew Gordon Wilson.

This project runs the AG News classification.


# preparation
To run the scripts you will need to clone the repo and install it locally. You can use the commands below.
```bash
git clone XXX
cd flow-methods-SSL/flowgmm-agnews
pip install -e .
```
Also, please download the .npz file training data (https://drive.google.com/file/d/13SVwGLHLpA6tnuzZkQiGbZ0cDul3DSOK/view?usp=sharing) and put it inside flowgmm-agnews/data/nlp_datasets.

## Dependencies
We have the following dependencies for FlowGMM that must be installed prior to install to FlowGMM
* Python 3.7+
* [PyTorch](http://pytorch.org/) version 1.0.1+
* [torchvision](https://github.com/pytorch/vision/) version 0.2.1+
* [tensorboardX](https://github.com/lanpa/tensorboardX)


## Text Classification
Train **FlowGMM** on AG-News (2000 labeled examples):
```bash
python experiments/train_flows/flowgmm_tabular_new.py --trainer_config "{'unlab_weight':.6}" --net_config "{'k':1024,'coupling_layers':7,'nperlayer':1}" --network RealNVPTabularWPrior --trainer SemiFlow --num_epochs 200  --dataset AG_News --lr 3e-4 
```







# References
RealNVP: [github.com/chrischute/real-nvp](https://github.com/chrischute/real-nvp)
