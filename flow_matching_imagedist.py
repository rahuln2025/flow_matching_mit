from abc import ABC, abstractmethod
from typing import Optional, List, Type, Tuple, Dict
import math

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.axes._axes import Axes
import torch
import torch.distributions as D
from torch.func import vmap, jacrev
from tqdm import tqdm
import seaborn as sns
from sklearn.datasets import make_moons, make_circles


from src.distributions import *
from src.paths import *
from image_to_dist import *
from src.utils_plot import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import yaml



def parse_args():
    parser = argparse.ArgumentParser(description='Script to convert image to dist of points')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    return parser.parse_args()

def main(config_path):

    # Load configuration from YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # get points from the image (png)
    points_array = image_to_dist(config_path)

    # create distribution from the points
    p_data = ImageSampleable(device, points_file = None, points_array=points_array, scale = 5.0)

    # get initial distribution (Gaussian)
    p_simple = Gaussian.isotropic(dim = 2, std = 1.0)

    # Construct linear conditional probability path
    path = LinearConditionalProbabilityPath(
        p_simple = p_simple, 
        p_data = p_data
    ).to(device)

    # construct learnable vector field using simple MLP
    bridging_flow_model = MLPVectorField(dim = 2, hiddens=[100, 100, 100, 100, 100, 100, 100])

    # construct trainer
    trainer = ConditionalFlowMatchingTrainer(path, bridging_flow_model)
    num_epochs = config.get('training').get('epochs')
    losses = trainer.train(num_epochs = num_epochs, 
                           device = device, 
                           lr = float(config.get('training').get('learning_rate')), 
                           batch_size = config.get('training').get('batch_size'))
    
    if config.get('training').get('plot_losses') == True:
        plt.figure(figsize=(8, 6))
        epochs = np.linspace(0, len(losses), len(losses))
        plt.plot(epochs, losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.savefig('loss_curve.png')
        plt.close()

    # plot the marginals in the true and learned paths

    num_samples = config.get('output').get('num_samples')
    num_marginals = config.get('output').get('num_marginals')

    ##############
    # Setup Plots #
    ##############

    fig, axes = plt.subplots(2, num_marginals, figsize=(6 * num_marginals, 6 * 2))
    axes = axes.reshape(2, num_marginals)
    scale = 5.0

    ###########################
    # Graph Ground-Truth Marginals #
    ###########################
    ts = torch.linspace(0.0, 1.0, num_marginals).to(device)
    for idx, t in enumerate(ts):
        tt = t.view(1,1).expand(num_samples,1)
        xts = path.sample_marginal_path(tt)
        #print(xts)
        hist2d_samples(samples=xts.cpu(), ax=axes[0, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
        axes[0, idx].set_xlim(-scale, scale)
        axes[0, idx].set_ylim(-scale, scale)
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])
        axes[0, idx].set_title(f'$t={t.item():.2f}$', fontsize=15)
    axes[0, 0].set_ylabel("Ground Truth", fontsize=20)

    ###############################################
    # Graph Learned Marginals #
    ###############################################
    ode = LearnedVectorFieldODE(bridging_flow_model)
    simulator = EulerSimulator(ode)
    ts = torch.linspace(0,1,200).to(device)
    record_every_idxs = record_every(len(ts), len(ts) // (num_marginals - 1))
    x0 = path.p_simple.sample(num_samples)
    xts = simulator.simulate_with_trajectory(x0, ts.view(1,-1,1).expand(num_samples,-1,1))
    xts = xts[:,record_every_idxs,:]
    for idx in range(min(xts.shape[1], num_marginals)):
        xx = xts[:,idx,:]
        hist2d_samples(samples=xx.cpu(), ax=axes[1, idx], bins=200, scale=scale, percentile=99, alpha=1.0)
        axes[1, idx].set_xlim(-scale, scale)
        axes[1, idx].set_ylim(-scale, scale)
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        tt = ts[record_every_idxs[idx]]
        axes[1, idx].set_title(f'$t={tt.item():.2f}$', fontsize=15)
    axes[1, 0].set_ylabel("Learned", fontsize=20)

    plt.savefig('learned_marginals.png')

if __name__ == "__main__":
    args = parse_args()
    main(args.config)
