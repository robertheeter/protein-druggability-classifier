'''
NOTES

02/29/2024
- 35808884: radius_pocket = 4, expand_pocket_by_res = False, epochs = 100
- 35818916: radius_pocket = 4, expand_pocket_by_res = False, with surface_accessible mask, epochs = 200

    # scheduler = ?? over more epochs, learning rate should be reduced to remain in the minimum
    # at the end of each epoch, add scheduler.step()
    # try without scheduler, then add

    # add assessment of validation set performance

# balanced accuracy
# recall score
# f1 score
# precision

# 75% train/25% validation/%0 test
# GOAL IS 85% POSITIVE RECALL (pocket detection reaches this) , ideally this does not use pocket detection and exceeds this
# goal to reach 65-70%, then optimize more


'''

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric as pyg
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import random

from utils import Meter
from load import GraphDataset
from log.logger import Logger

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


# parameters
DATASET_PATH = "data/dataset/graphed.csv"
TRAIN_VALIDATION_TEST_SPLIT = [0.75, 0.0, 0.25]
BATCH_SIZE = 8
NUM_WORKERS = 8

HIDDEN_CHANNELS = 64

EPOCHS = 2
LEARNING_RATE = 1e-5

OUTPUT_DIR = "hpc/out"


# define model
class GCN(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim):
        super().__init__()

        self.conv1 = pyg.nn.GATv2Conv(in_channels=in_channels, out_channels=hidden_channels, edge_dim=edge_dim)
        self.conv2 = pyg.nn.GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, edge_dim=edge_dim)
        self.conv3 = pyg.nn.GATv2Conv(in_channels=hidden_channels, out_channels=out_channels, edge_dim=edge_dim)

        mu_cov = 1.30 # typical covalent radius
        dev_cov = 0.15 # covalent deviation
        mu_ncov = 10.0 # typical non-covalent radius
        dev_ncov = 1.0 # non-covalent deviation

        self.mu_cov = nn.Parameter(torch.Tensor([mu_cov]).float())
        self.dev_cov = nn.Parameter(torch.Tensor([dev_cov]).float())
        self.mu_ncov = nn.Parameter(torch.Tensor([mu_ncov]).float())
        self.dev_ncov = nn.Parameter(torch.Tensor([dev_ncov]).float())

    def forward(self, x, edge_index, edge_attr):
        edge_attr[:, 1] = torch.where(edge_attr[:, 0] == 1,
            torch.exp(-torch.pow(edge_attr[:, 1] - self.mu_cov, 2) / self.dev_cov), # covalent normalization
            torch.exp(-torch.pow(edge_attr[:, 1] - self.mu_ncov, 2) / self.dev_ncov)) # noncovalent normalization
        
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.2, training=True)
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.2, training=True)
        x = self.conv3(x, edge_index, edge_attr)
        
        return x


def main():

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # log parameters
    logger = Logger(os.path.join(OUTPUT_DIR, 'model.log'))

    logger.critical("MODEL PARAMETERS")
    logger.info(f"dataset_path = {DATASET_PATH}")
    logger.info(f"train_validation_test_split = {TRAIN_VALIDATION_TEST_SPLIT}")
    logger.info(f"batch_size = {BATCH_SIZE}")
    logger.info(f"num_workers = {NUM_WORKERS}")
    logger.info(f"hidden_channels = {HIDDEN_CHANNELS}")
    logger.info(f"epochs = {EPOCHS}")
    logger.info(f"learning_rate = {LEARNING_RATE}")


    # prepare dataset
    dataset_df = pd.read_csv(DATASET_PATH, sep=',', header=0, dtype=str)
    graph_path_list = dataset_df['graph_path'].tolist()

    num_train = int(len(graph_path_list) * TRAIN_VALIDATION_TEST_SPLIT[0])
    num_validation = int(len(graph_path_list) * TRAIN_VALIDATION_TEST_SPLIT[1])
    num_test = int(len(graph_path_list) * TRAIN_VALIDATION_TEST_SPLIT[2])

    random.shuffle(graph_path_list)

    train_path_list = random.sample(graph_path_list, num_train)
    path_list = list(set(graph_path_list) - set(train_path_list))
    validation_path_list = random.sample(path_list, num_validation)
    path_list = list(set(graph_path_list) - set(validation_path_list) - set(train_path_list))
    test_path_list = random.sample(path_list, num_test)

    train_dataset = GraphDataset(graph_path_list=train_path_list)
    # validation_dataset = GraphDataset(graph_path_list=validation_path_list)
    test_dataset = GraphDataset(graph_path_list=test_path_list)
    
    example_graph = train_dataset.get(0)

    logger.critical("MODEL PROPERTIES")
    logger.info(f"num_train = {num_train}")
    logger.info(f"num_validation = {num_validation}")
    logger.info(f"num_test = {num_test}")
    logger.info(f"example_graph = {example_graph}")


    # prepare data loaders
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
    # vaidation_loader = pyg.loader.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)


    # prepare model
    IN_CHANNELS = example_graph.num_node_features
    OUT_CHANNELS = 1
    EDGE_DIM = example_graph.num_edge_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = GCN(in_channels=IN_CHANNELS, hidden_channels=HIDDEN_CHANNELS, out_channels=OUT_CHANNELS, edge_dim=EDGE_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([10])).to(device)
    
    logger.critical("MODEL ARCHITECTURE")
    logger.info(f"device = {device}")
    logger.info(f"model = {model}")
    logger.info(f"optimizer = {optimizer}")
    logger.info(f"criterion = {criterion}")


    # start training
    logger.critical("MODEL TRAINING")

    batch_loss = Meter()
    training_loss = []

    model.train()

    for epoch in range(EPOCHS):
        for data in train_loader:
            data.to(device)

            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out[data.surface_accessible], data.y[data.surface_accessible].reshape(-1,1).float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_loss.update(loss.item(), data.num_nodes)

        epoch_loss = batch_loss.average()
        batch_loss.reset()
        training_loss.append(epoch_loss)

        torch.cuda.empty_cache()
        
        logger.info(f"epoch {epoch:03d}: epoch_loss = {epoch_loss:.4f}, mu_ncov = {model.mu_ncov.cpu().data.numpy()[0]:.4f}")

    plt.figure()
    plt.plot(training_loss)
    plt.title(f'training, epochs = {EPOCHS}, num_train = {num_train}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_loss.png'))


    # start testing
    logger.critical("MODEL TESTING")

    pred = torch.empty((0), dtype=torch.int).to(device)
    y = torch.empty((0), dtype=torch.int).to(device)

    model.eval()

    for data in test_loader:
        data.to(device)

        out = model(data.x, data.edge_index, data.edge_attr)
        out = torch.sigmoid(out)
        out = torch.where(out >= 0.5, 1, 0)

        pred = torch.cat((pred, out[data.surface_accessible]), axis=0)
        y = torch.cat((y, data.y[data.surface_accessible].reshape(-1,1)), axis=0)
    
    test_confusion_matrix = skm.confusion_matrix(y.cpu().detach().numpy(), pred.cpu().detach().numpy())
    test_recall = skm.recall_score(y.cpu().detach().numpy(), pred.cpu().detach().numpy())

    test_tp = test_confusion_matrix[1, 1]
    test_tn = test_confusion_matrix[0, 0]
    test_fp = test_confusion_matrix[0, 1]
    test_fn = test_confusion_matrix[1, 0]

    logger.info(f"true positive: {test_tp}")
    logger.info(f"true negative: {test_tn}")
    logger.info(f"false positive: {test_fp}")
    logger.info(f"false negative: {test_fn}")

    logger.info(f"recall = {test_recall}")
    
    logger.info(f"mu_cov = {model.mu_cov.cpu().data.numpy()[0]:.4f}")
    logger.info(f"dev_cov = {model.dev_cov.cpu().data.numpy()[0]:.4f}")
    logger.info(f"mu_ncov = {model.mu_ncov.cpu().data.numpy()[0]:.4f}")
    logger.info(f"dev_ncov = {model.dev_ncov.cpu().data.numpy()[0]:.4f}")


if __name__ == "__main__":
    main()
