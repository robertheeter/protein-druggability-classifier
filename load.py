import torch
from torch_geometric.data import Dataset, Batch


class GraphDataset(Dataset):

    def __init__(self, graph_path_list, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(transform, pre_transform)
        self.graph_path_list = graph_path_list
    
    def len(self):
      return len(self.graph_path_list)
    
    def indices(self):
      return range(0, len(self.graph_path_list))
    
    def get(self, index):
      return torch.load(self.graph_path_list[index])
    
    def collate_fn(self, batch):
      return Batch.from_data_list(batch)
    

# example implementation
if __name__ == '__main__':
   
   # loading parameters
   data_path = "data/dataset/graphed.csv"

    # test loading
   dataset = GraphDataset(data_path)
   graph = dataset.get(0)
   print(graph)