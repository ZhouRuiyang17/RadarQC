import os
from typing import List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import reader


class TrainingDataset(Dataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset. 
        azimuth_range (List[int]): Range of azimuth. 
        distance_range (List[int]): Range of radial distance. 
    """
    def __init__(self, root: str, azimuth_range: List[int], distance_range: List[int]):
        super().__init__()
        self.azimuth_range = azimuth_range
        self.distance_range = distance_range
        self.sample_num = 0
        self.files = []
        date_list = sorted(os.listdir(root))
        for date in date_list:
            file_list = sorted(os.listdir(os.path.join(root, date)))
            for file_ in file_list:
                self.files.append(os.path.join(root, date, file_))
                self.sample_num += 1

    def __getitem__(self, index):
        filename = self.files[index]
        elev, ref = reader.read_radar_bin(filename)
        ref = ref[:, self.azimuth_range[0]: self.azimuth_range[1], self.distance_range[0]: self.distance_range[1]]
        elev, ref = torch.from_numpy(elev), torch.from_numpy(ref)
        return elev, ref
    
    def __len__(self):
        return self.sample_num


class SampleDataset(TrainingDataset):
    """Customized dataset.

    Args:
        root (str): Root directory for the dataset. 
        azimuth_range (List[int]): Range of azimuth. 
        distance_range (List[int]): Range of radial distance. 
    """

    def __init__(self, root: str, sample_index: int, azimuth_range: List[int], distance_range: List[int]):
        super().__init__(root)
        self.azimuth_range = azimuth_range
        self.distance_range = distance_range
        self.sample_index = sample_index

    def __getitem__(self, index: int):
        filename = self.files[self.sample_index]
        elev, ref = reader.read_radar_bin(filename)
        ref = ref[:, self.azimuth_range[0]: self.azimuth_range[1], self.distance_range[0]: self.distance_range[1]]
        elev, ref = torch.from_numpy(elev), torch.from_numpy(ref)
        return elev, ref

    def __len__(self):
        return 1


def load_data(root: str, batch_size: int, num_workers: int, train_ratio: float, valid_ratio: float,
              azimuth_range: List[int] = [0, 360], distance_range: List[int] = [0, 460]) \
              -> Tuple[DataLoader, DataLoader, DataLoader]:
    r"""Load training and test data.

    Args:
        root (str): Path to the dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of processes.
        train_ratio (float): Training ratio of the whole dataset.
        valid_ratio (float): Validation ratio of the whole dataset.
        azimuth_range (List[int]): Range of azimuth. Default: [0, 360]
        distance_range (List[int]): Range of radial distance. Default: [0, 460]
    
    Returns:
        DataLoader: Dataloader for training and test.
    """

    dataset = TrainingDataset(root, azimuth_range, distance_range)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_node = round(train_ratio * dataset_size)
    val_node = round(valid_ratio * dataset_size)
    train_indices = indices[:train_node]
    val_indices = indices[train_node: train_node + val_node]
    test_indices = indices[train_node + val_node:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers,
                             shuffle=False, drop_last=True, pin_memory=True)

    print('\nTrain Loader')
    print('----Batch Num:', len(train_loader))

    print('\nVal Loader')
    print('----Batch Num:', len(val_loader))

    print('\nTest Loader')
    print('----Batch Num:', len(test_loader))

    return train_loader, val_loader, test_loader


def load_sample(root: str, sample_index: int, azimuth_range: List[int] = [0, 360], 
                distance_range: List[int] = [0, 460]) -> DataLoader:
    r"""Load sample data.

    Args:
        root (str): Path to the dataset.
        sample_index (int): Index of the sample. Default is None, meaning the last sample is selected.
        azimuth_range (List[int]): Range of azimuth. Default: [0, 360]
        distance_range (List[int]): Range of radial distance. Default: [0, 460]
    
    Returns:
        DataLoader: Dataloader for sample.
    """

    sample_set = SampleDataset(root, sample_index, azimuth_range, distance_range)
    sample_loader = DataLoader(sample_set, batch_size=1)
    return sample_loader


if __name__ == '__main__':
    root = 'D:\Data\SBandDataAll\SBandBasicUnzip'
    train_loader, val_loader, test_loader = load_data(root, 16, 4, 0.7, 0.1, [45, 225], [0, 200])
    for i, (elev, ref) in enumerate(val_loader):
        print(i, elev.size(), ref.size())