from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, tokinized_data, sequence_length):
        self.data = tokinized_data
        self.sequence_length = sequence_length

    def __len__(self):
        return len((self.data - 1) // self.sequence_length)

    def __getitem__(self, idx):
        data = data[idx * self.sequence_length: (idx + 1) * self.sequence_length]
        labels = data[idx * self.sequence_length + 1: (idx + 1) * self.sequence_length + 1]
        return (data, labels)
