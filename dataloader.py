import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class BilibiliDataset(Dataset):
    def __init__(self, metadata_csv_file, content_features_path, temporal_file_path):
        """
            Args:
                metadata_csv_file (string): Path to the CSV file containing video BVID and MID
                content_features_path (string): Path to the content features file
                temporal_file_path (string): Path to the file with popularity sequences and labels
        """
        self.metadata = pd.read_csv(metadata_csv_file)
        self.content_features = np.load(content_features_path, allow_pickle=True)
        self.temporal_data = np.load(temporal_file_path, allow_pickle=True)

        if len(self.content_features) != len(self.temporal_data):
            raise ValueError(
                f"The length of the content feature matrix ({len(self.content_features)}) is inconsistent with "
                f"the length of the time series matrix ({len(self.temporal_data)})"
            )

        # The split points of visual, acoustic, text, metadata, and social features
        self.content_splits = [4096, 4096 + 2688, 4096 + 2688 + 1538, 4096 + 2688 + 1538 + 22]

        # The split points of short-term popularity sequences, long-term popularity sequences, and t_p
        self.sequences_short_size = 72 * 4
        self.sequences_long_size = 6 * 3
        self.t_p_size = 3

    def __len__(self):
        return len(self.content_features)

    def __getitem__(self, idx):
        bvid_from_meta = self.metadata.iloc[idx]['bvid'] if idx < len(self.metadata) else f"IDX_{idx}"

        try:
            content_features = self.content_features[idx]

            # Split content features
            visual_features = content_features[:self.content_splits[0]]
            acoustic_features = content_features[self.content_splits[0]:self.content_splits[1]]
            textual_features = content_features[self.content_splits[1]:self.content_splits[2]]
            metadata_features = content_features[self.content_splits[2]:self.content_splits[3]]
            social_features = content_features[self.content_splits[3]:]

            features_dict = {
                'visual': torch.tensor(visual_features, dtype=torch.float32),
                'acoustic': torch.tensor(acoustic_features, dtype=torch.float32),
                'textual': torch.tensor(textual_features, dtype=torch.float32),
                'metadata': torch.tensor(metadata_features, dtype=torch.float32),
                'social': torch.tensor(social_features, dtype=torch.float32),
            }
        except Exception as e:
            print(f"Error loading content features at index {idx} (BVID may be {bvid_from_meta}): {e}")
            raise

        try:
            concatenated_features = self.temporal_data[idx].flatten()

            sequences_short = torch.tensor(
                concatenated_features[:self.sequences_short_size].reshape(72, 4),
                dtype=torch.float32
            )

            sequences_long = torch.tensor(
                concatenated_features[
                self.sequences_short_size:self.sequences_short_size + self.sequences_long_size].reshape(6, 3),
                dtype=torch.float32
            )

            t_p = torch.tensor(
                concatenated_features[
                self.sequences_short_size + self.sequences_long_size:self.sequences_short_size + self.sequences_long_size + self.t_p_size],
                dtype=torch.float32
            )

            labels = torch.tensor(
                concatenated_features[self.sequences_short_size + self.sequences_long_size + self.t_p_size:],
                dtype=torch.float32
            )
        except Exception as e:
            print(f"Error loading temporal data at index {idx} (BVID may be {bvid_from_meta}): {e}")
            raise

        features_dict['temporal_short'] = sequences_short
        features_dict['temporal_long'] = sequences_long
        features_dict['t_p'] = t_p

        return features_dict, labels
