import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor


class VideoDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.time_indices = list(range(4, 76))
        self.feature_min = {}
        self.feature_max = {}
        self.label_min = {}
        self.label_max = {}
        self.day_popularity_score_min = {}
        self.day_popularity_score_max = {}

        self.bvids = self.data['bvid'].unique()
        self.save_dir = r'...\popularity time series and labels'
        self.popularity_csv_save_path = r'...\train_result\day_popularity_scores.csv'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not self.check_samples_exist():
            self.calculate_normalization_params()
            self.calculate_and_save_samples()

    def check_samples_exist(self):
        # Check the number of files in save_dir and label_save_dir
        num_files_in_save_dir = len(os.listdir(self.save_dir))
        return num_files_in_save_dir > 0

    def calculate_normalization_params(self):
        base_feature_names = ['time_popularity_score']
        time_indices = self.time_indices
        day_popularity_score_records = []

        for bvid in self.bvids:
            video_data = self.data[self.data['bvid'] == bvid]

            # Calculate the minimum and maximum values of the time_popularity_score
            features_concatenated_data = []
            for name in base_feature_names:
                cols = [f'{name}.{i}' for i in time_indices]
                concatenated_data = video_data[cols].values.flatten()
                features_concatenated_data.extend(concatenated_data)

            feature_min = np.min(features_concatenated_data)
            feature_max = np.max(features_concatenated_data)
            self.feature_min[bvid] = feature_min
            self.feature_max[bvid] = feature_max

            # Calculate the minimum and maximum values of the day_popularity_score
            day_popularity_score_data = []
            for i in range(6):  # The day_popularity_score for 6 days
                start_idx = 4 + i * 12
                end_idx = start_idx + 12
                day_popularity_score = video_data[[f'time_popularity_score.{j}' for j in range(start_idx, end_idx)]].sum(axis=1).values
                day_popularity_score_data.extend(day_popularity_score)
                day_popularity_score_records.append([bvid, i, day_popularity_score[0]])

            day_popularity_score_min = np.min(day_popularity_score_data)
            day_popularity_score_max = np.max(day_popularity_score_data)
            self.day_popularity_score_min[bvid] = day_popularity_score_min
            self.day_popularity_score_max[bvid] = day_popularity_score_max

        # Save the day_popularity_score_data to a CSV file.
        day_popularity_score_df = pd.DataFrame(day_popularity_score_records, columns=['bvid', 'day', 'day_popularity_score'])
        day_popularity_score_df.to_csv(self.popularity_csv_save_path, index=False)
        print(f"Saved day_popularity_scores to {self.popularity_csv_save_path}")

    def process_temporal_data(self, bvid):
        """
        Extract the long and short term popularity series and temporal contexts
        of 72 time points in the past six days
        """
        print(f"Processing video with bvid: {bvid}")
        video_data = self.data[self.data['bvid'] == bvid]
        samples = []
        feature_indices = list(range(4, 76))
        features = []

        for j in feature_indices:
            # Extract the temporal contexts for each time point
            timestamp_features = video_data[
                [f'time_zone.{j}', f'is_day_before_holiday.{j}', f'is_holiday.{j}']].values.flatten()
            raw_popularity_score = video_data[f'time_popularity_score.{j}'].values
            if self.feature_max[bvid] != self.feature_min[bvid]:
                normalized_popularity_score = (raw_popularity_score - self.feature_min[bvid]) / (
                        self.feature_max[bvid] - self.feature_min[bvid])
            else:
                normalized_popularity_score = raw_popularity_score * 0

            # Concatenate the temporal contexts and the normalized short_term_popularity
            feature = np.concatenate((timestamp_features, normalized_popularity_score))
            features.append(feature)

        short_time_features = np.array(features)

        # Extract the long_term_popularity for each day
        long_time_features = []
        for i in range(6):  # first six days
            start_idx = 4 + i * 12
            end_idx = start_idx + 12
            day_popularity_score = video_data[[f'time_popularity_score.{j}' for j in range(start_idx, end_idx)]].sum(
                axis=1).values
            if self.day_popularity_score_max[bvid] != self.day_popularity_score_min[bvid]:
                normalized_day_popularity_score = (day_popularity_score - self.day_popularity_score_min[bvid]) / (
                        self.day_popularity_score_max[bvid] - self.day_popularity_score_min[bvid])
            else:
                normalized_day_popularity_score = day_popularity_score * 0

            j = start_idx
            daily_features = video_data[
                [f'is_day_before_holiday.{j}', f'is_holiday.{j}']
            ].values.flatten()

            long_time_feature = np.concatenate((daily_features, normalized_day_popularity_score))
            long_time_features.append(long_time_feature)

        long_time_features = np.array(long_time_features)

        # Extract labels
        raw_label = video_data[[f'time_popularity_score.{i}' for i in range(76, 88)]].sum(axis=1).values
        raw_label[raw_label <= 0] = 0
        normalized_label = np.log(raw_label + 1)

        # Extract the temporal contexts t_p of the target prediction date
        t_p = video_data[
            [f'is_day_before_holiday.76', f'is_holiday.76', f'days_after_upload.76']
        ].values.flatten()

        sample = (short_time_features, long_time_features, normalized_label, t_p)
        samples.append(sample)

        return bvid, samples

    def calculate_and_save_samples(self):
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.process_temporal_data, bvid) for bvid in self.bvids]
            for future in futures:
                bvid, samples = future.result()
                for short_time_features, long_time_features, labels, t_p in samples:
                    # Flatten the time series into one dimension
                    short_time_features = short_time_features.reshape(-1)  # shape -> [72*4]
                    long_time_features = long_time_features.reshape(-1)  # shape -> [6*3]

                    concatenated_features = np.concatenate([
                        short_time_features,
                        long_time_features,
                        t_p,
                        labels
                    ])

                    feature_label_file = os.path.join(self.save_dir, f'{bvid}_temporal_feature_label.npy')
                    np.save(feature_label_file, concatenated_features)

                    print(f"Saved {bvid} temporal features, labels, and timestamp.")

    def __len__(self):
        return len(self.bvids)

    def __getitem__(self, idx):
        bvid = self.bvids[idx]
        feature_label_file = os.path.join(self.save_dir, f'{bvid}_temporal_feature_label.npy')

        if not os.path.exists(feature_label_file):
            raise FileNotFoundError(f"Expected file not found: {feature_label_file}")

        concatenated_features = np.load(feature_label_file)

        # Restore the shape of the time series
        short_time_feature_size = 72 * 4
        long_time_feature_size = 6 * 3
        t_p_size = 3

        short_time_features = (torch.tensor
                               (concatenated_features[:short_time_feature_size].reshape(72, 4), dtype=torch.float32).cuda())
        long_time_features = (torch.tensor
                              (concatenated_features[short_time_feature_size:short_time_feature_size + long_time_feature_size].reshape(6, 3), dtype=torch.float32).cuda())
        t_p = (torch.tensor
                      (concatenated_features[short_time_feature_size + long_time_feature_size:short_time_feature_size + long_time_feature_size + t_p_size], dtype=torch.float32).cuda())
        labels = (torch.tensor
                  (concatenated_features[short_time_feature_size + long_time_feature_size + t_p_size:], dtype=torch.float32).cuda())

        return bvid, short_time_features, long_time_features, t_p, labels


class BilibiliDataset(Dataset):
    def __init__(self, meta_csv_file, content_features_path, temporal_file_path):
        """
        Args:
            meta_csv_file (string): The path to the CSV file containing the video BVIDs and MIDs(user IDs).
            content_features_path (string): The path where the concatenated content features of each modality
            (in the format of {bvid}.npy) are stored.
            temporal_file_path (string): The path to the file containing the video popularity time series.
        """
        self.meta_data = pd.read_csv(meta_csv_file)
        self.non_temp_feature_path = content_features_path

        self.temporal_data = pd.read_csv(temporal_file_path)
        self.temporal_bvid_to_idx = {bvid: idx for idx, bvid in enumerate(self.temporal_data['bvid'].unique())}

        self.temporal_dataset = VideoDataset(temporal_file_path)

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        bvid = self.meta_data.iloc[idx]['bvid']

        # Read and process content features
        try:
            feature_path = os.path.join(self.non_temp_feature_path, f'{bvid}.npy')
            combined_feature = np.load(feature_path)

            visual_feature = combined_feature[:4096]
            acoustic_feature = combined_feature[4096:4096 + 2688]
            textual_feature = combined_feature[4096 + 2688:4096 + 2688 + 1538]
            metadata_feature = combined_feature[4096 + 2688 + 1538:4096 + 2688 + 1538 + 22]
            social_feature = combined_feature[4096 + 2688 + 1538 + 22:]

            multimodal_features = {
                'visual': torch.tensor(visual_feature, dtype=torch.float32).cuda(),
                'acoustic': torch.tensor(acoustic_feature, dtype=torch.float32).cuda(),
                'textual': torch.tensor(textual_feature, dtype=torch.float32).cuda(),
                'metadata': torch.tensor(metadata_feature, dtype=torch.float32).cuda(),
                'social': torch.tensor(social_feature, dtype=torch.float32).cuda(),
            }
        except Exception as e:
            print(f"Error loading features for {bvid}: {e}")
            raise

        # Find the index of bvid in the temporal series CSV data
        if bvid not in self.temporal_bvid_to_idx:
            raise KeyError(f"bvid {bvid} not found in temporal data")

        temporal_idx = self.temporal_bvid_to_idx[bvid]

        # Obtain the popularity time series and labels
        bvid_temporal, temporal_short, temporal_long, t_p, labels = self.temporal_dataset[temporal_idx]

        # Ensure that the content features and the popularity time series belong to the same video
        assert bvid == bvid_temporal, f"Feature bvid mismatch: {bvid}!= {bvid_temporal}"

        multimodal_features['temporal_short'] = temporal_short  # Short-term patterns in the popularity time series
        multimodal_features['temporal_long'] = temporal_long  # Long-term patterns in the popularity time series
        multimodal_features['t_p'] = t_p  # Temporal contexts of the target prediction date


        return multimodal_features, labels