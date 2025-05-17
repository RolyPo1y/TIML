import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import csv
import numpy as np
import random
from tqdm import tqdm
from utils import init_weights
from metrics import calculate_mae, calculate_spearman, calculate_r_squared
from dataloader import BilibiliDataset
from model import TIML
import os


def set_random_seed(seed_value):
    """
    Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True  # CUDNN optimizer
        torch.backends.cudnn.benchmark = False


def plot_results(epoch, train_values, val_values, value_name, save_path=None):
    plt.figure(figsize=(10, 6))
    epochs = epoch + 1
    plt.plot(epochs, train_values, label=f'Training {value_name}')
    plt.plot(epochs, val_values, label=f'Validation {value_name}')
    plt.xlabel('Epochs')
    plt.ylabel(value_name)
    plt.title(f'{value_name} over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs, patience, fold):
    best_val_mse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        all_outputs = []
        all_targets = []

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Training", unit="batch"):
            # Extract features and labels
            features_dict = batch[0]
            targets = batch[1].to(device)

            visual_input = features_dict['visual'].to(device)
            acoustic_input = features_dict['acoustic'].to(device)
            textual_input = features_dict['textual'].to(device)
            metadata_input = features_dict['metadata'].to(device)
            social_input = features_dict['social'].to(device)
            temporal_short_input = features_dict['temporal_short'].to(device)
            temporal_long_input = features_dict['temporal_long'].to(device)
            t_p = features_dict['t_p'].to(device)

            optimizer.zero_grad()

            outputs = model(visual_input, acoustic_input, textual_input, metadata_input, social_input,
                            temporal_short_input, temporal_long_input, t_p)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metrics on the training set
        epoch_loss = criterion(all_outputs, all_targets).item()
        epoch_mae = calculate_mae(all_targets, all_outputs)
        epoch_spearman = calculate_spearman(all_targets, all_outputs)
        epoch_r_square = calculate_r_squared(all_targets, all_outputs)

        print(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}, "
            f"Spearman: {epoch_spearman:.4f}, R-squared: {epoch_r_square:.4f}")

        # Validation
        model.eval()
        val_outputs = []
        val_targets = []

        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{epochs} - Validation", unit="batch"):
                features_dict = val_batch[0]
                targets = val_batch[1].to(device)

                visual_input = features_dict['visual'].to(device)
                acoustic_input = features_dict['acoustic'].to(device)
                textual_input = features_dict['textual'].to(device)
                metadata_input = features_dict['metadata'].to(device)
                social_input = features_dict['social'].to(device)
                temporal_short_input = features_dict['temporal_short'].to(device)
                temporal_long_input = features_dict['temporal_long'].to(device)
                t_p = features_dict['t_p'].to(device)

                outputs = model(visual_input, acoustic_input, textual_input, metadata_input, social_input,
                                temporal_short_input, temporal_long_input, t_p)

                val_outputs.append(outputs.detach())
                val_targets.append(targets.detach())

            val_outputs = torch.cat(val_outputs, dim=0)
            val_targets = torch.cat(val_targets, dim=0)

        # Calculate metrics on the validation set
        val_epoch_loss = criterion(val_outputs, val_targets).item()
        val_epoch_mae = calculate_mae(val_targets, val_outputs)
        val_epoch_spearman = calculate_spearman(val_targets, val_outputs)
        val_epoch_r_square = calculate_r_squared(val_targets, val_outputs)

        print(
            f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_epoch_loss:.4f}, MAE: {val_epoch_mae:.4f}, "
            f"Spearman: {val_epoch_spearman:.4f}, R-squared: {val_epoch_r_square:.4f}")

        # Save the model weights that currently perform best on the validation set
        if val_epoch_loss < best_val_mse:
            best_val_mse = val_epoch_loss
            patience_counter = 0  # Reset counter when improvement is found
            best_model_path = f'...\\TIML ckpts\\model_{fold}.pth'
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # plot_results(epoch, epoch_loss, val_epoch_loss, 'Loss', save_path=None)
        # plot_results(train_maes, val_maes, 'MAE', save_path=None)
        # plot_results(train_spearmans, val_spearmans, 'Spearman', save_path=None)
        # plot_results(train_r_squares, val_r_squares, 'R-Square', save_path=None)

        # 记录并保存训练过程中的指标
        results_path = f'...\\TIML\\train_result\\training_results_{fold}.csv'
        with open(results_path, 'a', newline='') as csvfile:
            fieldnames = ['Epoch', 'Train Loss', 'Val Loss', 'Train MAE', 'Val MAE',
                          'Train Spearman', 'Val Spearman', 'Train R-squared', 'Val R-squared']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if epoch == 0:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch + 1,
                'Train Loss': float(epoch_loss),
                'Val Loss': float(val_epoch_loss),
                'Train MAE': float(epoch_mae),
                'Val MAE': float(val_epoch_mae),
                'Train Spearman': float(epoch_spearman),
                'Val Spearman': float(val_epoch_spearman),
                'Train R-squared': float(epoch_r_square),
                'Val R-squared': float(val_epoch_r_square)
            })

        # Check for early stopping
        if patience_counter >= patience:
            print(f"Early stopping was triggered at epoch {epoch + 1}")
            break


def main():
    # set random seed
    set_random_seed(42)

    metadata_emb_sizes = [
        (16, 4),  # UGV partition
        (105, 11),  # UGV type
        (2, 2),  # HDE
        (2, 2),  # HLD
        (12, 4),  # TMZ
        (4, 2),  # Recommend count
        (51, 8),  # UGV tag count
        (148, 13),  # UGV segment count
        (2, 2),  # UGV originality
        (2, 2),  # UGV format
        (2, 2),  # Activity participation
        (2, 2),  # UGV chargeable
        (2, 2),  # UGV downloadable
        (2, 2),  # UGV reprintable
        (2, 2),  # UGV collaboration
        (2, 2),  # UGV collection
        (2, 2),  # UGV Privileged
        (2, 2),  # High bitrate
        (2, 2),  # Panoramic video
        (2, 2),  # UGV interactivity
        (1157, 35)  # UGV resolution
    ]

    social_emb_sizes = [
        (7, 3),  # User account level
        (3, 2),  # User gender
        (3, 2),  # User member type
        (3, 2),  # User verification
        (10, 4),  # User verification type
    ]

    short_term_tcxt_indices = [0, 1, 2]
    short_term_popularity_tseries_index = [3]

    long_term_tcxt_indices = [0, 1]
    long_term_popularity_tseries_index = [2]

    short_term_tcxt_num_embeddings = [12, 2, 2]  # The number of categories in each short-term temporal context
    short_term_tcxt_embedding_dims = [4, 2, 2]  # The embedding dimension of each short-term temporal context

    long_term_tcxt_num_embeddings = [2, 2]  # The number of categories in each long-term temporal context
    long_term_tcxt_embedding_dims = [2, 2]  # The embedding dimension of each long-term temporal context

    meta_csv_file = r'...\metadata.csv'
    content_features_path = r'...\content features'
    temporal_file_path = r'...\popularity_time_series_data.csv'

    dataset = BilibiliDataset(meta_csv_file, content_features_path, temporal_file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    metrics = ['Fold', 'Test Loss', 'Test MAE', 'Test Spearman', 'Test R-squared']
    results_path = r'...\TIML\train_result\test_result_TIML.csv'
    if not os.path.exists(results_path):
        with open(results_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics)
            writer.writeheader()

    for fold in range(1, 6):
        print(f"Fold {fold}")

        train_size = int(0.80 * len(dataset))
        val_size = int(0.10 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        print(train_size, val_size, test_size)

        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=val_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

        model = TIML(
            attention_hidden_dim=32,
            metadata_emb_sizes=metadata_emb_sizes,
            social_emb_sizes=social_emb_sizes,
            short_term_tcxt_indices=short_term_tcxt_indices,
            short_term_popularity_tseries_index=short_term_popularity_tseries_index,
            long_term_tcxt_indices=long_term_tcxt_indices,
            long_term_popularity_tseries_index=long_term_popularity_tseries_index,
            short_term_tcxt_num_embeddings=short_term_tcxt_num_embeddings,
            short_term_tcxt_embedding_dims=short_term_tcxt_embedding_dims,
            long_term_tcxt_num_embeddings=long_term_tcxt_num_embeddings,
            long_term_tcxt_embedding_dims=long_term_tcxt_embedding_dims
        ).to(device)

        model.apply(init_weights)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        train(model, train_dataloader, val_dataloader, criterion, optimizer, device, epochs=50, patience=50, fold=fold)

        # Evaluate the model on the test set
        model.load_state_dict(torch.load(f'...\\TIML ckpts\\model_{fold}.pth'))
        model.eval()
        test_outputs = []
        test_targets = []

        with torch.no_grad():
            for test_batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
                features_dict = test_batch[0]
                targets = test_batch[1].to(device)

                visual_input = features_dict['visual'].to(device)
                acoustic_input = features_dict['acoustic'].to(device)
                textual_input = features_dict['textual'].to(device)
                metadata_input = features_dict['metadata'].to(device)
                social_input = features_dict['social'].to(device)
                temporal_short_input = features_dict['temporal_short'].to(device)
                temporal_long_input = features_dict['temporal_long'].to(device)
                t_p = features_dict['t_p'].to(device)

                outputs = model(visual_input, acoustic_input, textual_input, metadata_input, social_input,
                                temporal_short_input, temporal_long_input, t_p)

                test_outputs.append(outputs.detach())
                test_targets.append(targets.detach())

            test_outputs = torch.cat(test_outputs, dim=0)
            test_targets = torch.cat(test_targets, dim=0)

            results = {
                'Fold': fold,
                'Test Loss': float(criterion(test_outputs, test_targets).item()),
                'Test MAE': float(calculate_mae(test_targets, test_outputs)),
                'Test Spearman': float(calculate_spearman(test_targets, test_outputs)),
                'Test R-squared': float(calculate_r_squared(test_targets, test_outputs))
            }

            with open(results_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=metrics)
                writer.writerow(results)


if __name__ == "__main__":
    main()
