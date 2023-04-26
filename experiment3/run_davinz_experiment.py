import torch
import pickle
import os
import traceback

import torch.nn as nn
import polars as pl
import numpy as np

from tqdm import tqdm
from time import time
from torch.utils.data import TensorDataset, DataLoader
from functorch import make_functional, vmap, vjp, jvp, jacrev
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from ntk_bound import davinz_bound

# Environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Global variables
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_PATH = './training_results'

VALID_SIZE = 2000

TRAIN_SIZE = 500
TRAIN_CLASS_1_RATIO = 0.3

N_EPOCHS = 300

NEW_TRAIN_SIZE_MIN = 100
NEW_TRAIN_SIZE_MAX = 3500
NEW_TRAIN_SIZE_STEP = 200
NEW_TRAIN_CLASS_1_RATIO = 0.6
N_EPOCHS_NEW = 500

LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3


def preprocess_dataset(data: pl.DataFrame) -> pl.DataFrame:
    # Encode Gender
    enc = OneHotEncoder(handle_unknown='error', sparse_output=False)
    gender_oh_encoded = enc.fit_transform(data['Gender'].to_numpy().reshape(-1, 1))

    # Encode Vehicle_Age
    enc = OrdinalEncoder(categories=[['< 1 Year', '1-2 Year', '> 2 Years']], handle_unknown='error')
    vehicle_age_encoded = enc.fit_transform(data['Vehicle_Age'].to_numpy().reshape(-1, 1))

    # Encode Vehicle_Damage
    enc = OneHotEncoder(handle_unknown='error', sparse_output=False)
    vehicle_damage_encoded = enc.fit_transform(data['Vehicle_Damage'].to_numpy().reshape(-1, 1))

    # Standardize variables
    scaler = StandardScaler()
    age_standard = scaler.fit_transform(data['Age'].to_numpy().reshape(-1, 1))
    annual_premium_standard = scaler.fit_transform(data['Annual_Premium'].to_numpy().reshape(-1, 1))
    vintage_standard = scaler.fit_transform(data['Vintage'].to_numpy().reshape(-1, 1))

    data = data.drop(['Age', 'Annual_Premium', 'Vintage', 'Gender', 'Vehicle_Age', 'Vehicle_Damage']).with_columns(
        [
            pl.Series('Female', values=gender_oh_encoded[:, 0]),
            pl.Series('Male', values=gender_oh_encoded[:, 1]),
            pl.Series('Age', values=age_standard[:, 0]),
            pl.Series('Annual_Premium', values=annual_premium_standard[:, 0]),
            pl.Series('Vintage', values=vintage_standard[:, 0]),
            pl.Series('Vehicle_Age', values=vehicle_age_encoded[:, 0]),
            pl.Series('No_vehicle_damage', values=vehicle_damage_encoded[:, 0]),
            pl.Series('Vehicle_Damage', values=vehicle_damage_encoded[:, 1])
        ]
    )
    return data

def get_validation_split(
        data_only_0: pl.DataFrame,
        data_only_1: pl.DataFrame,
        size: int
) -> tuple[pl.DataFrame]:
    # Balance out dataset
    num_y_1 = size * 50 // 100
    num_y_0 = size - num_y_1
    
    # Select validation slices
    data_only_1_valid = data_only_1[:num_y_1]
    data_only_0_valid = data_only_0[:num_y_0]

    # Remove validation slices from the whole set (avoid overlapping with training set)
    data_only_1 = data_only_1[num_y_1:]
    data_only_0 = data_only_0[num_y_0:]

    data_valid = pl.concat(
        [
            data_only_1_valid,
            data_only_0_valid
        ],
        how='vertical'
    ).sample(frac=1, shuffle=True, seed=83409)

    assert data_valid.filter(pl.col('Response') == 1).shape[0] == data_valid.filter(pl.col('Response') == 0).shape[0]

    return data_valid, data_only_0, data_only_1

def get_training_split(
        data_only_0: pl.DataFrame, 
        data_only_1: pl.DataFrame, 
        size: int, 
        class_1_ratio: float,
        seed: int
) -> tuple[pl.DataFrame]:
    num_y_1 = int(size * class_1_ratio)
    num_y_0 = size - num_y_1

    data_train = pl.concat(
        [
            df_y_1[:num_y_1],
            df_y_0[:num_y_0]
        ],
        how='vertical'
    ).sample(frac=1, shuffle=True, seed=seed)

    # Remove training slices from the whole set (avoid future overlapping)
    data_only_1 = data_only_1[num_y_1:]
    data_only_0 = data_only_0[num_y_0:]

    return data_train, data_only_0, data_only_1

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layer1 = nn.Linear(12, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x


def predict(model_with_logit: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model_with_logit.eval()
    with torch.inference_mode():
        # Predicted probabilities
        pred_probs = torch.sigmoid(model_with_logit(X.to(DEVICE)))
    # Predicted labels
    if was_training is True:
        model_with_logit.train()
    return torch.round(pred_probs)

def predict_probs(model_with_logit: nn.Module, X: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model_with_logit.eval()
    with torch.inference_mode():
        # Predicted probabilities
        pred_probs = torch.sigmoid(model_with_logit(X.to(DEVICE)))
    # Predicted labels
    if was_training is True:
        model_with_logit.train()
    return pred_probs

def train_mlp(
        model: nn.Module, 
        dataloader_train: DataLoader, 
        criterion,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        X_valid: torch.Tensor,
        y_valid: torch.Tensor):
    # Load validation set to device
    X_valid = X_valid.to(DEVICE)
    y_valid = y_valid.to(DEVICE)

    train_loss_history = []
    valid_loss_history = []

    start_time = time()
    for epoch in range(n_epochs):
        running_train_loss = 0.0
        # Batches
        for inputs, labels in tqdm(dataloader_train):
            model.train()
            # Copy batch to device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Rremove gradients from previous batch
            optimizer.zero_grad()
            # Predict using current model state
            outputs = model(inputs)

            # Compute batch loss
            loss = criterion(outputs, labels)
            # Backpropagate
            loss.backward()
            optimizer.step()

            # Add training loss
            running_train_loss += loss.item()
        # Epoch training loss (divide by num of batches)
        epoch_train_loss = running_train_loss / len(dataloader_train) 
        train_loss_history.append(epoch_train_loss)

        # Epoch validation loss
        with torch.inference_mode():
            outputs_valid = model(X_valid)
            epoch_valid_loss = criterion(outputs_valid, y_valid).item()
            # epoch_valid_loss = loss_valid.item() * X_valid.size(0)
            valid_loss_history.append(epoch_valid_loss)
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {epoch_train_loss:.4f}, Validation Loss = {epoch_valid_loss:.4f}")

    print(f"Took {time() - start_time:.2f} seconds to train in total")
    return train_loss_history, valid_loss_history


def get_accuracy(model_with_logit: nn.Module, X: torch.Tensor, y: torch.Tensor):
    X = X.to(DEVICE)
    y = y.to(DEVICE)
    # Predict labels using model
    pred_labels = predict(model_with_logit, X).to('cpu')
    X = X.to('cpu')
    y = y.to('cpu')
    # Compare predicted with ground truth
    is_equal_tensor = torch.eq(pred_labels.squeeze(), y.squeeze())
    accuracy = is_equal_tensor.sum() / len(X)

    diff = (pred_labels.squeeze() - y.squeeze())
    fpos = diff[diff == -1].shape[0]
    fneg = diff[diff == 1].shape[0]
    
    return accuracy.item(), fpos, fneg


def load_from_file(file_path) -> tuple[nn.Module, torch.optim.Optimizer]:
    # Load checkpoint dict from file
    checkpoint = torch.load(file_path)
    model = MLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


if __name__ == '__main__':
    df = pl.read_csv('./health-insurance-data/train.csv')
    df = df.sample(frac=1, shuffle=True, seed=68123)

    # Preprocess Dataset
    df = preprocess_dataset(data=df)
    print(f'Columns after preprocessing: {df.columns}')

    # Split dataset based on the binary label, then shuffle
    df_y_0 = df.filter(pl.col('Response') == 0).sample(frac=1, shuffle=True, seed=11897)
    df_y_1 = df.filter(pl.col('Response') == 1).sample(frac=1, shuffle=True, seed=4199)

    valid_size = VALID_SIZE
    # Get validation split and remove it from dataset to avoid overlapping with training set
    df_valid, df_y_0, df_y_1 = get_validation_split(data_only_0=df_y_0, data_only_1=df_y_1, size=valid_size)

    train_size = TRAIN_SIZE
    # Select unbalanced training set
    df_train, df_y_0, df_y_1= get_training_split(
        data_only_0=df_y_0,
        data_only_1=df_y_1, 
        size=train_size, 
        class_1_ratio=TRAIN_CLASS_1_RATIO, 
        seed=4128211
    )

    # Create training and validation tensors
    X_train = torch.from_numpy(df_train.select(pl.exclude(['id', 'Response'])).to_numpy()).float()
    y_train = torch.from_numpy(df_train['Response'].to_numpy().reshape(-1, 1)).float()
    X_valid = torch.from_numpy(df_valid.select(pl.exclude(['id', 'Response'])).to_numpy()).float()
    y_valid = torch.from_numpy(df_valid['Response'].to_numpy().reshape(-1, 1)).float()

    # Create tensor dataset for training
    dataset_train = TensorDataset(X_train, y_train)
    # torch.manual_seed(2298)
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)

    # Create model
    model = MLP().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # lr=0.00005 seems best with Adam
    # Train model
    start = time()
    train_loss_history, valid_loss_history = train_mlp(
        model, 
        dataloader_train, 
        criterion, 
        optimizer, 
        n_epochs=N_EPOCHS, 
        X_valid=X_valid, 
        y_valid=y_valid
    )
    time_training_delta = time() - start

    # Compute accuracy, false positives and false negatives
    accuracy_train, fpos_train, fneg_train = get_accuracy(model, X_train, y_train)
    accuracy_valid, fpos_valid, fneg_valid = get_accuracy(model, X_valid, y_valid)
    
    print(f'Best Training Loss = {np.min(train_loss_history):.4f}')
    print(f'Training Accuracy = {accuracy_train:.4f}')
    print(f'Training False Positive = {fpos_train}')
    print(f'Training False Negative = {fneg_train}')  

    print(f'\nBest Validation Loss = {np.min(valid_loss_history):.4f}')
    print(f'Validation Accuracy = {accuracy_valid:.4f}')
    print(f'Validation False Positive = {fpos_valid}')
    print(f'Validation False Negative = {fneg_valid}')

    # Save trained model
    baseline_checkpoint_path = os.path.join(RESULTS_PATH, "baseline_mlp.pt")
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, 
        baseline_checkpoint_path)
    # Will save performance of trained models to dict
    model_performance_dict = {
        'train_acc': [accuracy_train],
        'train_false_pos': [fpos_train],
        'train_false_neg': [fneg_train],
        'valid_acc': [accuracy_valid],
        'valid_false_pos': [fpos_valid],
        'valid_false_neg': [fneg_valid],
        'train_loss_history': [train_loss_history],
        'valid_loss_history': [valid_loss_history],
        'time_training_delta': [time_training_delta],
        'dataset_size': [TRAIN_SIZE],
        'n_epochs': [N_EPOCHS],
        'valid_size': VALID_SIZE
    }
    # Will save davinz results to dict
    davinz_results_dict = {
        'k': [],
        'sqrt_q': [],
        'mmd': [],
        'time_delta': []
    }
        
    # Delete unused device memory
    del X_train
    del y_train
    del model
    del optimizer
    del criterion

    # Get new training dataset that is different from previous one
    # This is the dataset under sale (data market scenario)
    new_train_sizes = np.arange(NEW_TRAIN_SIZE_MIN, NEW_TRAIN_SIZE_MAX, NEW_TRAIN_SIZE_STEP) # np.arange(500, 4100, 100)
    for i, new_train_size in enumerate(new_train_sizes):
        # torch.cuda.empty_cache()

        # Get new training split only with class 1
        df_new, _, _ = get_training_split(
            data_only_0=df_y_0, 
            data_only_1=df_y_1, 
            size=new_train_size, 
            class_1_ratio=NEW_TRAIN_CLASS_1_RATIO, 
            seed=None
        )
        # Check duplication
        df_test = pl.concat([df_new, df_train], how='vertical')
        print(f'\nNum duplicated rows = {df_test.is_duplicated().sum()}')
        print(f"New dataset of size {df_new.shape[0]}")
        print(f"Contains {df_new.filter(pl.col('Response') == 1).shape[0]} datapoints of class 1")

        # Create training and validation tensors
        X_new = torch.from_numpy(df_new.select(pl.exclude(['id', 'Response'])).to_numpy()).float()
        y_new = torch.from_numpy(df_new['Response'].to_numpy().reshape(-1, 1)).float()
        # Create tensor dataset for training
        dataset_new = TensorDataset(X_new, y_new)
        # torch.manual_seed(2298)
        dataloader_new = DataLoader(dataset_new, batch_size=32, shuffle=True) # TODO: try without shuffle

        # Load trained model
        model, optimizer = load_from_file(baseline_checkpoint_path)

        start_training_time = time() 
        # Fine-tune on new dataset
        print(f'\nFine-tuning model {i + 1}/{new_train_sizes.shape[0]}')
        criterion = nn.BCEWithLogitsLoss()
        train_loss_history, valid_loss_history = train_mlp(
            model, 
            dataloader_new, 
            criterion, 
            optimizer, 
            n_epochs=N_EPOCHS_NEW, 
            X_valid=X_valid, 
            y_valid=y_valid
        )
        # Compute fine-tuned model performance
        accuracy_train, fpos_train, fneg_train = get_accuracy(model, X_new, y_new)
        accuracy_valid, fpos_valid, fneg_valid = get_accuracy(model, X_valid, y_valid)
        # Computation time
        time_training_delta = time() - start_training_time
        # Print fine-tuned training and validation accuracy
        print(f'Finished fine-tuning model {i + 1}/{new_train_sizes.shape[0]}')
        print(f'Best Training Loss = {np.min(train_loss_history):.4f}')
        print(f'Training Accuracy = {accuracy_train:.4f}')
        print(f'Training False Positive = {fpos_train}')
        print(f'Training False Negative = {fneg_train}')  

        print(f'\nBest Validation Loss = {np.min(valid_loss_history):.4f}')
        print(f'Validation Accuracy = {accuracy_valid:.4f}')
        print(f'Validation False Positive = {fpos_valid}')
        print(f'Validation False Negative = {fneg_valid}')

        # Save fine-tuned model performance
        model_performance_dict['train_acc'].append(accuracy_train)
        model_performance_dict['train_false_pos'].append(fpos_train)
        model_performance_dict['train_false_neg'].append(fneg_train)
        model_performance_dict['valid_acc'].append(accuracy_valid)
        model_performance_dict['valid_false_pos'].append(fpos_valid)
        model_performance_dict['valid_false_neg'].append(fneg_valid)
        model_performance_dict['train_loss_history'].append(train_loss_history)
        model_performance_dict['valid_loss_history'].append(valid_loss_history)
        model_performance_dict['time_training_delta'].append(time_training_delta)
        model_performance_dict['dataset_size'].append(new_train_size)
        model_performance_dict['n_epochs'].append(N_EPOCHS_NEW)
            
        # Delete mem allocated for fine-tuning
        del model
            
        # Load trained model
        model, optimizer = load_from_file(baseline_checkpoint_path)
        # Perform DAVinz estimation
        score = torch.nan
        k = torch.nan
        sqrt_q = torch.nan
        mmd = torch.nan
        time_delta = torch.nan
        try:
            start_davinz_time = time()
            k, sqrt_q, mmd = davinz_bound(model, predict_probs, X_new, X_valid, y_new)
            time_delta = time() - start_davinz_time
            print(f'DAVinz took {time_delta:.2f}s to run')
            # Add result to dict
        except Exception as e:
            print(f'EXCEPTION: {e}')
            print(traceback.format_exc())
        davinz_results_dict['k'].append(k)
        davinz_results_dict['sqrt_q'].append(sqrt_q)
        davinz_results_dict['mmd'].append(mmd)
        davinz_results_dict['time_delta'].append(time_delta)
        del model

    # Write results to file
    with open(os.path.join(RESULTS_PATH, "mlp_performance.pkl"), 'wb') as file:
        pickle.dump(model_performance_dict, file)

    with open(os.path.join(RESULTS_PATH,'davinz_results.pkl'), 'wb') as file:
        pickle.dump(davinz_results_dict, file)

        



