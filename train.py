import os
import numpy as np
import random
import torch

from tensorboardX import SummaryWriter
from torchsummary import summary
from torch.utils.data import DataLoader

from utils import model_mapping
from dataset_utils import get_data_paths, ECGDataset
from solver import Solver
from eval_functions import visualize_dataset


config = {
    # Dataset configs
    'data_path': 'data/sleep_dataset',      # path to dataset
    'fs': 256,                              # frequency of the training sequence
    'segment_length': 8,                    # length of every training sequence in seconds
    'load_gender': True,                    # make the dataset return the gender in addition to the ecg
    'split': [.8, .1, .1],                  # Train / validation / test split
    'seed': 123,                            # Random seed for reproducable results

    # Preprocessing
    'instance_normalization': True,         # Set every sequence to zero mean and unit variance
    'high_pass': True,                      # Use high pass filter when loading data
    'notch': True,                          # Use notch filter when loading data

    # Data augmentation
    'noise': 0.01,                          # Add random noise to x to make a denoising autoencoder
    'intensity_range': [0.9, 1.1],

    # Training configs
    'save_path': 'saves',
    'num_workers': 4,
    'do_overfitting': False,                # Overfit on small subset
    'num_overfit': 10,
    'max_num_patients': None,               # Choose None for whole dataset
    'batch_size': 4,
    'num_epochs': 100,
    'save_interval': 10,

    # Model
    'model': 'ConvModelGender',

    # Classification?
    'use_embedding': False,                 # Fine tune the output of a pretrained encoder for classification
    'encoder_path': '',                     # Path to encoder if use_embedding == True

    # Hyperparameters
    'learning_rate': 5e-4,
    'lr_decay_factor': 0.8,
    'lr_decay_patience': 2,
    'weight_decay': 0,                      # L2 weight penalty
    'l1_penalty': 0,                        # L1 weight penalty
    'rec_weight': 1.,                       # scale reconstruction loss during training
    'gender_weight': 1.,                    # scale gender loss during training
    'dropout': 0.0,

    # Visualization
    'show_samples': False,                  # Plot some samples of the dataset before training
    'show_gradient_flow': False,

    # Continue training?
    'continue_training': False,
    'model_path': 'saves/train20190620123534/model1000',
    'solver_path': 'saves/train20190620123534/solver1000',

    'tensorboard_log_dir': 'tensorboard_log/exp_1',
}


""" Make paths absolute """

file_dir = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(file_dir, config['data_path'])
save_path = os.path.join(file_dir, config['save_path'])
model_path = os.path.join(file_dir, config['model_path'])
solver_path = os.path.join(file_dir, config['solver_path'])
tensorboard_log_dir = os.path.join(file_dir, config['tensorboard_log_dir'])
encoder_path = os.path.join(file_dir, config['encoder_path'])


""" Add a seed to have reproducible results """

torch.manual_seed(config['seed'])
random.seed(config['seed'])

""" Configure training with or without cuda """

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(config['seed'])
    kwargs = {'pin_memory': True}
    print("GPU available. Training on {}.".format(device))
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}
    print("No GPU. Training on {}.".format(device))


""" Initialize tensorboard summary writer """

tensorboard_writer = SummaryWriter(config['tensorboard_log_dir'])


""" Load dataset """

print("Loading dataset, length of segments is {}".format(config['segment_length']))

data_paths = get_data_paths(data_path)
random.shuffle(data_paths)

print("Found {} patients in total".format(len(data_paths)))

if config['do_overfitting']:
    dataset_size = config['num_overfit']
    batch_size = min(config['num_overfit'], config['batch_size'])
else:
    dataset_size = len(data_paths)
    batch_size = config['batch_size']

split1 = int(np.floor(config['split'][0] * dataset_size))
split2 = int(np.floor((config['split'][0] + config['split'][1]) * dataset_size))

train_paths = data_paths[:split1]
val_paths = data_paths[split1:split2]
test_paths = data_paths[split2:]

train_dataset = ECGDataset(
    data_paths=train_paths,
    is_test=False,
    fs=config['fs'],
    seg_length=config['segment_length'],
    do_overfitting=config['do_overfitting'],
    instance_normalization=config['instance_normalization'],
    high_pass=config['high_pass'],
    notch=config['notch'],
    intensity_range=config['intensity_range'],
)

val_dataset = ECGDataset(
    data_paths=val_paths,
    is_test=True,
    fs=config['fs'],
    seg_length=config['segment_length'],
    do_overfitting=config['do_overfitting'],
    instance_normalization=config['instance_normalization'],
    high_pass=config['high_pass'],
    notch=config['notch'],
    intensity_range=config['intensity_range'],
)


print("Using {} patients for training and {} for validation.".format(
    len(train_paths), len(val_paths)))

# Create data loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=config['num_workers'],
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=config['num_workers'],
    shuffle=True,
    drop_last=True,
)


""" Visualize data """

if config['show_samples']:
    print("Visualizing part of the data")
    visualize_dataset(dataset, num_samples=5)


""" Initialize model and solver """

if config['continue_training']:
    print("Continuing training with model: {} and solver: {}".format(
        model_path, solver_path)
    )
    model = torch.load(model_path)
    model.to(device)
    solver = Solver()
    solver.optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    solver.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        solver.optim,
        factor=config['lr_decay_factor'],
        patience=config['lr_decay_patience'],
        min_lr=1e-6,
        verbose=True
    )
    solver.load(solver_path, device=device)
    rec_criterion = None
    gender_criterion = None
    optimizer = None
    scheduler = None
else:
    print("Initializing model. Model used: {}".format(config['model']))
    model = model_mapping(
        config['model'],
        input_dim=config['fs'] * config['segment_length'],
        dropout=config['dropout'],
    )

    model.to(device)
    solver = Solver()

    # Specify loss functions
    rec_criterion = None
    gender_criterion = None
    if model.is_autoencoder:
        rec_criterion = torch.nn.SmoothL1Loss()
    if model.predict_gender:
        gender_criterion = torch.nn.CrossEntropyLoss()

    # Select optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config['lr_decay_factor'],
        patience=config['lr_decay_patience'],
        min_lr=1e-6,
        verbose=True
    )


""" Load encoder if necessary """

if config['use_embedding']:
    print('Loading encoder...')
    encoder = torch.load(encoder_path, map_location=device).encoder
    encoder.eval().to(device)
else:
    encoder = None


""" Print model summary """

print("Printing model summary...")
example_input, gender = next(iter(train_loader))
example_input = example_input.to(device)
if config['use_embedding']:
    example_input = encoder(example_input)
print(summary(model, input_size=example_input.shape[1:]))


""" Perform training """

if __name__ == "__main__":
    solver.train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        scheduler=scheduler,
        tensorboard_writer=tensorboard_writer,
        optim=optimizer,
        rec_criterion=rec_criterion,
        gender_criterion=gender_criterion,
        rec_weight=config['rec_weight'],
        gender_weight=config['gender_weight'],
        l1_penalty=config['l1_penalty'],
        noise=config['noise'],
        num_epochs=config['num_epochs'],
        save_after_epochs=config['save_interval'],
        device=device,
        encoder=encoder,
        show_gradient_flow=config['show_gradient_flow'],
    )
