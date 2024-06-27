# Add the ArabicNER package to the system path
import sys
import argparse
sys.path.append('ArabicNER/')

# Import train function
from arabiner.bin.train import main as train



# Setup the model arguments
args_dict = {
    # Model output path to save artifacts and model predictions
    "output_path": "output_model/",

    # train/test/validation data paths
    "train_path": "data/train.txt",
    "test_path": "data/test.txt",
    "val_path": "data/val.txt",

    # seed for randomization
    "seed": 1,

    "batch_size": 32,

    # Nmber of workers for the dataloader
    "num_workers": 1,

    # GPU/device Ids to train model on
    # For two GPUs use [0, 1]
    # For three GPUs use [0, 1, 2], etc.
    "gpus": [0],

    # Overwrite data in output_path directory specified above
    "overwrite": True,

    # How often to print the logs in terms of number of steps
    "log_interval": 300,

    # Data configuration
    # Here we specify the dataset class and there are two options:
    #  arabiner.data.datasets.DefaultDataset: for flat NER
    #  arabiner.data.datasets.NestedTagsDataset: for nested NER
    #
    # kwargs: keyword arguments to the dataset class
    # This notebook used the DefaultDataset for flat NER
    "data_config": {
        "fn": "arabiner.data.datasets.DefaultDataset",
        "kwargs": {"max_seq_len": 512, "bert_model": "aubmindlab/bert-base-arabertv02-twitter"}
    },

    # Neural net configuration
    # There are two NNs:
    #   arabiner.nn.BertSeqTagger: flat NER tagger
    #   arabiner.nn.BertNestedTagger: nested NER tagger
    #
    # kwargs: keyword arguments to the NN
    # This notebook uses BertSeqTagger for flat NER tagging
    "network_config": {
        "fn": "arabiner.nn.BertSeqTagger",
        "kwargs": {"dropout": 0.3, "bert_model": "aubmindlab/bert-base-arabertv02-twitter"}#-twitter
    },

    # Model trainer configuration
    #
    #  arabiner.trainers.BertTrainer: for flat NER training
    #  arabiner.trainers.BertNestedTrainer: for nested NER training
    #
    # kwargs: keyword arguments to arabiner.trainers.BertTrainer
    #         additional arguments you can pass includes
    #           - clip: for gradient clpping
    #           - patience: number of epochs for early termination
    # This notebook uses BertTrainer for fat NER training
    "trainer_config": {
        "fn": "arabiner.trainers.BertTrainer",
        "kwargs": {"max_epochs": 50}
    },

    # Optimizer configuration
    # Our experiments use torch.optim.AdamW, however, you are free to pass
    # any other optmizers such as torch.optim.Adam or torch.optim.SGD
    # lr: learning rate
    # kwargs: keyword arguments to torch.optim.AdamW or whatever optimizer you use
    #
    # Additional optimizers can be found here:
    # https://pytorch.org/docs/stable/optim.html
    "optimizer": {
        "fn": "torch.optim.AdamW",
        "kwargs": {"lr": 0.0001}
    },

    # Learning rate scheduler configuration
    # You can pass a learning scheduler such as torch.optim.lr_scheduler.StepLR
    # kwargs: keyword arguments to torch.optim.AdamW or whatever scheduler you use
    #
    # Additional schedulers can be found here:
    # https://pytorch.org/docs/stable/optim.html
    "lr_scheduler": {
        "fn": "torch.optim.lr_scheduler.ExponentialLR",
        "kwargs": {"gamma": 1}
    },

    # Loss function configuration
    # We use cross entropy loss
    # kwargs: keyword arguments to torch.nn.CrossEntropyLoss or whatever loss function you use
    "loss": {
        "fn": "torch.nn.CrossEntropyLoss",
        "kwargs": {}
    }
}

# Convert args dictionary to argparse namespace
args = argparse.Namespace()
args.__dict__ = args_dict



# Start training the model 
train(args)