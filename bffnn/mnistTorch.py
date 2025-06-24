import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import yaml
from pathlib import Path
#import ignite.metrics as metrics  # TODO: would be more efficient for accuracy


class MNISTDataSet:
    '''
    Conveniently manges MNIST data.
    '''
    def __init__(self, data_dir, batch_size=4, num_workers=2) -> None:
        # datasets
        self.trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transforms.ToTensor())
        self.testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transforms.ToTensor())
        self.classes = self.trainset.classes

        # dataloaders
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
        
        # one batch
        self.trainloader_all = torch.utils.data.DataLoader(
            self.trainset, batch_size=len(self.trainset),
            shuffle=True, num_workers=num_workers)

        self.testloader_all = torch.utils.data.DataLoader(
            self.testset, batch_size=len(self.testset),
            shuffle=False, num_workers=num_workers)


class MNISTNet(nn.Module):
    '''
    Input -> hidden -> hidden -> Output network
    '''
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(MNISTNet, self).__init__()

        def activation_hook(module, args, output):
            if module == self.fc1:
                self.ac1 = output.detach()
            elif module == self.fc2:
                self.ac2 = output.detach()
            elif module == self.fco:
                self.aco = output.detach()

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fco = nn.Linear(hidden2_size, output_size)
        
        self.activation = nn.Sigmoid()

        self.fc1.register_forward_hook(activation_hook)
        self.fc2.register_forward_hook(activation_hook)
        self.fco.register_forward_hook(activation_hook)

    def forward(self, x):
        '''
        Receives input data in 1 tensor and returns results.
        '''
        x = x.view(-1, 28 * 28)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fco(x))
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    

class Trainer:
    '''
    Convenience class that contains training routines
    for the given net, criterion and optimizer.
    '''
    def __init__(self, net, criterion, optimizer) -> None:
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
    
    def train_step(self, inputs, labels):
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss

    def train_epoch(self, trainloader):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            loss = self.train_step(inputs, labels)
            running_loss += loss.item()
        return running_loss / trainloader.batch_size
    
    def accuracy(self, dataloader):
        acc = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            outputs = self.net(inputs)
            labels_pred = torch.argmax(outputs, dim=1)
            acc += torch.sum(labels_pred == labels) / len(labels)
        return acc / len(dataloader)


    def full_train(self, num_epochs, data_set, save_every=1, save_dir=None):
        '''
        Train for the number of epocs.
        save_every: save params and statistics every indicated steps if cero or
                    less it doesn't save.
        '''
        if save_every > 0 and not save_dir:
            raise Exception
        
            with open(f"{save_dir}/stats.txt", "a") as myfile:
                myfile.write("Epoch\tLoss\tTest accuracy\n")

        last_loss = -1
        for epoch in range(num_epochs):
            last_loss = self.train_epoch(data_set.trainloader)
            print(f"Completed epoch {epoch} with loss {last_loss}")

            if save_every > 0 and epoch % save_every == 0:
                self.net.save(f"{save_dir}/{epoch}.pth")
                accuracy = self.accuracy(data_set.testloader_all)
                with open(f"{save_dir}/stats.txt", "a") as myfile:
                    myfile.write(f"{epoch}\t{last_loss}\t{accuracy}\n")
        return last_loss


class NetConfig:
    '''
    Contains configuration parameters for creation/trainning of a net.
    '''
    CONFIG_FILE_NAME = 'nn_config.yaml'
    DATA_DIR = 'nn-data'
    RESULTS_DIR = 'nn-saved'

    def __init__(self, module_dir, model_dir):
        '''
        module_dir: directory where this script is located at excecution time
        model_dir: name of the directory where training info is/will be placed
        '''
        self.module_dir = module_dir
        self._model_dir = os.path.join(module_dir, NetConfig.RESULTS_DIR, model_dir)
        self._data_dir = os.path.join(module_dir, NetConfig.DATA_DIR)
        self._load_config()

    @property
    def model_dir(self):
        return self._model_dir
    
    @property
    def data_dir(self):
        return self._data_dir
    
    def files_of_weights(self):
        weights_files = [file_name for file_name in os.listdir(self.model_dir) if file_name.endswith(".pth")]
        weights_files.sort(key=lambda k: int(k[:-4]))
        return weights_files

    def __getitem__(self, key):
        return self.params[key]

    def _load_config(self):
        '''
        Loads the configuration dictionary for creation/trainning of a net.
        '''
        load_path = os.path.join(self.model_dir, NetConfig.CONFIG_FILE_NAME)
        with open(load_path) as stream:
            try:
                net_config = yaml.safe_load(stream)
                print(f"Loaded config from {load_path}")
                self.params = net_config
            except yaml.YAMLError as exc:
                print(exc)

    def save_config(self):
        '''
        Saves the configuration dictionary for creation/trainning of a net.
        '''
        save_path = os.path.join(self.model_dir, NetConfig.CONFIG_FILE_NAME)

        # if dir does not exist, create it.
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # save configuration dictionary
        with open(save_path, 'w') as outfile:
            yaml.dump(self, outfile)
            print(f"yaml writen to {save_path}")

    def __str__(self):
        return f"MODEL_DIR: {self.model_dir}\nMODEL_DIR: {self.data_dir}\n{str(self.params)}"


def example(module_dir, model_dir):
    net_config = NetConfig(module_dir, model_dir)

    net = MNISTNet(net_config['IMG_INPUT_SIZE'] * net_config['IMG_INPUT_SIZE'],
                   net_config['HIDDEN1_SIZE'],
                   net_config['HIDDEN2_SIZE'],
                   net_config['OUTPUT_SIZE'])
    data_set = MNISTDataSet(net_config.data_dir, net_config['BATCH_SIZE'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=net_config['LEARNING_RATE'])
    trainer = Trainer(net, criterion, optimizer)
    last_lost = trainer.full_train(net_config['NUM_EPOCHS'], data_set, net_config['SAVE_EVERY'], net_config.model_dir)
    print("Loss after training is", last_lost)


if __name__ == '__main__':
    mnist_module_path = os.path.join(os.getcwd(), "simple-ffnn")
    example(mnist_module_path, "net_001")
