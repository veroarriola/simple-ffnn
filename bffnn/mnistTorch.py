import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ignite.metrics as metrics


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


    def full_train(self, num_epocs, data_set, save_every=1, save_path="save_dir"):
        '''
        Train for the number of epocs.
        save_every: save params and statistics every indicated steps if cero or
                    less it doesn't save.
        '''
        with open(f"{save_path}/stats.txt", "a") as myfile:
            myfile.write("Epoch\tLoss\tTest accuracy\n")

        last_loss = -1
        for epoch in range(num_epocs):
            last_loss = self.train_epoch(data_set.trainloader)
            print(f"Completed epoch {epoch} with loss {last_loss}")

            if save_every > 0 and epoch % save_every == 0:
                self.net.save(f"{save_path}/{epoch}")
                accuracy = self.accuracy(data_set.testloader_all)
                with open(f"{save_path}/stats.txt", "a") as myfile:
                    myfile.write(f"{epoch}\t{last_loss}\t{accuracy}\n")
        return last_loss


def example():
    IMG_INPUT_SIZE = 28
    HIDDEN1_SIZE = 16
    HIDDEN2_SIZE = 9
    OUTPUT_SIZE = 10

    DATA_DIR = "../nn-data"
    BATCH_SIZE = 10

    LEARNING_RATE = 0.003
    NUM_EPOCHS = 100
    SAVE_EVERY = 1
    SAVE_PATH = "../save_dir"

    from pathlib import Path
    Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

    net = MNISTNet(IMG_INPUT_SIZE * IMG_INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE)
    data_set = MNISTDataSet(DATA_DIR, BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(net, criterion, optimizer)
    last_lost = trainer.full_train(NUM_EPOCHS, data_set, SAVE_EVERY, SAVE_PATH)
    print("Loss after training is", last_lost)


if __name__ == '__main__':
    example()