import snntorch as snn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid

#Initialises batch size
batchSize = 8

#Sets up device - uses cuda GPU if available, otherwise uses CPU
dtype = torch.float
device = torch.device("cpu")

#Defines a transform to resize each image to 28x28, makes sure they are grey (to reduce number of channels), converts the image to a
#tensor, and normalises each pixel value between -1 and 1
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

#Uploads data from file
dataset = datasets.ImageFolder(root = "HandGesture/images", transform = transform)

#Splits data randomly into training (80% of whole dataset) and testing (20% of whole dataset)
trainSize = int(0.8 * len(dataset))
testSize = len(dataset) - trainSize
trainDataset, testDataset = random_split(dataset, [trainSize, testSize])

ftrain = datasets.FashionMNIST("data/fashionmnist", train=True, download=True, transform=transform)
ftest = datasets.FashionMNIST("data/fashionmnist", train=False, download=True, transform=transform)

#Creates dataloaders for training and test data
trainDataLoader = DataLoader(ftrain, batch_size = batchSize, shuffle = True, drop_last = True)
testDataLoader = DataLoader(ftest, batch_size = batchSize, shuffle = True, drop_last = True)

#Initialises the number of training steps and membrane potential decay rate (beta)
trainingStepNumber = 20
beta = 0.95

#Class that defines the network
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        #Layer 1: spiking convolutional layer (with pooling)
        self.conv1 = nn.Conv2d(1, 16, 5, padding="same")
        self.lif1 = snn.Leaky(beta=beta)
        self.maxPool1 = nn.MaxPool2d(2)

        #Layer 2: spiking convolutional layer (with pooling)
        self.conv2 = nn.Conv2d(16, 64, 5, padding="same")
        self.lif2 = snn.Leaky(beta=beta)
        self.maxPool2 = nn.MaxPool2d(2)

        #Layer 3: fully connected, spiking output layer
        self.fullyConnected = nn.Linear(7*7*64, 10)
        self.lif3 = snn.Leaky(beta=beta)

    def forward(self, input):

        #Initialises membrane potentials of neurons in each layer
        membranePotential1 = self.lif1.init_leaky()
        membranePotential2 = self.lif2.init_leaky()
        membranePotential3 = self.lif3.init_leaky()
       
        #Lists to store the output spikes and membrane potentials
        finalSpikes = []
        finalPotentials = []

        #Repeats training step
        for step in range(trainingStepNumber):

            #First convolution gives a current
            current1 = self.conv1(input)

            #Current may generate a spike and change membrane potential
            spike1, membranePotential1 = self.lif1(self.maxPool1(current1), membranePotential1)

            #Second convolution gives a current, which may generate a spike and change membrane potential
            current2 = self.conv2(spike1)
            spike2, membranePotential2 = self.lif2(self.maxPool2(current2), membranePotential2)

            #A fully connected layer sorts the images to their gesture types and this information is spiked
            current3 = self.fullyConnected(spike2.flatten(1))
            spike3, membranePotential3 = self.lif3(current3, membranePotential3)

            #The presence
            finalSpikes.append(spike3)
            finalPotentials.append(membranePotential3)

        return torch.stack(finalSpikes, dim=0), torch.stack(finalPotentials, dim=0)
       
#Loads network to device
network = Network().to(device)

#Defines loss funtion and optimiser
lossFunction = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(network.parameters(), lr=1e-4, betas=(0.9, 0.999))

#Initialises number of epochs
numberOfEpochs = 1

counter = 0

#Training loop
for epoch in range(numberOfEpochs):
    batch = iter(trainDataLoader)

    #Training loop for each batch
    for data, gestures in batch:
        data = data.to(device)
        gestures = gestures.to(device)

        #Forward pass
        network.train()
        spikes, _ = network(data)

        #Sums loss over time
        loss = torch.zeros((1), dtype=dtype, device=device)
        loss = lossFunction(spikes.sum(0), gestures)

        #Calculates gradient and updates weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        #Prints loss every ten iterations
        if counter % 10 == 0:
            print(f"Iteration: {counter} \t Train Loss: {loss.item()}")
        counter += 1

        if counter > 520:
            break
   
#Function that measures the accuracy of the model using the test data
def measureAccuracy(model, dataLoader):
    with torch.no_grad():
        model.eval()
        runningLength = 0
        accuracy = 0

        #Loops through each image in the test data and sees if the predicted gesture is equal to the real gesture
        for data, gestures in iter(dataLoader):
            data = data.to(device)
            gestures = gestures.to(device)

            #Forward Pass
            spikes, _ = model(data)
            numberOfSpikes = spikes.sum(0)
            _, maxNumberOfSpikes = numberOfSpikes.max(1)

            #Checks if the predicted gesture (represented by number of spikes) is equal to the real gesture
            numberOfCorrectGestures = (maxNumberOfSpikes == gestures).sum()

            #Calculates final accuracy
            runningLength += len(gestures)
            accuracy += numberOfCorrectGestures
            finalAccuracy = (accuracy / runningLength)
        return finalAccuracy.item()

print(f"Network Accuracy: {measureAccuracy(network, testDataLoader)}")