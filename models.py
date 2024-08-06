import torch
import torch.nn as nn
from ftnna import CollaborativeLogisticClassifier

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class CollaborativeSimpleMLP(nn.Module):
    def __init__(self, num_classifiers=7):
        super(CollaborativeSimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.collaborative_classifier = CollaborativeLogisticClassifier(64, num_classifiers)

    def forward(self, x):
        # TODO: can be made dynamic using modulelist
        with torch.no_grad():
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
        x = self.collaborative_classifier(x)
        return x