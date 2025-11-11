import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import SimpleCNN
from data.data_loader import get_client_loader
from utils.metrics import accuracy  # Implement this in utils/metrics.py

class FederatedClient:
    def __init__(self, client_id, device='cpu', lr=0.001):
        self.client_id = client_id
        self.device = device
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = get_client_loader(client_id, "train")
        self.test_loader = get_client_loader(client_id, "test")

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            print(f"Client {self.client_id} Epoch {epoch+1} Loss: {total_loss/len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Client {self.client_id} Test Accuracy: {acc:.4f}")
        return acc

    def get_parameters(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)

# Example usage:
# if __name__ == '__main__':
#     client = FederatedClient(client_id=1)
#     client.train(epochs=1)
#     client.evaluate()