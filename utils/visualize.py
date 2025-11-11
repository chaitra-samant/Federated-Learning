import matplotlib.pyplot as plt

def plot_metrics(train_losses, test_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Round/Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Round/Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()