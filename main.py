import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import SimpleMLP, CollaborativeSimpleMLP
from utils import get_mnist, train, test, get_confusion_matrix, inject_variation, copy_params
from ftnna import dnn_favorable_searching_code, train_collaborative, test_collaborative, CollaborativeLoss

import argparse

def main(args):
    # Process args
    seed = args.seed
    batch_size = args.batch_size
    train_epochs = args.train_epochs
    finetune_epochs = args.finetune_epochs
    num_classifiers = args.num_classifiers
    mean, std_dev = args.mean, args.std_dev
    hamming_distance = args.hamming_distance

    # Initialize seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Step 1: Pre-train a simple network on MNIST
    train_loader, test_loader = get_mnist(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(train_epochs):
        train(model, train_loader, optimizer, criterion, device)
        accuracy = test(model, test_loader, device)
        print(f"Epoch {epoch+1}/{train_epochs}, Accuracy: {accuracy:.4f}")

    # Inject weight (resistance) variation and compute accuracy again
    log_normal_dist = torch.distributions.LogNormal(mean, std_dev)
    inject_variation(log_normal_dist, model.fc1, inplace=True)
    inject_variation(log_normal_dist, model.fc2, inplace=True)

    accuracy = test(model, test_loader, device)
    print(f"After noise injection, Accuracy: {accuracy:.4f}")

    # Step 2: Extract confusion matrix
    conf_matrix = get_confusion_matrix(model, test_loader, device)

    # Step 3: Replace softmax classifiers with collaborative logistic classifiers
    # Transfer weights from the pre-trained model to the modified model
    modified_model = CollaborativeSimpleMLP().to(device)
    copy_params(model.fc1, modified_model.fc1)
    copy_params(model.fc2, modified_model.fc2)

    # Step 4: Fine-tune the logistic classifiers
    codeword_dict, codewords = dnn_favorable_searching_code(conf_matrix, num_classifiers, hamming_distance=hamming_distance) # Generate codewords

    collaborative_loss = CollaborativeLoss(codewords, modified_model)
    optimizer = optim.Adam(modified_model.collaborative_classifier.parameters())

    for epoch in range(finetune_epochs):
        train_collaborative(modified_model, train_loader, optimizer, collaborative_loss, codewords, device)
        accuracy = test_collaborative(modified_model, test_loader, codewords, device)
        print(f"Epoch {epoch+1}/{finetune_epochs}, Accuracy: {accuracy:.4f}")

    # Final evaluation
    final_accuracy = test_collaborative(modified_model, test_loader, codewords, device)
    print(f"Final Accuracy with Collaborative Logistic Classifiers: {final_accuracy:.4f}")

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description='Run a script with configurable parameters.')

    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed for reproducibility. Default is 0.')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training. Default is 64.')
    parser.add_argument('--train_epochs', type=int, default=10, 
                        help='Number of epochs for training. Default is 10.')
    parser.add_argument('--finetune_epochs', type=int, default=10, 
                        help='Number of epochs for fine-tuning. Default is 10.')
    parser.add_argument('--num_classifiers', type=int, default=7, 
                        help='Number of classifiers. Default is 7.')
    parser.add_argument('--hamming_distance', type=int, default=3, 
                        help='Hamming distance for code-searching. Default is 3.')
    parser.add_argument('--mean', type=float, default=0, 
                        help='Mean of the log-normal distribution. Default is 0.')
    parser.add_argument('--std_dev', type=float, default=0.5, 
                        help='Standard deviation of the log-normal distribution. Default is 0.5.')

    args = parser.parse_args()
    main(args)