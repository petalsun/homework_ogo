# Визуализация результатов
from matplotlib import pyplot as plt
import numpy as np

from train import *



def first_task():

    plt.figure(figsize=(12, 8))

    # Графики потерь
    plt.subplot(2, 2, 1)
    for name in models:
        plt.plot(results[name]['train_losses'], label=f'{name} Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for name in models:
        plt.plot(results[name]['test_losses'], label=f'{name} Test')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Графики точности
    plt.subplot(2, 2, 3)
    for name in models:
        plt.plot(results[name]['train_accs'], label=f'{name} Train')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 4)
    for name in models:
        plt.plot(results[name]['test_accs'], label=f'{name} Test')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    

def second_task():
    # Визуализация результатов
    plt.figure(figsize=(12, 8))

    # Графики потерь
    plt.subplot(2, 2, 1)
    for name in cifar_models:
        plt.plot(cifar_results[name]['train_losses'], label=f'{name} Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for name in cifar_models:
        plt.plot(cifar_results[name]['test_losses'], label=f'{name} Test')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Графики точности
    plt.subplot(2, 2, 3)
    for name in cifar_models:
        plt.plot(cifar_results[name]['train_accs'], label=f'{name} Train')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 4)
    for name in cifar_models:
        plt.plot(cifar_results[name]['test_accs'], label=f'{name} Test')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    