import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import seaborn as sns

class ZoidbergMNIST:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
    
    def stats(self):
        """Affiche la distribution des chiffres."""
        sns.countplot(x=self.y_train)
        plt.title("Distribution des chiffres (train)")
        plt.show()
        sns.countplot(x=self.y_test)
        plt.title("Distribution des chiffres (test)")
        plt.show()

    def show_image(self, index=0, dataset='train'):
        """Affiche une image spécifique"""
        if dataset == 'train':
            img = self.x_train[index]
            label = self.y_train[index]
        else:
            img = self.x_test[index]
            label = self.y_test[index]
        plt.imshow(img, cmap='gray')
        plt.title(f"Chiffre : {label}")
        plt.show()
    
    def show_mean_digits(self):
        """Affiche la moyenne des images pour chaque chiffre"""
        for digit in range(10):
            mean_img = np.mean(self.x_train[self.y_train == digit], axis=0)
            plt.imshow(mean_img, cmap='gray')
            plt.title(f"Moyenne du chiffre {digit}")
            plt.show()
    
    def reshape_data(self):
        """Transforme les images en vecteurs (n,784)"""
        self.x_train_flat = self.x_train.reshape(-1, 784)
        self.x_test_flat = self.x_test.reshape(-1, 784)

zoid = ZoidbergMNIST()
zoid.reshape_data()