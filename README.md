# MNIST Digit Classifier

Ce projet est un mini système de reconnaissance de chiffres manuscrits basé sur le dataset MNIST. Il utilise un réseau de neurones MLP (`scikit-learn`) et propose des visualisations claires ainsi qu’une analyse des erreurs.

||

This project is a lightweight handwritten digit recognition system based on the MNIST dataset.  
It uses an MLP neural network (`scikit-learn`) and provides clear visualizations along with an error analysis.

## Contenu du projet / Project Content

- `main.py` – Script principal qui orchestre le tout.
- `ZoidbergMNIST.py` – Classe pour charger et préparer les données MNIST.
- `ZoidbergPredictor.py` – Entraînement, prédictions et analyse d’erreurs.
- `.gitignore` – Pour ne pas versionner les fichiers inutiles.
- `requirements.txt` – Dépendances à installer.

||

- `main.py` – Main script that orchestrates everything.
- `ZoidbergMNIST.py` – Class to load and preprocess MNIST data.
- `ZoidbergPredictor.py` – Model training, prediction, and error analysis.
- `.gitignore` – Excludes unnecessary files from version control.
- `requirements.txt` – List of dependencies to install.

##  Prérequis / Prerequisites

- Python 3.8+
- pip

## Installation

```bash
git clone https://github.com/BadbooX/mnist-digit-classifier.git
cd mnist-digit-classifier
python -m venv venv
source venv/bin/activate  # sous Windows : venv\Scripts\activate
pip install -r requirements.txt



