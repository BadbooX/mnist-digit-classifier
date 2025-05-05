import ZoidbergMNIST
import ZoidbergPredictor

# Charger les données
zoid = ZoidbergMNIST.ZoidbergMNIST()
zoid.reshape_data()

# Prédiction
predictor = ZoidbergPredictor.ZoidbergPredictor(
    zoid.x_train_flat, zoid.y_train,
    zoid.x_test_flat, zoid.y_test
)

predictor.train_and_predict()
predictor.show_misclassification_stats()
predictor.show_misclassified_images(zoid.x_test_flat, max_images=20)
