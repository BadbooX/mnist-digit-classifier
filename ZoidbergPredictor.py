class ZoidbergPredictor:
    def __init__(self, x_train, y_train, x_test, y_test):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(x_train)
        self.x_test = self.scaler.transform(x_test)
        self.y_train = y_train
        self.y_test = y_test


    def train_and_predict(self):
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(150, 100, 68),  
            max_iter=500,               
            random_state=42,             
            early_stopping=True,        
            verbose=True                
        )
        self.model.fit(self.x_train, self.y_train)
        self.y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return self.y_pred


    def show_misclassification_stats(self):
        from collections import Counter
        import seaborn as sns
        import matplotlib.pyplot as plt

        errors = self.y_test != self.y_pred
        total_per_digit = Counter(self.y_test)
        errors_per_digit = Counter(self.y_test[errors])
        error_rates = {d: errors_per_digit[d]/total_per_digit[d]*100 for d in range(10)}

        sns.barplot(x=list(error_rates.keys()), y=list(error_rates.values()))
        plt.title("Taux d’erreur par chiffre (%)")
        plt.ylabel("Pourcentage")
        plt.xlabel("Chiffre")
        plt.tight_layout()
        plt.savefig("erreurs_par_chiffre.png")
        print("📊 Graphique enregistré sous 'erreurs_par_chiffre.png'")

    def show_misclassified_images(self, x_original, max_images=10):
        import matplotlib.pyplot as plt

        errors = self.y_test != self.y_pred
        x_errors = x_original[errors]
        y_errors = self.y_test[errors]
        y_preds = self.y_pred[errors]

        print(f"🔍 Nombre total d’erreurs : {len(x_errors)}")

        for i in range(min(max_images, len(x_errors))):
            img = x_errors[i].reshape(28, 28)
            true_label = y_errors[i]
            predicted_label = y_preds[i]

            plt.imshow(img, cmap='gray')
            plt.title(f"Vrai : {true_label} | Prédit : {predicted_label}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"erreur_{i}_vrai{true_label}_pred{predicted_label}.png")
            print(f"💾 erreur_{i}_vrai{true_label}_pred{predicted_label}.png enregistrée")
