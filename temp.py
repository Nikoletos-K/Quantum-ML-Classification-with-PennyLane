class QMLBinaryClassifier:
    def __init__(self, n_qubits, num_layers, optimizer, learning_rate=0.1):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.weights = 0.01 * np.random.randn(num_layers, n_qubits)
        self.bias = np.array(0.0, requires_grad=True)
        self.opt = optimizer(learning_rate)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev)
        def circuit(weights, x):
            qml.templates.AngleEmbedding(x, wires=range(n_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return qml.expval(qml.PauliZ(0))
        
        self.circuit = circuit

    def variational_classifier(self, weights, bias, x):
        return self.circuit(weights, x) + bias
    
    def cost(self, weights, bias, X, Y):
        predictions = [self.variational_classifier(weights, bias, x) for x in X]
        return np.mean((np.array(predictions) - Y) ** 2)
    
    def train(self, X, Y, epochs=50, batch_size=5):
        for it in range(epochs):
            batch_index = np.random.randint(0, len(X), (batch_size,))
            X_batch = X[batch_index]
            Y_batch = Y[batch_index]
            self.weights, self.bias, _, _ = self.opt.step(self.cost, self.weights, self.bias, X_batch, Y_batch)
            
            predictions = [self.variational_classifier(self.weights, self.bias, x) for x in X]
            acc = np.mean((np.array(predictions) > 0.5) == Y)
            print(f"Iter: {it+1} | Cost: {self.cost(self.weights, self.bias, X, Y):.4f} | Accuracy: {acc:.4f}")
    
    def evaluate(self, X, Y):
        predictions = np.array([self.variational_classifier(self.weights, self.bias, x) for x in X])
        accuracy = np.mean((predictions > 0.5) == Y)
        print(f"Accuracy: {accuracy:.4f}")
        return predictions, accuracy
    
    def visualize_performance(self, X, Y, predictions):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        correctly_classified = (predictions > 0.5) == Y
        incorrectly_classified = ~correctly_classified

        ax.scatter(X[correctly_classified, 0], X[correctly_classified, 1], X[correctly_classified, 2], c='green', marker='o', label='Correctly Classified')
        ax.scatter(X[incorrectly_classified, 0], X[incorrectly_classified, 1], X[incorrectly_classified, 2], c='red', marker='x', label='Incorrectly Classified')

        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        ax.set_title("Data Classification Performance")
        ax.legend()
        plt.show()
