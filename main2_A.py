import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def sigmoid_derivative(sigmoid_output):
    result = sigmoid_output * (1 - sigmoid_output)
    return result


# MSE loss function
def mse_loss(predictions, targets):
    result = np.mean((predictions - targets) ** 2)
    return result

def mse_loss_derivative(predictions, targets):
    result = 2 * (predictions - targets) / targets.size
    return result


# Initialize weights with Xavier
def initialize(input_size, output_size):
    # Constraint for initialization
    limit = np.sqrt(2.0 / (input_size + output_size))
    ## Initialize weights and biases
    weights = np.random.randn(input_size, output_size) * limit
    bias = np.zeros((1, output_size))
    return weights, bias


# Line layer
def linear_forward(x, weights, bias):
    # Forward pass: x * W + b
    output = np.dot(x, weights) + bias
    # Save data for backward
    result = (x, weights)
    return output, result

def linear_backward(grad_output, result):
    # Backward pass: calculating gradients
    x, weights = result
    # Gradient weights
    grad_weights = np.dot(x.T, grad_output)
    # Offset gradient
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)
    grad_input = np.dot(grad_output, weights.T)
    return grad_input, grad_weights, grad_bias

def forward(x, params):
    # Hidden layer: linear transform + tanh
    z1, result1 = linear_forward(x, params['w1'], params['b1'])
    a1 = tanh(z1)

    # Output layer: linear transform + sigmoid
    z2, result2 = linear_forward(a1, params['w2'], params['b2'])
    predictions = sigmoid(z2)
    # Save the date for the backward
    results = (result1, result2, a1)
    return predictions, results

def backward(predictions, targets, result):
    
    # Date unpacking    
    result1, result2, a1 = result
    
    # Gradients for the output layer
    grad_loss = mse_loss_derivative(predictions, targets)
    grad_z2 = grad_loss * sigmoid_derivative(predictions)
    grad_a1, grad_w2, grad_b2 = linear_backward(grad_z2, result2)

    # Gradients for hidden layer
    grad_z1 = grad_a1 * tanh_derivative(a1)
    _, grad_w1, grad_b1 = linear_backward(grad_z1, result1)

    grads = {
        'w1': grad_w1, 'b1': grad_b1,
        'w2': grad_w2, 'b2': grad_b2
    }
    return grads

#Updating parameters
def update(params, grads, learning_rate, momentum, velocities):
    for key in params:
        if momentum > 0:
            velocity = momentum * velocities.get(key, 0) - learning_rate * grads[key]
            velocities[key] = velocity
            params[key] += velocity
        else:
            params[key] -= learning_rate * grads[key]
    return params, velocities

def train(x, y, params, epochs, learning_rate, momentum):
    velocities = {key: np.zeros_like(value) for key, value in params.items()}
    losses = []

    for epoch in range(epochs):
        predictions, caches = forward(x, params)

        loss = mse_loss(predictions, y)
        losses.append(loss)

        grads = backward(predictions, y, caches)

        params, velocities = update(params, grads, learning_rate, momentum, velocities)

    return params, losses

def tanh(x):
    result = np.tanh(x)
    return result

def tanh_derivative(tanh_output):
    result = 1 - tanh_output**2
    return result

def test_model(x, y, params):
    predictions, _ = forward(x, params)
    correct = 0
    print("\nTesting the trained model:")
    for i in range(len(x)):
        pred = predictions[i][0]
        pred_label = 1 if pred > 0.5 else 0
        print(f"Input: {x[i]}, Predicted: {pred:.4f}, Pred Label: {pred_label}, Actual: {y[i][0]}")
        if pred_label == y[i][0]:
            correct += 1
    accuracy = correct / len(x) * 100
    print(f"Accuracy: {accuracy:.2f}%")

def plot_losses(losses_no_momentum, losses_with_momentum):
    plt.figure(figsize=(10, 6))
    plt.plot(losses_no_momentum, label="Without Momentum")
    plt.plot(losses_with_momentum, label="With Momentum")
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.show()

def relu(x):
    result = np.maximum(0, x)
    return result

def relu_deriv(output):
    result = (output > 0).astype(float)
    return result

def main():
    # XOR
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    # AND
    # x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0], [0], [0], [1]])
    # OR
    # x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0], [1], [1], [1]])

    params = {
        'w1': np.random.randn(2, 4),
        'b1': np.zeros((1, 4)),
        'w2': np.random.randn(4, 1),
        'b2': np.zeros((1, 1))
    }

    print("training without momentum")
    params_no_momentum = params.copy()
    params_no_momentum, losses_no_momentum = train(x, y, params_no_momentum, epochs=500, learning_rate=0.07, momentum=0.0)
    test_model(x, y, params_no_momentum)

    print("\ntraining with momentum")
    params_with_momentum = params.copy()
    params_with_momentum, losses_with_momentum = train(x, y, params_with_momentum, epochs=500, learning_rate=0.07, momentum=0.5)
    test_model(x, y, params_with_momentum)

    plot_losses(losses_no_momentum, losses_with_momentum)

if __name__ == "__main__":
    main()
