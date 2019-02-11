
from typing import List

import utilities

import numpy as np
import matplotlib.pyplot as mpl


class LearningDatum:
    def __init__(self, input_nodes, expected_output_nodes):
        self.input_nodes = np.reshape(input_nodes, (len(input_nodes), 1))
        self.expected_output_nodes = np.reshape(expected_output_nodes, (len(expected_output_nodes), 1))


class BackpropResult:
    def __init__(self, bias_delta_gradient, weight_delta_gradient):
        self.bias_delta_gradient = bias_delta_gradient
        self.weight_delta_gradient = weight_delta_gradient


class NeuralNet:
    def __init__(self,
                 layer_sizes,
                 learning_rate=0.5
                 ):
        self._num_layers = len(layer_sizes)
        self._layer_sizes = layer_sizes
        self._learning_rate = learning_rate

        # 2D array - one scalar value for each neuron in each layer
        self._init_biases()

        # 3D array - each neuron has an array of scalars defining the
        # activation weights all the neurons in the previous layer
        self._init_weights()

        print("Neural net initialized!")

    def _init_biases(self):
        self._biases = [np.random.randn(neurons_in_layer, 1) for neurons_in_layer in self._layer_sizes[1:]]

    def _init_weights(self):
        self._weights = []

        for i in range(self._num_layers - 1):
            previous_layer_nodes = self._layer_sizes[i]
            next_layer_nodes = self._layer_sizes[i + 1]

            self._weights.append(np.random.randn(next_layer_nodes, previous_layer_nodes))

    def evaluate_input(self, input_nodes):
        activation = input_nodes

        for i in range(self._num_layers - 1):
            z_val = np.dot(self._weights[i], activation) + self._biases[i]
            activation = NeuralNet.sigmoid(z_val)

        return activation

    def apply_training_data(self,
                            training_data: List[LearningDatum]
                            ):
        if len(training_data) == 0:
            print("Empty training set - nothing to do")
            return

        if not self._validate_training_data_size(training_data):
            print("Invalid training data size!")
            return

        eta = self._learning_rate / len(training_data)

        bias_gradient = [np.zeros(layer_biases.shape) for layer_biases in self._biases]
        weight_gradient = [np.zeros(layer_weights.shape) for layer_weights in self._weights]

        for datum in training_data:
            backprop_result = self._backprop(datum)

            for i in range(self._num_layers - 1):
                bias_gradient[i] = bias_gradient[i] + backprop_result.bias_delta_gradient[i]
                weight_gradient[i] = weight_gradient[i] + backprop_result.weight_delta_gradient[i]

        for i in range(self._num_layers - 1):
            avg_bias_delta = eta * bias_gradient[i]
            self._biases[i] = self._biases[i] - avg_bias_delta

            avg_weight_delta = eta * weight_gradient[i]
            self._weights[i] = self._weights[i] - avg_weight_delta

    def _validate_training_data_size(self,
                                     training_data: List[LearningDatum]
                                     ):
        for datum in training_data:
            if len(datum.input_nodes) != self._layer_sizes[0]:
                print("{} input nodes provided but network only has {}".format(len(datum.input_nodes), self._layer_sizes[0]))
                return False

            if len(datum.expected_output_nodes) != self._layer_sizes[-1]:
                print("{} output nodes expected but network only has {}".format(len(datum.expected_output_nodes),
                                                                               self._layer_sizes[-1]))
                return False

        return True

    def _backprop(self,
                  datum: LearningDatum
                  ):
        inputs = datum.input_nodes

        bias_delta_gradient = [np.zeros(layer_biases.shape) for layer_biases in self._biases]
        weight_delta_gradient = [np.zeros(layer_weights.shape) for layer_weights in self._weights]

        layer_activation_vector = inputs

        activation_vectors = [inputs]

        z_vectors = []

        for i in range(self._num_layers - 1):
            layer_biases = self._biases[i]
            layer_weights = self._weights[i]

            layer_z_vector = np.dot(layer_weights, layer_activation_vector) + layer_biases
            z_vectors.append(layer_z_vector)

            layer_activation_vector = NeuralNet.sigmoid(layer_z_vector)
            activation_vectors.append(layer_activation_vector)

        cost_gradient = self._find_cost_gradient(activation_vectors[-1], datum.expected_output_nodes)
        error_vector = cost_gradient * NeuralNet.sigmoid_derivative(z_vectors[-1])

        bias_delta_gradient[-1] = error_vector
        activation = activation_vectors[-2].transpose()
        weight_delta_gradient[-1] = np.dot(error_vector, activation)

        # Calculate bias and weight deltas from backfed error function
        for i in range(self._num_layers - 2, 0, -1):
            sigmoid_rate = NeuralNet.sigmoid_derivative(z_vectors[i - 1])

            error_vector = np.dot(self._weights[i].transpose(), error_vector)
            error_vector = error_vector * sigmoid_rate

            bias_delta_gradient[i - 1] = error_vector
            weight_delta_gradient[i - 1] = np.dot(error_vector, activation_vectors[i - 1].transpose())

        return BackpropResult(bias_delta_gradient, weight_delta_gradient)

    def _find_cost_gradient(self, output_activations, expected_output):
        cost = output_activations - expected_output
        return cost

    @staticmethod
    def sigmoid(vector):
        return 1.0 / (1.0 + np.exp(-vector))

    @staticmethod
    def sigmoid_derivative(vector):
        return NeuralNet.sigmoid(vector) * (1 - NeuralNet.sigmoid(vector))


def run_model(hls, eta, num_epochs, x_axis, y_axis):
    nn = NeuralNet([64, hls, 2], learning_rate=eta)

    for i in range(num_epochs):
        training_data = generate_learning_data(50)
        nn.apply_training_data(training_data)

        correct_ratio = run_tests(nn, i)
        x_axis.append(i)
        y_axis.append(correct_ratio)


def generate_learning_data(size):
    data = []

    in_vals = np.random.randint(1, 1000, size=size)

    for i in range(len(in_vals)):
        output = [0, 0]
        answer = in_vals[i] % 2 == 0
        if answer:
            output[1] = 1
        else:
            output[0] = 1

        n = in_vals[i]
        byte_array = utilities.int_to_byte_array(n)

        data.append(LearningDatum(byte_array, output))

    return data


def run_tests(nn, epoch_number):
    num_tests = 0
    num_correct = 0

    test_data = generate_learning_data(10000)
    for datum in test_data:
        output_nodes = nn.evaluate_input(datum.input_nodes)

        highest_output_node_index = np.argmax(output_nodes)
        expected_node = np.argmax(datum.expected_output_nodes)

        if highest_output_node_index == expected_node:
            num_correct += 1

        num_tests += 1

    correct_ratio = num_correct / num_tests
    print("{}: {}".format(epoch_number, correct_ratio))
    return correct_ratio


def main():
    np.random.seed(666)

    eta = 4.5
    hidden_layer_size = 30
    number_of_epochs = 30

    x_axis = []
    y_axis = []

    run_model(hidden_layer_size, eta, number_of_epochs, x_axis, y_axis)

    mpl.plot(x_axis, y_axis)
    mpl.title("Neural Net Correctness Over Time (HLS = {}, eta = {})".format(hidden_layer_size, eta))
    mpl.ylim(0, 1)
    mpl.show()


if __name__ == "__main__":
    main()
