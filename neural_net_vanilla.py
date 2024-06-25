# Feedforward neural network in vanilla python
#
# This framework is written without numpy/pytorch/tensorflow/scikit-learn for
# educational purposes. It is wildly inefficient.

import math
import random
from typing import List, Callable, TypedDict

# The Node, Layer, and Net classes constitute a complete feed forward neural
# net framework with forward inference (`Net.forward`), backward propagation
# (`Net.backward`), and batch gradient descent training (`Net.batch_train`).
# The object oriented approach makes the structure of the network more
# transparent, at the cost of obscuring some of the math (in that the weights
# and biases are maintained within the Node objects, rather than directly
# passed between the computing functions as in a more procedural approach).

class Node:
    def __init__(self, L, N, weights:int, activation_fx:Callable, bias:float = 0.):
        self.layer = L     # layer index, debugging convenience
        self.node = N      # node index, debugging convenience
        self.weights = [random.uniform(-0.2, 0.2) for _ in range(weights)]
        self.bias = bias
        self.activation_fx = activation_fx

    def activate(self, inputs:list) -> float:
        '''Weight each input, add the bias, then apply the node's activation function.'''
        z = sum(weight * input for weight, input in zip(self.weights, inputs))
        z += self.bias
        return self.activation_fx(z)


class Layer:
    def __init__(self, L_index, weights:int, nodes:int, activation_fx:Callable):
        '''Construct a layer of `nodes` nodes, each with `weights` weights and the given activation function'''
        self.nodes = [Node(L_index, N_index, weights, activation_fx) for N_index in range(nodes)]

    def activate(self, input:list) -> List[float]:
        '''Activate each node in the layer, returning a list of activation values'''
        return [node.activate(input) for node in self.nodes]


class Net:
    def __init__(self, inputs:int, shape:List[int], activation_fxs:List[Callable], loss_fx:Callable):
        '''Construct a neural network with `inputs` input nodes and `loss_fx` as the loss function for training,
        where layer `n` has `shape[n]` nodes and `activation_fxs[n]` activation function.'''
        assert len(shape) == len(activation_fxs)
        shape.insert(0, inputs)
        self.layers = [Layer(i, shape[i], shape[i+1], activation_fxs[i]) for i in range(len(shape)-1)]
        self.loss_fx = loss_fx

    def show_params(self):
        '''Print the network's parameters (weights and biases of each node)'''
        for i,l in enumerate(self.layers):
            for j,n in enumerate(l.nodes):
                print(f'layer {i+1}, node {j+1}, {n.weights=} {n.bias=}')

    def forward(self, x:List[float]) -> List[List[float]]:
        '''Given input `x`, compute activations for each node in the network by iterating through layers'''
        return [x := layer.activate(x) for layer in self.layers]

    class Gradients(TypedDict):
        weight: List[List[List[float]]]
        bias: List[List[float]]
    def backward(self, x:list, y:float, activations:List[list]) -> Gradients:
        '''Compute partial derivatives ("gradients") of the loss w.r.t. every weight and bias, given the input `x`,
        the correct output `y`, and the `activations` from a forward pass. (The gradients indicate how a change to a
        weight or bias--while holding the other parameters fixed--would change the loss, and are what we'll use to train)'''
        # the prediction is just the final layer's output
        ŷ = activations[-1][0]
        # we don't need the actual loss, only the derivative of the loss
        dLdŷ = loss_derivative[self.loss_fx](y, ŷ)

        # create weight and bias gradient lists so we can index on them
        weight_grads = [[] for _ in range(len(self.layers))]
        bias_grads = [[] for _ in range(len(self.layers))]
        # initialize node gradient values to zero so we can accumulate on them
        node_grads = [[0] * len(self.layers[i].nodes) for i in range(len(self.layers))]
        # base case: derivative of the loss relative to our prediction
        node_grads[-1] = [dLdŷ]

        # iterate backwards through the layers
        for i in range(len(self.layers)-1, -1, -1):
            # for each node, its activation from the forward pass, and its gradient from the backward pass...
            for n, a, dLda in zip(self.layers[i].nodes, activations[i], node_grads[i]):
                # compute da/dz (partial derivative of the node's activation w.r.t. the pre-activation value)
                dadz = derivative_from_a[n.activation_fx](a)
                # compute dL/dz using the chain rule
                dLdz = dadz * dLda

                # Now distribute dLdz backwards to each component of z: bias, weights, and prior layer activations

                ### bias gradient ###
                # Since b is directly added into z, dz/db = 1, so by the chain rule, dL/db = 1 * dL/dz = dL/dz:
                bias_grads[i].append(dLdz)

                ### weight gradients ###
                # Since the weights scale an incoming signal (either an activation from the prior layer or an input),
                # the derivative of z w.r.t. the weight is just the value of that signal: dz/dw[j] = (a[i-1][j] or x[j])
                # Then, by the chain rule: dL/dw[j] = dz/dw[j] * dL/dz
                in_values = activations[i-1] if i != 0 else x
                weight_grads[i].append([dzdw * dLdz for dzdw in in_values])

                ### prior layer activation gradients ###
                # Likewise, the derivative of z w.r.t. the prior activation is just the weight: dz/da[i-1][j] = w[j]
                # However, because each node's activation can affect the loss via multiple downstream nodes, we must
                # accumulate all of these local effects to get the derivative of the loss: dL/da[i] += dz/da * dL/dz
                if i == 0:
                    # don't calculate gradients for the input (x); skip to the next node
                    continue
                for j in range(len(n.weights)):
                    dzda = n.weights[j]
                    dLda_increment = dzda * dLdz
                    node_grads[i-1][j] += dLda_increment
                    # print(f'added {dLda_increment} to node grad[{i-1}][{j}]')

        return {'weight': weight_grads, 'bias': bias_grads}

    # Last but not least, our training function. When the training data is small, we can do full batch training
    def batch_train(self, training_data:List[dict], epochs:int, learning_rate:float=0.1) -> None:
        '''Train network using batch gradient descent with `training_data` for `epochs` batches'''
        def checkpoint(epoch):
            return True if epoch == 0 or (epoch+1) % 1_000 == 0 else False

        batch_loss = []
        for epoch in range(epochs):
            batch = []
            # (1) collect gradients for each case in the training data using a forward and backward pass for each
            for training_case in random.sample(training_data, len(training_data)):
                x, y = training_case.values()
                activations = self.forward(x)
                gradients = self.backward(x, y, activations)

                batch.append(gradients)
                if checkpoint(epoch):
                    ŷ = activations[-1][0]
                    batch_loss.append(self.loss_fx(y, ŷ))

            if checkpoint(epoch):
                print(f'epoch {epoch+1:{len(str(epochs))}d} loss={sum(batch_loss)/len(batch)}')
                batch_loss = []

            # (2) compute the average gradient for each weight and bias across all training cases in the batch
            average_bias_grad = [[] for _ in range(len(self.layers))]
            average_weight_grads = [[] for _ in range(len(self.layers))]
            for layer in range(len(self.layers)):
                transposed_bias_grads = list(zip(*[batch[b]['bias'][layer] for b in range(len(batch))]))
                average_bias_grad[layer] = [sum(grads)/len(batch) for grads in transposed_bias_grads]

                average_weight_grads[layer] = [[] for _ in range(len(self.layers[layer].nodes))]
                for node in range(len(self.layers[layer].nodes)):
                    transposed_weight_grads = list(zip(*[batch[b]['weight'][layer][node] for b in range(len(batch))]))
                    average_weight_grads[layer][node] = [sum(grads)/len(batch) for grads in transposed_weight_grads]

            # (3) nudge the weights in the direction opposite the gradient (which points towards the steepest ascent of loss)
            for layer, layer_b_grads, layer_w_grads in zip(self.layers, average_bias_grad, average_weight_grads):
                for node, b_grad, w_grads in zip(layer.nodes, layer_b_grads, layer_w_grads):
                    node.bias += -b_grad * learning_rate
                    for i in range(len(node.weights)):
                        node.weights[i] += -w_grads[i] * learning_rate



#########################################################
# Activation functions, loss functions, and derivatives #
#########################################################

# Activation and loss functions
def ReLU(z:float):
    '''rectified linear unit function, aka squash at zero'''
    return max(0, z)

def sigmoid(z:float):
    '''sigmoid function for binary classification'''
    exp_z = math.exp(z)          # benchmark: 10-14% faster to store exp(z) rather than calculate twice
    return exp_z / (1 + exp_z)   # alternate form for numerical stability; 1/(1+math.exp(-z)) will overflow

def log_loss(y:float, y_predicted:float):
    '''cross-entropy loss for binary classification; penalizes confident but incorrect predictions more heavily'''
    y_predicted = max(1e-15, min(1-1e-15, y_predicted))                   # clip ŷ to [1e-15, 1-1e15] to avoid log(0)
    return -( (y*math.log(y_predicted)) + (1-y)*math.log(1-y_predicted) ) # simplifies when y ∈ {0,1}

# Derivatives of our activation and loss functions, for backpropagation:
def ReLU_derivative(v):
    return 1 if v > 0 else 0

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1-sig)

def sigmoid_derivative_from_a(a):
    return a * (1-a)

def log_loss_derivative(y_actual:float, y_predicted:float):
    y_predicted = max(1e-15, min(1-1e-15, y_predicted))           # clip ŷ to [1e-15, 1-1e15] to avoid dividing by 0
    return -(y_actual/y_predicted)+((1-y_actual)/(1-y_predicted))

# mappings
derivative_from_a = {
    ReLU: ReLU_derivative,
    sigmoid: sigmoid_derivative_from_a,
}
loss_derivative = {
    log_loss: log_loss_derivative,
}
