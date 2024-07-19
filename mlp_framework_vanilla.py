# Feedforward neural network in vanilla python
#
# The Node, Layer, and Net classes constitute a complete multilayer perceptron
# neural network framework with forward inference (`Net.forward`), backward
# propagation (`Net.backward`), and gradient descent training (`Net.train`)
#
# This framework is written without numpy/pytorch/tensorflow/scikit-learn for
# educational purposes. It is wildly inefficient.

import math
import random
from typing import Callable, TypedDict

Vector = list[float]
Activations = list[Vector]
class Gradients(TypedDict):
    weight: list[list[Vector]]
    bias: list[Vector]
class Sample(TypedDict):
    x: list[float]
    y: list[float]
TrainingData = list[Sample]

class Node:
    def __init__(self, num_inputs:int, activation_fx:Callable, bias:float = 0.):
        '''Initialize a new Node, with random small-but-non-negligible weights for each input'''
        self.weights = [random.uniform(0.2, 0.5)*random.choice((1, -1)) for _ in range(num_inputs)]
        self.bias = bias
        self.activation_fx = activation_fx

    def activate(self, inputs:Vector) -> float:
        '''Weight each input, add the bias, then apply the node's activation function'''
        z = sum(weight * input for weight, input in zip(self.weights, inputs))
        z += self.bias
        return self.activation_fx(z)


class Layer:
    def __init__(self, layer_num, num_inputs:int, num_nodes:int, activation_fx:Callable):
        '''Construct a layer of `num_nodes` Nodes, each with `num_inputs` weights'''
        self.layer_num = layer_num
        self.nodes = [Node(num_inputs, activation_fx) for _ in range(num_nodes)]

    def activate(self, input:Vector) -> Vector:
        '''Activate each node in the layer, returning a list of activation values'''
        return [node.activate(input) for node in self.nodes]


class Net:
    def __init__(self, inputs:int, shape:list[int], activation_fxs:list[Callable], loss_fx:Callable):
        '''Construct a neural network with `inputs` input nodes and `loss_fx` as the loss function for
        training, where layer `n` has `shape[n]` nodes and `activation_fxs[n]` activation function.'''
        assert len(shape) == len(activation_fxs)
        shape.insert(0, inputs)
        self.layers = [Layer(n, shape[n], shape[n+1], activation_fxs[n]) for n in range(len(shape)-1)]
        self.loss_fx = loss_fx

    def show_params(self):
        '''Print the network's parameters (weights and biases of each node)'''
        for i,layer in enumerate(self.layers, 1):
            for j,node in enumerate(layer.nodes, 1):
                print(f'layer {i}, node {j}, {node.weights=} {node.bias=}')

    def forward(self, x:Vector) -> Activations:
        '''Given input `x`, compute activations for each node in the network by iterating through layers'''
        return [x := layer.activate(x) for layer in self.layers]

    def backward(self, x:Vector, y:Vector, activations:Activations) -> Gradients:
        '''Compute partial derivatives ("gradients") of the loss w.r.t. every weight and bias in the Net, given
        input `x`, correct output `y`, and `activations` from a forward pass. (The gradients indicate how changing
        a weight or bias--holding the other parameters fixed--changes the loss, and are needed for training.)'''
        # ŷ, the prediction, is just the final layer's output
        ŷ = activations[-1]
        # we start by calculating dL/dŷ, the derivative of the loss with regard to the prediction:
        dLdŷ = loss_derivative[self.loss_fx](y, ŷ)

        # create weight and bias gradient lists so we can index on them
        weight_grads = [[] for _ in range(len(self.layers))]
        bias_grads = [[] for _ in range(len(self.layers))]
        # initialize node gradient values to zero so we can accumulate on them
        node_grads = [[0] * len(self.layers[i].nodes) for i in range(len(self.layers))]
        # base case: derivative of the loss relative to our prediction
        node_grads[-1] = dLdŷ

        # iterate backwards through the layers
        for layer in reversed(self.layers):
            layer_num = layer.layer_num

            # for each node, its activation from the forward pass, and its gradient from the backward pass...
            for n, a, dLda in zip(layer.nodes, activations[layer_num], node_grads[layer_num]):
                # compute da/dz (partial derivative of the node's activation w.r.t. the pre-activation value)
                dadz = derivative_from_a[n.activation_fx](a)
                # compute dL/dz using the chain rule
                dLdz = dadz * dLda

                # Now distribute dLdz backwards to each component of z: bias, weights, and prior layer activations

                ### bias gradient ###
                # Since b is directly added into z, dz/db = 1, so by the chain rule, dL/db = 1 * dL/dz = dL/dz:
                dLdb = dLdz
                bias_grads[layer_num].append(dLdb)

                ### weight gradients ###
                # Since the weights scale an incoming signal (either an activation from the prior layer or an input),
                # dz/dw (z's rate of change when the weight changes) is just that signal value:
                #   dz/dw[i] = x[i]      for the first layer, and
                #            = a[L-1][i] for layer L > 1
                # Then, by the chain rule: dL/dw[i] = dz/dw[i] * dL/dz
                in_signal = activations[layer_num-1] if layer_num != 0 else x
                dLdw = [dzdw * dLdz for dzdw in in_signal]
                weight_grads[layer_num].append(dLdw)

                ### prior layer activation gradients ###
                if layer_num == 0: # we don't need gradients for the input; skip to the next node
                    continue
                # Likewise, the derivative of z w.r.t. the prior activation is just the weight: dz/da[L-1][i] = w[i]
                # However, because each node's activation can affect the loss via multiple downstream paths, we must
                # accumulate each individual effect to get the total derivative of the loss: dL/da[i] += dz/da * dL/dz
                for i, weight in enumerate(n.weights):
                    dzda = weight
                    # the effect of this node's activation on the loss *through just this specific weight*:
                    dLda_increment = dzda * dLdz
                    # add the local effect to the overall effect:
                    node_grads[layer_num-1][i] += dLda_increment

        return {'weight': weight_grads, 'bias': bias_grads}

    def train(self,
              training_data:TrainingData,
              epochs:int,
              batch_size:int=32,
              learning_rate:float=0.1,
              batch_progress_every=20,      # report average loss every N batches for monitoring during training
              epoch_progress_every=1        # report average loss over the entire epoch every N epochs
              ) -> list[float]:             # average epoch losses
        '''Train network with `training_data` using gradient descent, for `epochs` epochs`. By default, this will
        perform minibatch gradient descent with a `batch_size` of 32. For batch gradient descent, set `batch_size`
        to len(training_data) and for stochastic gradient decent, set `batch_size` to 1. Return a list of the
        average loss of each epoch'''
        expected_batches = math.ceil(len(training_data)/batch_size)
        epoch_losses = []

        for epoch in range(1, epochs+1):
            print_this_epoch = (epoch == 1 or epoch == epochs or (epoch % epoch_progress_every == 0))

            epoch_loss = 0
            random.shuffle(training_data)

            # gather minibatches of training data and pass them to gradient_descent()
            for batch_num, batch_index in enumerate(range(0, len(training_data), batch_size), 1):
                minibatch = training_data[batch_index:(batch_index + batch_size)]
                batch_loss = self.gradient_descent(minibatch, learning_rate)

                epoch_loss += batch_loss
                if batch_num % batch_progress_every == 0:
                    print(f'{epoch=} {batch_num=}/{expected_batches}, average loss {batch_loss/len(minibatch):.5f}')

            epoch_average_loss = epoch_loss/len(training_data)
            epoch_losses.append(epoch_average_loss)
            if print_this_epoch:
                print(f'epoch {epoch:{len(str(epochs))}d} complete, ' \
                      f'{batch_num} batch{"es" if batch_num != 1 else ""}, ' \
                      f'average loss {epoch_average_loss:.5f}')

        return epoch_losses

    def gradient_descent(self, batch:TrainingData, learning_rate:float) -> float:
            '''Perform gradient descent on the given `batch`: compute the gradients, average them, and update the
            network weights scaled by `learning_rate`. Return the total accumulated loss for the batch.'''
            batch_size = len(batch) # NB: may be smaller than Net.train()'s batch_size (final minibatch of an epoch)

            # (1) for each training sample, collect gradients of the loss by running a forward and backward pass
            batch_grads = []
            batch_loss = 0
            for sample in batch:
                x, y = sample.values()

                activations = self.forward(x)
                gradients = self.backward(x, y, activations)
                batch_grads.append(gradients)

                ŷ = activations[-1]
                batch_loss += self.loss_fx(y, ŷ)

            # (2) compute the average gradient for each weight and bias across all runs in the batch
            average_bias_grads = []
            average_weight_grads = []
            for layer_i, layer in enumerate(self.layers):
                # gather the layer's bias gradients from each run (shape=num_runs, num_nodes)
                layer_bias_grads = [batch_grads[run_i]['bias'][layer_i] for run_i in range(batch_size)]
                # transpose the matrix so the gradients are grouped by node (shape=num_nodes, num_runs)
                transposed_bias_grads = list(zip(*layer_bias_grads))
                # average each node's gradients and store the result
                average_bias_grads.append([sum(grads)/batch_size for grads in transposed_bias_grads])

                layer_weight_grads = []
                for node_i, node in enumerate(layer.nodes):
                    # gather the node's weight gradients from each run (shape=num_runs, num_weights)
                    node_weight_grads = [batch_grads[run_i]['weight'][layer_i][node_i] for run_i in range(batch_size)]
                    # transpose the matrix so the gradients are grouped by weight (shape=num_weights, num_runs)
                    transposed_weight_grads = list(zip(*node_weight_grads))
                    # average each weight's gradients and store the result
                    layer_weight_grads.append([sum(grads)/batch_size for grads in transposed_weight_grads])
                average_weight_grads.append(layer_weight_grads)

            # (3) nudge each weight and bias against its gradient (which points towards the steepest ascent of loss)
            for layer, layer_b_grads, layer_w_grads in zip(self.layers, average_bias_grads, average_weight_grads):
                for node, b_grad, w_grads in zip(layer.nodes, layer_b_grads, layer_w_grads):
                    node.bias += -b_grad * learning_rate
                    for i, w_grad in enumerate(w_grads):
                        node.weights[i] += -w_grad * learning_rate

            return batch_loss


#########################################################
# Activation functions, loss functions, and derivatives #
#########################################################

# Activation functions
def ReLU(z:float):
    '''rectified linear unit function, aka squash at zero'''
    return max(0, z)

def leaky_ReLU(z, alpha=0.01):
    return z if z >= 0 else z*alpha

def sigmoid(z:float) -> float:
    '''sigmoid function for binary classification'''
    exp_z = math.exp(z)          # benchmark: 10-14% faster to store exp(z) rather than calculate twice
    return exp_z / (1 + exp_z)   # alternate form for numerical stability; 1/(1+math.exp(-z)) will overflow

# Derivatives of our activation functions, for backpropagation:
def sigmoid_derivative(z:float) -> float:
    '''calculate the derivative of sigmoid given pre-activation value z'''
    sig = sigmoid(z)
    return sig * (1-sig)

def sigmoid_derivative_from_a(a:float) -> float:
    '''during backprop we already have sigmoid(z) = the node's activation a, so calculate the derivative directly'''
    return a * (1-a)

def ReLU_derivative(v:float) -> float:
    '''the formula for the derivative of ReLU is identical whether calculated from z or from a'''
    return 1. if v > 0 else 0.

def leaky_ReLU_derivative(v:float, alpha=0.01) -> float:
    return 1. if v > 0 else alpha

# Loss functions and their derivatives
def log_loss(y_actual:Vector, y_predicted:Vector) -> float:
    '''cross-entropy loss for binary classification; penalizes confident but incorrect predictions more heavily'''
    loss = 0
    for y, ŷ in zip(y_actual, y_predicted):                         # accumulate the loss for each output node
        ŷ = max(1e-15, min(1-1e-15, ŷ))                             # clip ŷ to [1e-15, 1-1e-15] to avoid log(0)
        loss += -((y * math.log(ŷ)) + ((1 - y) * math.log(1 - ŷ)))  # NB: simplifies when y ∈ {0,1}
    return loss / len(y_actual)                                     # return mean log loss

def log_loss_derivative(y_actual:Vector, y_predicted:Vector) -> Vector:
    '''Compute the derivative of the log loss for binary classification.'''
    derivatives = []
    for y, ŷ in zip(y_actual, y_predicted):
        ŷ = max(1e-15, min(1-1e-15, ŷ))
        derivative = -(y / ŷ) + ((1 - y) / (1 - ŷ))
        derivatives.append(derivative)
    return derivatives

# mappings
derivative_from_a = {
    sigmoid: sigmoid_derivative_from_a,
    ReLU: ReLU_derivative,
    leaky_ReLU: leaky_ReLU_derivative,
}
loss_derivative = {
    log_loss: log_loss_derivative,
}

# Hat tips for their explanations of backpropagation:
# * Rumelhart, Hinton, and Williams, "Learning representations by back-propagating errors" _Nature_ (1986)
# * Michael A. Nielsen, "Neural networks and deep learning" http://neuralnetworksanddeeplearning.com/ (2015)