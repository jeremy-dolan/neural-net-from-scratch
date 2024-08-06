# MLP framework in vanilla python
#
# The Node, Layer, and Net classes constitute a complete multilayer perceptron
# neural network framework with forward inference (`Net.forward`) and gradient
# descent training (`Net.train`) by backpropagation of error (`Net.backward`).
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
LabeledDataset = list[Sample]
class TrainingResults(TypedDict):
    training_loss: list[float]
    test_loss: list[float|None]
    test_accuracy: list[float|None]

class Node:
    def __init__(self, num_inputs:int, bias:float = 0.):
        '''Initialize a new Node, with random small-but-non-negligible weights for each input'''
        self.weights = [random.uniform(0.2, 0.5)*random.choice((1, -1)) for _ in range(num_inputs)]
        # FIXME parameterize and improve weight initialization
        self.bias = bias

    def compute(self, inputs:Vector) -> float:
        '''Sum the bias and each weighted input for the Node'''
        return self.bias + sum(weight * input for weight, input in zip(self.weights, inputs))

class Layer:
    def __init__(self, num_inputs:int, num_nodes:int, activation_fx:Callable):
        '''Construct a layer of `num_nodes` Nodes, each with `num_inputs` weights'''
        self.activation_fx = activation_fx
        self.nodes = [Node(num_inputs) for _ in range(num_nodes)]

    def activate(self, input:Vector) -> Vector:
        '''Activate each node in the layer, returning a list of activation values'''
        pre_activation_values = [node.compute(input) for node in self.nodes]
        return self.activation_fx(pre_activation_values)

class Net:
    def __init__(self, inputs:int, shape:list[int], activation_fxs:list[Callable], loss_fx:Callable):
        '''Construct a neural network with `inputs` input nodes and `loss_fx` as the loss function for
        training, where layer `n` has `shape[n]` nodes and `activation_fxs[n]` activation function.'''
        shape.insert(0, inputs)
        self.layers = [Layer(shape[n], shape[n+1], activation_fxs[n]) for n in range(len(shape)-1)]
        self.loss_fx = loss_fx
        self.sanity_checks()

    def forward(self, x:Vector) -> Activations:
        '''Given input `x`, compute activations for each node in the network by iterating through layers'''
        return [x := layer.activate(x) for layer in self.layers]

    def backward(self, x:Vector, y:Vector, activations:Activations) -> Gradients:
        '''Compute partial derivatives ("gradients") of the loss w.r.t. every weight and bias in the Net, given
        input `x`, correct output `y`, and `activations` from a forward pass. (The gradients indicate how changing
        a weight or bias--holding the other parameters fixed--changes the loss, and are needed for training.)'''
        # ŷ, the prediction, is just the final layer's output
        ŷ = activations[-1]
        # we start by calculating dL/dŷ, the derivative of the loss with respect to the prediction:
        dLdŷ = loss_derivative[self.loss_fx](y, ŷ)

        # create weight and bias gradient lists so we can index on them
        weight_grads = [[] for _ in range(len(self.layers))]
        bias_grads = [[] for _ in range(len(self.layers))]
        # initialize node gradient values to zero so we can accumulate on them
        node_grads = [[0] * len(self.layers[i].nodes) for i in range(len(self.layers))]
        # base case: derivative of the loss relative to our prediction
        node_grads[-1] = dLdŷ

        # iterate backwards through the layers
        for enum, layer in enumerate(reversed(self.layers)):
            layer_num = len(self.layers)-1-enum

            # Compute da/dz (the partial derivative of a node's activation w.r.t. its pre-activation value) for
            # all nodes in the layer. We do this first, for the whole layer, because some activation functions
            # (notably, softmax) are dependent on every pre-activation value in that layer.
            # Note also that we are calculating these derivatives directly from the activation value `a`.
            layer_dadz = derivative_from_a[layer.activation_fx](activations[layer_num])

            # for each node, the partial derivative of its activation, and its loss gradient...
            for n, dadz, dLda in zip(layer.nodes, layer_dadz, node_grads[layer_num]):
                # compute dL/dz (derivative of the loss w.r.t. the pre-activation value) using the chain rule
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
              training_data:LabeledDataset,
              epochs:int=1,
              batch_size:int=32,
              learning_rate:float=0.1,
              batch_progress_every=100,     # report average loss every N batches for monitoring during training
              epoch_progress_every=1,       # report average loss over the entire epoch every N epochs
              test_every=0,                 # calculate test loss and test accuracy every N epochs
              test_data:LabeledDataset=None
              ) -> TrainingResults:
        '''Train network with `training_data` using gradient descent, for `epochs`. By default, this will perform
        minibatch gradient descent with a `batch_size` of 32. For batch gradient descent, set `batch_size` to
        len(training_data) and for stochastic gradient decent, set `batch_size` to 1. Return TrainingResults
        containing training loss from each epoch, and optionally test loss and test accuracy. NB: This is essentially
        a wrapper for .gradient_descent() that checks the data, batches it up, and optionally reports progress.'''
        expected_batches = math.ceil(len(training_data)/batch_size)
        training_losses = []
        test_losses = []
        test_accuracies = []

        for epoch in range(1, epochs+1):
            test_this_epoch = (test_every > 0 and test_data is not None and epoch % test_every == 0)
            print_this_epoch = (epoch_progress_every > 0 and epoch % epoch_progress_every == 0)

            cumulative_epoch_loss = 0

            # FIXME refactor to shuffle indices instead of the data itself
            random.shuffle(training_data)

            # gather minibatches of training data and pass them to gradient_descent()
            for batch_num, batch_index in enumerate(range(0, len(training_data), batch_size), 1):
                minibatch = training_data[batch_index:(batch_index + batch_size)]
                batch_loss = self.gradient_descent(minibatch, learning_rate)

                cumulative_epoch_loss += batch_loss
                if batch_progress_every > 0 and batch_num % batch_progress_every == 0:
                    print(f'{epoch=} {batch_num=}/{expected_batches}, ' \
                          f'average loss for batch: {batch_loss/len(minibatch):.5f}')

            training_loss = cumulative_epoch_loss/len(training_data)
            training_losses.append(training_loss)

            if test_this_epoch:
                test_loss, num_correct, num_samples = self.test(test_data)
                test_losses.append(test_loss)
                test_accuracies.append([num_correct, num_samples])

            if print_this_epoch:
                print(f'epoch {epoch:{len(str(epochs))}d} complete' \
                      f', {batch_num} batch{"es" if batch_num != 1 else ""}' \
                      f', {training_loss=:.5f}', end='')
                print(f', {test_loss=:.5f}, accuracy: {num_correct}/{num_samples}' if test_this_epoch else '')

        return {
            'training_loss': training_losses,
            'test_loss': test_losses,
            'test_accuracy': test_accuracies
        }

    def gradient_descent(self, batch:LabeledDataset, learning_rate:float) -> float:
            '''Perform gradient descent on the given `batch`: compute the gradients, average them, and update the
            network weights scaled by `learning_rate`. Return the total accumulated loss for the batch.'''
            # batch < Net.train()'s batch_size if len(training_data) < batch_size or for final minibatch of an epoch
            batch_size = len(batch)

            # (1) for each training sample, collect gradients of the loss by running a forward and backward pass
            batch_grads = []
            batch_loss = 0
            for sample in batch:
                x, y = sample['x'], sample['y']

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

    # FIXME assumes single-label classification (either binary or multi-class)
    # refactor for multi-label classification, regression (continuous outputs)...
    def test(self, test_data:LabeledDataset, threshold=0.5) -> tuple[float, int, int]:
        '''WIP'''
        cumulative_loss = 0.
        correct_predictions = 0
        num_samples = len(test_data)

        for sample in test_data:
            x, y = sample['x'], sample['y']
            ŷ = self.forward(x)[-1]

            cumulative_loss += self.loss_fx(y, ŷ)

            if len(y) == 1:         # assume binary classification
                true_class = y[0]
                predicted_class = 0 if ŷ < threshold else 1
            else:                   # assume one-hot vector vs. a probability distribution
                true_class = y.index(1)
                predicted_class = ŷ.index(max(ŷ))

            if true_class == predicted_class:
                correct_predictions += 1

        return cumulative_loss/num_samples, correct_predictions, num_samples

    def show_params(self):
        '''Print the network's parameters (weights and biases of each node)'''
        for i,layer in enumerate(self.layers, 1):
            for j,node in enumerate(layer.nodes, 1):
                print(f'layer {i}, node {j}, {node.weights=} {node.bias=}')

    def sanity_checks(self):
        for layer_i, layer in enumerate(self.layers,  1):
            # ensure Net instantiation has an activation function for each layer
            if not hasattr(layer, 'activation_fx'):
                raise ValueError(f'Layer {layer_i} has no activation function assigned')
            # ensure softmax() only used on final layer; derivative otherwise unimplemented
            if layer.activation_fx == softmax and layer_i != len(self.layers):
                raise ValueError(f'softmax activation is only supported for the final layer')
        # softmax() and categorical_cross_entropy() have their gradient calculated together, so they must be used
        # together or not at all. See docstring in `coupled_softmax_and_categorical_cross_entropy_gradient()`
        if (self.layers[-1].activation_fx == softmax) != (self.loss_fx == categorical_cross_entropy):
            raise ValueError('softmax and categorical_cross_entropy only supported when paired together')


#########################################################
# Activation functions, loss functions, and derivatives #
#########################################################

# Activation functions
def ReLU(V:Vector) -> Vector:
    '''rectified linear unit function, aka squash at zero'''
    return [max(0, z) for z in V]

def leaky_ReLU(V:Vector, alpha=0.01) -> Vector:
    '''avoid dying ReLUs (but also sparse activation) with a small slope for negative inputs'''
    return [z if z >= 0 else z*alpha for z in V] # slower: max(alpha*z, z)

def sigmoid(V:Vector) -> Vector:
    '''numerically stable logistic function; map each input to range (0, 1); useful for binary classification'''
    return [1 / (1 + math.exp(-z)) if z >= 0 else (exp := math.exp(z)) / (1 + exp) for z in V]

def softmax(V:Vector) -> Vector:
    '''normalize logits to a probability distribution; useful for multi-class classification'''
    max_V = max(V)
    numerators = [math.exp(z - max_V) for z in V] # exponentiate each logit; subtract max_V for numerical stability
    denominator = sum(numerators)
    return [numerator/denominator for numerator in numerators]

# Derivatives of our activation functions, for backpropagation:
def sigmoid_derivative(V:Vector) -> Vector:
    '''calculate the derivative of sigmoid given pre-activation values z'''
    return [(sig := sigmoid([z])[0]) * (1-sig) for z in V]

def sigmoid_derivative_from_a(V:Vector) -> Vector:
    '''during backprop we already have sigmoid(z) = the node's activation a, so calculate the derivative directly'''
    return [a * (1-a) for a in V]

def ReLU_derivative(V:Vector) -> Vector:
    '''the formula for the derivative of ReLU is identical whether calculated from z or from a'''
    return [1. if v > 0 else 0. for v in V]

def leaky_ReLU_derivative(V:Vector, alpha=0.01) -> Vector:
    return [1. if v > 0 else alpha for v in V]

def softmax_fake_derivative(V:Vector) -> Vector:
    '''coupled_softmax_and_categorical_cross_entropy_gradient() computes the combined gradient of softmax and
    cross-entropy, so here we just fabricate a vector of ones that yield multiplicative identity by the chain rule'''
    return [1] * len(V)

# Loss functions and their derivatives
# Note that loss functions themselves are only used for monitoring and evaluation of the training process;
# it's the derivative of the loss function that is directly used in training (by backpropagation).
def mse(y_actual:Vector, y_predicted:Vector) -> float:
    '''Quadratic loss for Vectors `y_actual` and `y_predicted`, useful for regression'''
    error = [y - ŷ for y, ŷ in zip(y_actual, y_predicted)]
    squared_error = [v**2 for v in error]
    mean_squared_error = sum(squared_error) / len(y_actual)
    return mean_squared_error
    
def mse_derivative(y_actual:Vector, y_predicted:Vector) -> Vector:
    '''Quick hack, make implementation intuitive. dMSE/dy = 2/n * (ŷ-y)'''
    scaling_factor = 2 / len(y_actual)
    return [scaling_factor * (ŷ - y) for y, ŷ in zip(y_actual, y_predicted)]

def binary_cross_entropy(y_actual:Vector, y_predicted:Vector) -> float:
    '''Log loss for independent binary classifications (including multi-label classification);
    log loss penalizes confident but incorrect predictions more heavily.'''
    loss = 0
    for y, ŷ in zip(y_actual, y_predicted):                         # accumulate the loss for each output node
        ŷ = min(max(ŷ, 1e-15), 1 - 1e-15)                           # clip ŷ to [1e-15, 1-1e-15] to avoid log(0)
        loss += -((y * math.log(ŷ)) + ((1 - y) * math.log(1 - ŷ)))  # NB: simplifies piecewise when y ∈ {0,1}
    return loss / len(y_actual)                                     # normalize for training stability

def binary_cross_entropy_derivative(y_actual:Vector, y_predicted:Vector) -> Vector:
    '''Given true output y_actual, return gradient of the loss with respect to each prediction ŷ ∈ y_predicted'''
    derivatives = []
    for y, ŷ in zip(y_actual, y_predicted):          # each prediction is independent
        ŷ = min(max(ŷ, 1e-15), 1 - 1e-15)            # clip ŷ to [1e-15, 1-1e-15] to avoid dividing by 0
        derivative = -(y / ŷ) + ((1 - y) / (1 - ŷ))  # NB: simplifies piecewise when y ∈ {0,1}
        derivatives.append(derivative)
    return derivatives

def categorical_cross_entropy(y_actual:Vector, y_predicted:Vector) -> float:
    '''Multi-class classification version of log loss, for use with one-hot encoded vectors.'''
    # We only calculate the negative logarithm of the predicted probability for the single true class. But because
    # the probability given by softmax depends on every logit, all of the outputs still indirectly affect the loss
    one_hot_y_index = y_actual.index(1)
    corresponding_ŷ = y_predicted[one_hot_y_index]
    clipped_ŷ = min(max(corresponding_ŷ, 1e-15), 1 - 1e-15) # avoid log 0 (undef) / log 1 (zero gradient)
    return -math.log(clipped_ŷ)

def categorical_cross_entropy_derivative(y_actual:Vector, y_predicted:Vector) -> Vector:
    '''Calculate loss gradient for the output node corresponding with the true (one hot) class. Only that node
    directly contributes to the loss with cross entropy (the error will then spread backwards through softmax)'''
    derivatives = [0] * len(y_actual) # initialize vector
    one_hot_index = y_actual.index(1)
    one_hot_prediction = y_predicted[one_hot_index]
    clipped_prediction = max(one_hot_prediction, 1e-15) # avoid division by 0
    derivatives[one_hot_index] = -1 / clipped_prediction
    return derivatives

def coupled_softmax_and_categorical_cross_entropy_gradient(y_actual:Vector, y_predicted:Vector) -> Vector:
    '''Calculate the combined gradient of softmax and cross entropy in the output layer. Calculating them together
    here and then faking it for softmax (`softmax_fake_derivative()`) is a hack to maintain the simple, linear flow
    of Net.backward(). Taken independently, softmax's derivative is a Jacobian. To calculate its contribution to the
    gradient of the categorical cross-entropy loss, we need to know the one-hot category used for the loss,
    which isn't accessible to the activation derivative functions in this framework, as it stands.

    Without introducing a significant new layer of abstraction to what is supposed to be a straightforward framework,
    there does not seem to be an elegant solution. This hack, at least, has the virtue of computational efficiency:
    softmax and cross entropy are tightly coupled, and their combined gradient simplifies to:'''
    return [ŷ - y for y, ŷ in zip(y_actual, y_predicted)]

# function/derivative mappings
derivative_from_a = {
    sigmoid: sigmoid_derivative_from_a,
    ReLU: ReLU_derivative,
    leaky_ReLU: leaky_ReLU_derivative,
    softmax: softmax_fake_derivative,
}
loss_derivative = {
    mse: mse_derivative,
    binary_cross_entropy: binary_cross_entropy_derivative,
    categorical_cross_entropy: coupled_softmax_and_categorical_cross_entropy_gradient,
}

# Hat tips for their explanations of backpropagation:
# * Rumelhart, Hinton, and Williams, "Learning representations by back-propagating errors" _Nature_ (1986)
# * Michael A. Nielsen, "Neural networks and deep learning" http://neuralnetworksanddeeplearning.com/ (2015)