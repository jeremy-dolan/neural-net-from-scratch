import mlp_framework_vanilla as mlp

xor_training = [
    {'x': [0,0], 'y': [0]},
    {'x': [0,1], 'y': [1]},
    {'x': [1,0], 'y': [1]},
    {'x': [1,1], 'y': [0]},
]

bool_training = [
    {'x': [0,0], 'y': [0, 0, 0, 1]},
    {'x': [1,0], 'y': [1, 0, 1, 0]},
    {'x': [0,1], 'y': [1, 0, 1, 0]},
    {'x': [1,1], 'y': [1, 1, 0, 0]},
]

def test_mlp_training():
    xor_net = mlp.Net(inputs=2, shape=[10, 1], activation_fxs=[mlp.sigmoid]*2, loss_fx=mlp.binary_cross_entropy)
    #xor_net.show_params()

    dataset = xor_training

    train_results = xor_net.train(training_data=dataset, epochs=10_000, learning_rate=1.75, epoch_progress_every_n=0, test_every_n=2_000)
    assert train_results['training_loss'][-1] < 0.05

    test_results = xor_net.test(dataset)
    assert len(test_results['incorrect_indices']) == 0

    #print(test_results)
