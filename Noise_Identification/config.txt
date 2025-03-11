input_dim = 3
hidden_dim = 128
latent_dim = 64
learning_rate = 0.001
num_classes = 9

label_map = {
    "clean": 0,
    "gaussian_low": 1, "gaussian_high": 2,
    "uniform_low": 3, "uniform_high": 4,
    "background_low": 5, "background_high": 6,
    "impulse_low": 7, "impulse_high": 8
}

noise_levels = [
    'uniform_1', 'uniform_2', 'uniform_3', 'uniform_4', 'uniform_5',
    'gaussian_1', 'gaussian_2', 'gaussian_3', 'gaussian_4', 'gaussian_5',
    'background_1', 'background_2', 'background_3', 'background_4', 'background_5', 
    'impulse_1', 'impulse_2', 'impulse_3', 'impulse_4', 'impulse_5'
]