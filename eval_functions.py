import os

import seaborn as sns
import torch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

from sklearn.manifold import TSNE

from torch.nn import ReLU


GENDER_ENUM = np.vectorize(lambda t: 'male' if t == 0 else 'female')


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    @author: Utku Ozbulak - github.com/utkuozbulak
    source: https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_block = list(self.model.encoder._modules.items())[0][1]
        first_layer = list(first_block._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.encoder._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_signal, target_class):
        # Forward pass
        _, gender_pred = self.model(input_signal)

        # Zero gradients
        self.model.zero_grad()

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, gender_pred.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Backward pass
        gender_pred.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.data.numpy()[0]

        return gradients_as_arr.mean(axis=0)[0]


class SaveFeatures:
    """
    Source: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/filter_visualizer.ipynb
    """
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()


def get_model_solver_paths(save_path, epoch):
    """
    Gets the path to the model and solver of an epoch if specified or to
    the best model and last solver if epoch is None

    args:
        save_path (str): Path to folder where models and solvers are stored
        epoch (int): Epoch at which to load model and solver (use None for
                     best model and last solver)

    returns:
        model_path (str): Path to model
        solver_path (str): Path to solver
    """
    print("Getting model and solver paths")
    model_paths = []
    solver_paths = []

    for _, _, fnames in os.walk(save_path):
        model_paths = [fname for fname in fnames if 'model' in fname]
        solver_paths = [fname for fname in fnames if 'solver' in fname]

    if not model_paths or not solver_paths:
        raise Exception('Model or solver not found.')

    if not epoch:
        model_path = os.path.join(save_path, 'best_model')
        solver_path = os.path.join(save_path, sorted(solver_paths, key=lambda s: int(s.split("solver")[1]))[-1])
    else:
        model_path = os.path.join(save_path, 'model' + str(epoch))
        solver_path = os.path.join(save_path, 'solver' + str(epoch))

    return model_path, solver_path


def show_reconstruction(dataset, model, num_samples, color='black'):
    """
    Creates plots which show the input signal, the reconstructed signal
    and the difference of the two next to each other

    args:
        dataset (torch.utils.data.Dataset): Dataset which contains signals
        model (torch.nn.Module): pytorch autoencoder model
        num_samples (int): Number of samples to plot
        color (str): Color for matplotlib text, axes labels and axes ticks
    """
    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_samples,
    )

    # Get next batch
    x, _ = next(iter(dataloader))
    target = x

    # Compute prediction and diff
    pred, _ = model(x)
    pred = pred.detach()
    diff = target - pred
    ymax = max(target.max(), pred.max())
    ymin = min(target.min(), pred.min())

    if len(x.shape) != 4:
        target = target[:, :, :, None]
        pred = pred[:, :, :, None]
        diff = diff[:, :, :, None]

    for i_channel in range(target.shape[-1]):
        # Create plot
        for i_sample in range(num_samples):
            f, axes = plt.subplots(1, 3, figsize=(20, 5))
            # f.suptitle("Input vs reconstruction, channel: {}".format(i_channel), fontsize=16)

            # Label rows
            labels = {0: 'Ground truth',
                      1: 'Prediction',
                      2: 'Deviation'}

            for i in range(3):
                plt.sca(axes[i])
                axes[i].set_title(labels[i], rotation=0, size=16)
                axes[i].set_ylim([ymin - .5, ymax + .5])
                axes[i].tick_params(labelsize=12)

            # Plot ground truth
            axes[0].plot(target[i_sample, 0, :, i_channel].numpy())

            # Plot prediction
            axes[1].plot(pred[i_sample, 0, :, i_channel].numpy())

            # Plot deviation
            axes[2].plot(diff[i_sample, 0, :, i_channel].numpy())

            plt.show()


def visualize_dataset(dataset, num_samples=10):
    """
    Creates plots which show example signals from the dataset

    args:
        dataset (torch.utils.data.Dataset): Dataset which contains signals
        num_samples (int): Number of samples to plot
    """

    # Get signals
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=num_samples,
    )
    signals, _ = next(iter(dataloader))
    signals = signals[:, 0].numpy()

    # Display signals in plot
    if num_samples == 1 or dataset.do_overfitting:
        plt.title("Datasample to overfit on")
        plt.plot(signals[0])
    else:
        f, axes = plt.subplots(num_samples, figsize=(8, 2 * num_samples))
        f.suptitle("{} Preprocessed data samples".format(num_samples), fontsize=16)

        for i_plot in range(num_samples):
            axes[i_plot].plot(signals[i_plot])

    plt.show(block=True)


def show_solver_history(solver, plot_train=True, plot_val=True, color='black'):
    """
    Creates plots with the training history of a solver.

    args:
        solver (Solver): Solver used for training
        plot_train (bool): Plot the training curves
        plot_val (bool): Plot the validation curves
        color (str): Color for matplotlib text, axes labels and axes ticks
    """
    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    print("Stop reason: %s" % solver.stop_reason)
    print("Stop time: %fs" % solver.training_time_s)

    has_gender_loss = np.array(solver.history['val_gender_loss']).sum() > 0.
    has_rec_loss = np.array(solver.history['val_rec_loss']).sum() > 0.

    train_loss = np.array(solver.history['train_loss'])

    if has_rec_loss:
        train_rec_loss = np.array(solver.history['train_rec_loss'])
        val_rec_loss = np.array(solver.history['val_rec_loss'])
    if has_gender_loss:
        train_gender_loss = np.array(solver.history['train_gender_loss'])
        val_gender_loss = np.array(solver.history['val_gender_loss'])

    plt.figure(figsize=(20, 10))

    if plot_train:
        if has_rec_loss:
            plt.plot(np.linspace(1, len(train_loss), len(train_rec_loss)),
                     train_rec_loss, label='Train Reconstruction loss')
        if has_gender_loss:
            plt.plot(np.linspace(1, len(train_loss), len(train_gender_loss)),
                     train_gender_loss, label='ATrain Gender loss')

    if plot_val:
        if has_rec_loss:
            plt.plot(np.linspace(1, len(train_loss), len(val_rec_loss)),
                     val_rec_loss, label='Val Reconstruction loss')
        if has_gender_loss:
            plt.plot(np.linspace(1, len(train_loss), len(val_gender_loss)),
                     val_gender_loss, label='Val Gender loss')

    plt.xlabel("Iterations", fontsize=18)
    plt.ylabel("Train/Val loss", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    legend = plt.legend(fontsize=14)
    for text in legend.get_texts():
        text.set_color("black")

    plt.show()

    if has_rec_loss:
        print("Final training reconstruction loss: {}".format(
            train_rec_loss[-1]))
        print("Final validation reconstruction loss: {}".format(
            val_rec_loss[-1]))
    if has_gender_loss:
        print("Final training gender loss: {}".format(train_gender_loss[-1]))
        print("Final validation gender loss: {}".format(val_gender_loss[-1]))


def plot_gender_prediction(gender, gender_pred, color='black'):
    """
    Create plot for the confusion matrix for binary gender prediction and
    compute scores for accuracy, precision, recall, f1 and auc

    args:
        gender (np.array): one-hot encoded array of true gender values
        gender_pred (np.array): one-hot encoded array of predicted gender values
        color (str): Color for matplotlib text, axes labels and axes ticks
    """
    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    # get confusion matrix
    cm = metrics.confusion_matrix(gender, gender_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # compute scores
    accuracy_score = metrics.accuracy_score(gender, gender_pred)
    precision_score = metrics.precision_score(gender, gender_pred)
    recall_score = metrics.recall_score(gender, gender_pred)

    # plot figures
    plt.figure(figsize=(8, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        xticklabels=['M', 'F'],
        yticklabels=['M', 'F'],
        fmt=".3f",
        linewidths=.5,
        square=True,
        cmap='Blues',
    )
    plt.ylabel('Actual label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    all_sample_title = 'Accuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\n'.format(
        accuracy_score, precision_score, recall_score)
    plt.title(all_sample_title, size=22)


def plot_t_sne(embedding, labels, info, color='black'):
    """
    Plot the T-SNE graph of an embedding

    args:
        embedding (np.array): embedding of the autoencoder
        labels (np.array): labels of the embedding
        info (list of str, len=2): state which embedding (e.g. 'validation set')
                                   and what kind of labels (e.g. 'gender')
    """
    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    tsne = TSNE(n_components=2).fit_transform(embedding)
    plt.figure(figsize=(20, 12))
    plt.title("T-SNE of {} embedding with {} labels\n".format(
        info[0], info[1]), fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    sns.scatterplot(
        x=tsne[:, 0], y=tsne[:, 1],
        hue=labels,
        palette=sns.color_palette("hls", 2),
        legend="full",
        alpha=0.5
    )
    l = plt.legend(['F', 'M'], fontsize=16)
    for text in l.get_texts():
        text.set_color("black")


def plot_saliency_map_sample(model, x, gender, threshold=0.1, color='black'):
    """
    Create plot of signal where points with a high saliency score are
    highlighted

    args:
        model (nn.Module): gender classification model
        x (torch.tensor): input signal, shape: (1, 1, signal_length)
        gender (int): true gender class
        threshold (float): Threshold of saliency score. Only saliency scores
                           above the threshold will be plotted
        color (str): Color for matplotlib text, axes labels and axes ticks
    """

    assert model.predict_gender

    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    GBP = GuidedBackprop(model)
    colors = ['blue', 'red']

    grads = GBP.generate_gradients(x, gender)

    _, score = model(x)
    score = torch.nn.functional.softmax(score, dim=1)
    score = score[0].detach().numpy()

    indices = np.argwhere(np.abs(grads) > threshold)

    x = x[0, 0].detach().numpy()
    plt.figure(figsize=(20, 4))
    plt.title("Saliency map of a {} sample, score: {} (0: male, 1: female)".format(
        GENDER_ENUM(gender), score), fontsize=22)
    plt.plot(x, color='gray')
    plt.scatter(indices, x[indices], marker='o', color=colors[gender])
    plt.show()


def plot_saliency_maps(model, dataset, num_samles, threshold=0.1, color='black'):
    """
    Create saliency maps for num_samples random samples

    args:
        model (nn.Module): gender classification model
        dataset (torch.utils.data.Dataset): dataset with ecg signals
        num_samples (int): number of saliency maps to be plotted
        threshold (float): Threshold of saliency score. Only saliency scores
                           above the threshold will be plotted
        color (str): Color for matplotlib text, axes labels and axes ticks
    """

    assert model.predict_gender

    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)

    for i_sample in range(num_samles):
        x, _, gender = next(iter(data_loader))
        gender = gender[0]
        plot_saliency_map_sample(model, x, gender, threshold, color)


def plot_selected_saliency_maps(
        model, x, gender, num_samples,
        inverted=False, score_threshold=0.95,
        plot_threshold=0.1, color='black'):
    """
    Create saliency maps for num_samples samples which have a high
    classification score

    args:
        model (nn.Module): gender classification model
        x (torch.tensor): tensor with input signals (preferably large)
        gender (np.array): corresponding true gender classes
        num_samples (int): maximum number of saliency maps to be plotted
        inverted (bool): If true, show samples which are confidently wrong
                         classified by the model
        score_threshold (float): Threshold for the classification score. Only
                                 samples with higher classification score will
                                 be considered
        plot_threshold (float): Threshold for the saliency score. Only saliency
                                scores above the threshold will be plotted
        color (str): Color for matplotlib text, axes labels and axes ticks
    """

    assert model.predict_gender

    _, logits = model(x)
    scores = torch.nn.functional.softmax(logits, dim=1).detach().numpy()

    if inverted:
        male = 1
        female = 0
    else:
        male = 0
        female = 1

    # Plot male
    num_plotted = 0
    max_idx = np.argwhere(scores[:, 0] > score_threshold)
    for idx in max_idx:
        idx = idx[0]
        if gender[idx] == male:
            x_plot = x[idx].view(-1, *x.shape[1:])
            plot_saliency_map_sample(model, x_plot, gender[idx],
                                     plot_threshold, color)
            num_plotted += 1
            if num_plotted >= num_samples:
                break

    # Plot female
    num_plotted = 0
    max_idx = np.argwhere(scores[:, 1] > score_threshold)
    for idx in max_idx:
        idx = idx[0]
        if gender[idx] == female:
            x_plot = x[idx].view(-1, *x.shape[1:])
            plot_saliency_map_sample(model, x_plot, gender[idx],
                                     plot_threshold, color)
            num_plotted += 1
            if num_plotted >= num_samples:
                break


def create_signal_which_maximizes_activation(model, layer, filt, input_size,
                                             lr=0.1, opt_steps=100,
                                             upscaling_steps=5,
                                             upscaling_factor=2.0,
                                             color='black'):
    """
    Create plot of artificial signal which maximizes the activation of
    a filter at a layer in a model using gradient ascent
    Source: https://github.com/fg91/visualizing-cnn-feature-maps/blob/master/filter_visualizer.ipynb

    args:
        model (nn.Module): any convolutional model
        layer (int): index of layer
        filt (int): index of filter
        input_size (tuple): shape of input signal expected from model
        lr (float): learning rate for gradient ascent optimizer
        opt_steps (int): number of training steps for gradient ascent
        upscaling_steps (int): number of upscaling steps during training
        upscaling_factor (float): factor of upscaling
        color (str): Color for matplotlib text, axes labels and axes ticks
    """
    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    img_var = torch.randn((1, 1, int(input_size * ((1 / upscaling_factor)**upscaling_steps))))
    activations = SaveFeatures(list(model.children())[layer])
    optimizer = torch.optim.Adam(
        [img_var.requires_grad_()], lr=lr, weight_decay=1e-6)
    loss_history = []

    for step in range(upscaling_steps + 1):
        for n in range(opt_steps):
            optimizer.zero_grad()
            model(img_var)
            loss = -activations.features[:, filt].mean()
            loss_history.append(loss)
            loss.backward()
            optimizer.step()

        if step < upscaling_steps:
            img_var = torch.nn.functional.interpolate(
                img_var, scale_factor=upscaling_factor, mode='linear')

    plt.figure(figsize=(20, 4))
    plt.plot(img_var.clone().detach().numpy()[0, 0])
    plt.title("Input which maximizes activation of layer: conv_{}, filter: {}".format(
        layer + 1, filt), fontsize=22)
    plt.show()

    return img_var


def create_signal_which_maximizes_class_score(
        model, target_class, input_size, lr=0.1, iterations=500, color='black'):
    """
    Create plot of artificial signal which maximizes the score of a target class
    using gradient ascent

    args:
        model (nn.Module): any model
        input_size (tuple): shape of input signal expected from model
        lr (float): learning rate for gradient ascent optimizer
        iterations (int): number of training steps for gradient ascent
        color (str): Color for matplotlib text, axes labels and axes ticks
    """

    mpl.rcParams['text.color'] = color
    mpl.rcParams['axes.labelcolor'] = color
    mpl.rcParams['xtick.color'] = color
    mpl.rcParams['ytick.color'] = color

    model.eval()
    img_var = torch.randint(-4, 4, (1, 1, input_size), dtype=torch.float32)
    img_var.requires_grad = True
    optimizer = torch.optim.SGD([img_var], lr=lr)

    for i in range(1, iterations):
        _, gender_pred = model(img_var)
        class_loss = -gender_pred[0, target_class]
        print('Iteration:', str(i), 'Loss', "{0:.2f}".format(
            class_loss.data.numpy()))
        model.zero_grad()
        class_loss.backward()
        optimizer.step()

    plt.plot(img_var[0, 0].detach().numpy())
    plt.show()
