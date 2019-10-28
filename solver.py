import datetime
import os
import time

import torch

from utils import time_left, plot_grad_flow


class Solver:
    def __init__(self):
        self.history = {'train_loss': [],
                        'train_rec_loss': [],
                        'train_gender_loss': [],
                        'val_loss': [],
                        'val_rec_loss': [],
                        'val_gender_loss': [],
                        }

        self.optim = []
        self.criterion = []
        self.training_time_s = 0
        self.stop_reason = ''
        self.config = {}

    def train(
            self,
            model,
            train_loader,
            val_loader,
            config,
            tensorboard_writer,
            optim=None,
            scheduler=None,
            rec_criterion=torch.nn.SmoothL1Loss(),
            gender_criterion=torch.nn.CrossEntropyLoss(),
            rec_weight=1.,
            gender_weight=1.,
            l1_penalty=0,
            noise=0,
            num_epochs=10,
            max_train_time_s=None,
            save_after_epochs=None,
            encoder=None,
            save_path='saves',
            device='cpu',
            show_gradient_flow=False,
    ):
        model.to(device)

        self.config = config

        start_epoch = len(self.history['val_loss'])

        if start_epoch == 0:
            self.optim = optim
            self.scheduler = scheduler
            self.rec_criterion = rec_criterion
            self.gender_criterion = gender_criterion

        iter_per_epoch = len(train_loader)
        print("Iterations per epoch: {}".format(iter_per_epoch))

        # Path to save model and solver
        save_path = os.path.join(save_path,
                                 config['model'] + '_train' + datetime.datetime.now().strftime("_%d_%H_%M_%S_") + str(config['num_epochs']) + 'E')

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs * iter_per_epoch
        i_iter = 0

        # Init average train loss
        avg_train_loss = 0.
        avg_train_rec_loss = 0.
        avg_train_gender_loss = 0.
        train_gender_acc = 0.

        best_val_loss = 1e9

        t_start_training = time.time()

        print('Start training at epoch ' + str(start_epoch + 1))

        # Do the training here
        for i_epoch in range(num_epochs):
            print("Starting epoch {} / {}".format(start_epoch + i_epoch + 1, start_epoch + num_epochs))

            t_start_epoch = time.time()
            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                i_iter += 1

                x, gender = batch
                x = x.to(device)
                gender = gender.to(device)

                if model.is_autoencoder:
                    target = x

                if noise != 0.:
                    x += torch.distributions.Normal(0, noise).sample(x.shape).to(device)

                if encoder is not None:
                    x = encoder(x)

                # Forward pass
                y_rec, gender_pred = model(x)

                reconstruction_loss = torch.tensor(0.).to(device)
                gender_prediction_loss = torch.tensor(0.).to(device)
                l1_loss = torch.tensor(0.).to(device)

                if model.is_autoencoder:
                    reconstruction_loss = self.rec_criterion(y_rec, target)

                if model.predict_gender:
                    gender_prediction_loss = self.gender_criterion(gender_pred, gender.to(device))
                    _, y_ = torch.max(gender_pred, 1)
                    train_gender_acc += torch.sum(y_ == gender.data)

                if l1_penalty != 0:
                    for param in model.parameters():
                        l1_loss += torch.sum(torch.abs(param))

                loss = rec_weight * reconstruction_loss + \
                    gender_weight * gender_prediction_loss + \
                    l1_loss * l1_penalty

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()

                if show_gradient_flow:
                    plot_grad_flow(model.named_parameters())

                self.optim.step()

                # Save loss to history
                self.history['train_loss'].append(loss.item())
                avg_train_loss += loss.item()
                avg_train_rec_loss += rec_weight * reconstruction_loss.item()
                avg_train_gender_loss += gender_weight * gender_prediction_loss.item()

            avg_train_loss /= len(train_loader)
            avg_train_rec_loss /= len(train_loader)
            avg_train_gender_loss /= len(train_loader)

            # Add train loss to tensorboard
            tensorboard_writer.add_scalar('train_loss', avg_train_loss, i_iter)
            tensorboard_writer.add_scalar('rec_loss', avg_train_rec_loss, i_iter)
            tensorboard_writer.add_scalar('gender_loss', avg_train_gender_loss, i_iter)

            # Add train loss to loss history
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_rec_loss'].append(avg_train_rec_loss)
            self.history['train_gender_loss'].append(avg_train_gender_loss)

            # Validate model
            print("\nValidate model after epoch {} / {}".format(i_epoch + 1, num_epochs))

            # Set model to evaluation mode
            model.eval()

            val_loss = 0.
            val_rec_loss = 0.
            val_gender_loss = 0.
            val_gender_acc = 0.

            for i, batch in enumerate(val_loader):
                x, gender = batch
                x = x.to(device)
                gender = gender.to(device)

                if encoder is not None:
                    x = encoder(x)

                # Forward pass
                y_rec, gender_pred = model(x)

                reconstruction_loss = torch.tensor(0.).to(device)
                gender_prediction_loss = torch.tensor(0.).to(device)
                l1_loss = torch.tensor(0.).to(device)

                if model.is_autoencoder:
                    if noise != 0.:
                        noise_added = torch.distributions.Normal(0, noise).sample(x.shape).to(device)
                        x += noise_added
                    reconstruction_loss = self.rec_criterion(y_rec, x)

                if model.predict_gender:
                    gender_prediction_loss = self.gender_criterion(gender_pred, gender.to(device))
                    _, y_ = torch.max(gender_pred, 1)
                    val_gender_acc += torch.sum(y_ == gender.data)

                if l1_penalty != 0:
                    for param in model.parameters():
                        l1_loss += torch.sum(torch.abs(param))

                temp_loss = rec_weight * reconstruction_loss + \
                    gender_weight * gender_prediction_loss + \
                    l1_loss * l1_penalty

                val_loss += temp_loss.item()
                val_rec_loss += rec_weight * reconstruction_loss.item()
                val_gender_loss += gender_weight * gender_prediction_loss.item()

            val_loss /= len(val_loader)
            val_rec_loss /= len(val_loader)
            val_gender_loss /= len(val_loader)

            # Add validation loss to tensorboard
            tensorboard_writer.add_scalar('val_loss', val_loss, i_iter)
            tensorboard_writer.add_scalar('val_rec_loss', val_rec_loss, i_iter)
            tensorboard_writer.add_scalar('val_gender_loss', val_gender_loss, i_iter)

            # Add validation loss to history
            self.history['val_loss'].append(val_loss)
            self.history['val_rec_loss'].append(val_rec_loss)
            self.history['val_gender_loss'].append(val_gender_loss)

            # update the learning rate
            scheduler.step(val_loss)

            # Logging
            log_str = '\n   Train loss: \t\t\t{:.4f} | Val loss: \t\t\t{:.4f}'.format(
                avg_train_loss, val_loss)
            if model.is_autoencoder:
                log_str += '\n   Train rec loss: \t{:.4f} | Val rec loss: \t{:.4f}'.format(
                    avg_train_rec_loss, val_rec_loss)
            if model.predict_gender:
                train_gender_acc = train_gender_acc.double() / (len(train_loader) * config['batch_size'])
                val_gender_acc = val_gender_acc.double() / (len(val_loader) * config['batch_size'])
                log_str += '\n   Train gender loss: \t{:.4f} | Val gender loss: \t{:.4f}'.format(
                    avg_train_gender_loss, val_gender_loss)
                log_str += '\n   Train gender acc: \t{:.4f} | Val gender acc: \t{:.4f}'.format(
                    train_gender_acc, val_gender_acc
                )

            log_str += "\n   Time elapsed: {}s".format(
                datetime.datetime.now() - datetime.datetime.fromtimestamp(t_start_training))
            log_str += '   time left: {}\n'.format(
                time_left(t_start_training, n_iters, i_iter))

            print(log_str)

            # Save best model
            if val_gender_loss < best_val_loss:
                best_val_loss = val_gender_loss
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path + '/best_model')
                model.to(device)

            # Save model and solver
            if save_after_epochs is not None and ((i_epoch + 1) % save_after_epochs == 0):
                os.makedirs(save_path, exist_ok=True)
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.training_time_s += time.time() - t_start_training
                self.save(save_path + '/solver' + str(i_epoch + 1))
                model.to(device)

            # Stop if training time is over
            if max_train_time_s is not None and (time.time() - t_start_training > max_train_time_s):
                print("Training time is over.")
                self.stop_reason = "Training time over."
                break

        if self.stop_reason is "":
            self.stop_reason = "Reached number of specified epochs."

        # Save model and solver after training
        os.makedirs(save_path, exist_ok=True)
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.training_time_s += time.time() - t_start_training
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.')

    def save(self, path):
        print('Saving solver... %s\n' % path)
        torch.save({
            'history': self.history,
            'stop_reason': self.stop_reason,
            'training_time_s': self.training_time_s,
            'criterion': self.criterion,
            'optim_state_dict': self.optim.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path, device, only_history=False):
        """
        Load solver from checkpoint
        """
        checkpoint = torch.load(path, map_location=device)

        if not only_history:
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            self.criterion = checkpoint['criterion']

        self.history = checkpoint['history']
        self.stop_reason = checkpoint['stop_reason']
        self.training_time_s = checkpoint['training_time_s']
        self.config = checkpoint['config']

