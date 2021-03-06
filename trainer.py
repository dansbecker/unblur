import numpy as np
from time import time
from collections import namedtuple
import pandas as pd

class Trainer(object):
    def __init__(self, gen_model, disc_model, gen_disc_model, data_feeder,
                 report_freq=20, start_new_hist_file=True,
                 train_hist_path='./tmp/train_hist.csv'):
        '''
        Object to train generator and discriminator models

        Inputs:
        -------
            gen_model, disc_model, gen_disc_model: Model objects to keep and train
            data_feeder: Data feeder object that returns training batches on request
            report_freq: number of batches between printed outputs
            start_new_hist_file: If True, old training history data is ignored & overwritten
                                 If False, old training history is appended to.
                                 This does not affect training itself, just what is recorded
                                 about training.
            train_hist_path: File path to save to (and optionally to read from)
        '''

        self.step_count = 0
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.gen_disc_model = gen_disc_model
        self.data_feeder = data_feeder
        self.report_freq = report_freq
        self.train_hist_path = train_hist_path
        if start_new_hist_file:
            self.g_loss = []
            self.d_loss = []
            self.when_was_gen_trained = []
            self.when_was_disc_trained = []
            self._update_disc()     # run once to avoid breaking "catching-up" logic in train step
            self._update_gen()
        else:
            self._load_prev_hist()

        print('Iteration   Gen Loss    Disc Loss   New Gen updates  New Disc updates')

    def _load_prev_hist(self):
        '''Updates object's training history data from saved file'''
        all_hist = pd.read_csv(self.train_hist_path)
        self.g_loss = all_hist.g_loss.tolist()
        self.d_loss = all_hist.d_loss.tolist()
        self.when_was_gen_trained = all_hist.when_was_gen_trained.tolist()
        self.when_was_disc_trained = all_hist.when_was_disc_trained.to_list()
        return


    def _save_hist(self):
        '''Save training history data to file'''
        all_hist = pd.DataFrame({'g_loss': self.g_loss,
                                 'd_loss': self.d_loss,
                                 'g_was_trained': self.when_was_gen_trained,
                                 'd_was_trained': self.when_was_disc_trained})
        all_hist.to_csv(self.train_hist_path, index=False)

    def train(self, n_steps=5000):
        '''Train the generator and discriminator stored in this object for n_step batches'''
        training_start_time = time()
        for i in range(n_steps):
            self._train_step()
            if len(self.g_loss) % self.report_freq == 0:
                self.report()
        # self.serialize_models()
        self._save_hist()
        print('Total training time: %.0f s' %(time() - training_start_time))

    def get_models(self):
        '''Returns the object's generator, discriminator and stacked model.

        These are not guaranteed to be trained. The caller is responsible for calling
        train method beforehand'''
        raise NotImplementedError


    def _train_step(self):
        '''
        Train models on a single batch.
        Whether the generator, discriminator or both are trained depends on previous scores.
        Training history data is updated too.
        '''

        if (self.g_loss[-1] > 1):  # let generator catch up
            self._update_gen()
            self._update_disc(eval_only=True)
        elif self.d_loss[-1] > 1: # let discriminator catch up
            self._update_disc()
            self._update_gen(eval_only=True)
        else:                                                      # update both discrim and gen
            self._update_disc()
            self._update_gen()


    def get_models(self):
        return self.gen_model, self.disc_model, self.gen_disc_model

    def report(self):
        '''
        Print 1 line containing most recent training status.
        Header for this table is printed in the object's constructor
        '''

        n_steps = len(self.g_loss)

        recent_g_updates = sum(self.when_was_gen_trained[-self.report_freq:])
        recent_d_updates = sum(self.when_was_disc_trained[-self.report_freq:])
        print("%.0f        %.4f       %.4f           %.0f             %.0f"
                %(n_steps, self.g_loss[-1], self.d_loss[-1],
                recent_g_updates, recent_d_updates))

    def _set_disc_trainability(self, model, trainable_mode):
        '''
        Sets trainable property of layers with names including 'disc' to value of trainable_mode
        Trainable_mode is either True or False
        '''
        relevant_layers = [layer for layer in model.layers if 'disc' in layer.name]
        for layer in relevant_layers:
            layer.trainable = trainable_mode
        return


    def _update_gen(self, eval_only=False):
        '''
        Run generator on 1 training batch. Update stored scores
        If eval_only=True, we do not update the generator params
        Otherwise we update the weights too.
        '''
        blur, is_real = self.data_feeder.get_gen_only_batch()
        if eval_only:
            new_loss = self.gen_disc_model.test_on_batch(blur, is_real)
            gen_was_trained = False
        else:
            self._set_disc_trainability(self.gen_disc_model, False)
            new_loss = self.gen_disc_model.train_on_batch(blur, is_real)
            self._set_disc_trainability(self.gen_disc_model, True)
            gen_was_trained = True

        self.g_loss.append(new_loss)
        self.when_was_gen_trained.append(gen_was_trained)
        return


    def _update_disc(self, eval_only=False):
        '''
        Run discriminator on 1 training batch. Update stored scores
        If eval_only=True, we do not update the discriminator params
        Otherwise we update the weights too.
        '''
        blur, clear, is_real = self.data_feeder.get_batch(self.gen_model)
        X = [blur, clear]
        if eval_only:
            new_loss = self.disc_model.test_on_batch(X, is_real)
            disc_was_trained = False
        else:
            new_loss = self.disc_model.train_on_batch(X, is_real)
            disc_was_trained = True

        self.d_loss.append(new_loss)
        self.when_was_disc_trained.append(disc_was_trained)
