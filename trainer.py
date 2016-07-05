import numpy as np
from time import time

class Trainer(object):
    def __init__(self, gen_model, disc_model, gen_disc_model, data_feeder, report_freq=20, serialize_freq=np.inf):
        self.step_count = 0
        self.gen_model = gen_model
        self.disc_model = disc_model
        self.gen_disc_model = gen_disc_model
        self.data_feeder = data_feeder
        self.report_freq = report_freq
        self.serialize_freq = serialize_freq
        self.g_loss = []
        self.d_loss = []
        self.n_steps = 1
        self._update_disc()     # Immediately run the discriminator and generator once
        self._update_gen()
        print('Iteration   Gen Loss    Disc Loss    Gen updates   Disc updates')

    def train(self, n_steps=5000):
        training_start_time = time()
        for i in range(n_steps):
            self._train_step()
            if self.n_steps % self.report_freq == 0:
                self.report()
            if self.n_steps % self.serialize_freq == 0:
                self.serialize_models()
        print('Total training time: %.0f s' %(time() - training_start_time))

    def serialize_models(self):
        raise NotImplementedError


    def _train_step(self):
        self.n_steps +=1
        if ((self.g_loss[-1] > 1) and (self.d_loss[-1] < 0.5)):  # let generator catch up
            self._update_gen()
        elif ((self.g_loss[-1] < 0.5) and (self.d_loss[-1] > 1)): # let discriminator catch up
            self._update_disc()
        else:  # run both discrim and gen
            self._update_disc()
            self._update_gen()

    def get_models(self):
        return self.gen_model, self.disc_model, self.gen_disc_model

    def report(self):
            print("%.0f        %.4f       %.4f       %.0f          %.0f"
                    %(self.n_steps, self.g_loss[-1], self.d_loss[-1],
                      len(self.g_loss), len(self.d_loss)))

    def _set_disc_trainability(self, model, trainable_mode):
        '''
        Sets trainable property of layers with names including 'disc' to value of trainable_mode
        Trainable_mode is either True or False
        '''
        relevant_layers = [layer for layer in model.layers if 'disc' in layer.name]
        for layer in relevant_layers:
            layer.trainable = trainable_mode
        return


    def _update_gen(self):
        self._set_disc_trainability(self.gen_disc_model, False)
        blur, is_real = self.data_feeder.get_gen_only_batch()
        self.g_loss.append(self.gen_disc_model.train_on_batch(blur, is_real))
        self._set_disc_trainability(self.gen_disc_model, True)

    def _update_disc(self):
        '''train the discriminator model.'''
        blur, clear, is_real = self.data_feeder.get_batch(self.gen_model)
        X = [blur, clear]
        self.d_loss.append(self.disc_model.train_on_batch(X, is_real))
