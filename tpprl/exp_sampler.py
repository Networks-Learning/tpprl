import decorated_options as Deco
import numpy as np

class ExpCDFSampler:
    """This is an exponential sampler."""

    @Deco.optioned()
    def __init__(self, vt, wt, bt, init_h, t_min, seed=42):
        self.seed = seed
        self.vt = np.asarray(vt).squeeze()
        self.wt = np.asarray(wt).squeeze()
        self.bt = np.asarray(bt).squeeze()

        self.w = np.asarray(wt).squeeze()
        self.random_state = np.random.RandomState(seed)
        self.reset(t_min, init_h, reset_sample=True)

    def reset_only_sample(self, cur_time):
        """Resets only the present sample.
        This does not change the c, but only the t0 and generates another sample."""

        self.c = self.c * np.exp(-self.w * (cur_time - self.t0))
        self.t0 = cur_time
        self.u_unif = self.random_state.rand()

        return self.generate_sample()

    def cdf(self, t):
        """Calculates the CDF assuming that the last event was at self.t0"""
        return 1 - np.exp(- (self.c / self.w) * (1 - np.exp(- self.w * (t - self.t0))))

    def reset(self, cur_time, init_h, reset_sample):
        """Reset the sampler for generating another event."""

        if reset_sample:
            self.u_unif = self.random_state.rand()
        else:
            self.u_unif -= self.cdf(cur_time)

        self.h = init_h
        self.c = np.squeeze(np.exp(self.vt.dot(self.h) + self.bt))
        self.t0 = cur_time

        return self.generate_sample()

    def register_event(self, time, new_h, own_event):
        """Saves the event and generated a new time for the next event."""
        self.reset(time, new_h, reset_sample=own_event)
        return self.generate_sample()

    def get_last_hidden_state(self):
        return self.h

    def generate_sample(self):
        """Find a sample from the Exp process."""
        # Have the uniform sample already drawn
        D = 1 + (self.w / self.c) * np.log(1 - self.u_unif)
        if D <= 0.0:
            # This is the probability that no event ever happens
            return np.inf
        else:
            return self.t0 - (1 / self.w) * np.log(D)

