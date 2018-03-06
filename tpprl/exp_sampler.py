import decorated_options as Deco
import numpy as np

class CDFSampler:
    """A generic sampler which assumes that the intensity u(t) has the form:

    f(vt * hi + bt + wt * (t - t0)).

    The function 'f' is left unimplemented and has to be implemented by sub-classes.

    Notationally, c = vt * hi + bt.
    """

    @Deco.optioned()
    def __init__(self, vt, wt, bt, init_h, t_min, seed=42):
        self.seed = seed
        self.vt = np.asarray(vt).squeeze()
        self.wt = np.asarray(wt).squeeze()
        self.bt = np.asarray(bt).squeeze()

        self.w = np.asarray(wt).squeeze()
        self.Q = 1.0

        self.random_state = np.random.RandomState(seed)
        self.reset(t_min, init_h, reset_sample=True)

    def reset_only_sample(self, cur_time):
        """Resets only the present sample.

        This allows generating multiple samples (by updating t0) from the
        intensity one after the other without registering any new events."""

        self.c = self.c + (self.w * (cur_time - self.t0))
        self.t0 = cur_time
        self.u_unif = self.random_state.rand()
        self.Q = 1.0

        return self.generate_sample()

    def cdf(self, t):
        """Return the CDF calculated at 't', given the current state of the intensity.
        It also assumes that the last event was at self.t0.
        """
        raise NotImplementedError('cdf has to be implemented by the sub-class')

    def reset(self, cur_time, init_h, reset_sample):
        """Reset the sampler for generating another event."""

        if reset_sample:
            self.u_unif = self.random_state.rand()
            self.Q = 1.0
        else:
            self.Q *= (1 - self.cdf(cur_time))

        self.h = init_h
        self.c = np.squeeze(self.vt.dot(self.h) + self.bt)
        self.t0 = cur_time

        return self.generate_sample()

    def register_event(self, time, new_h, own_event):
        """Saves the event and generated a new time for the next event."""
        return self.reset(time, new_h, reset_sample=own_event)

    def get_last_hidden_state(self):
        return self.h

    def generate_sample(self):
        """Find a sample from the Exp process."""
        raise NotImplementedError('generate_sample has to be implemented by the sub-class.')


class ExpCDFSampler(CDFSampler):
    """This is an exponential sampler."""

    def cdf(self, t):
        """Calculates the CDF assuming that the last event was at self.t0"""
        return 1 - np.exp((np.exp(self.c) / self.w) * (1 - np.exp(self.w * (t - self.t0))))

    def generate_sample(self):
        """Find a sample from the Exp process."""
        # Have the uniform sample already drawn
        D = 1 - (self.w / np.exp(self.c)) * np.log((1 - self.u_unif) / self.Q)
        if D <= 0:
            # This is the probability that no event ever happens
            return np.inf
        else:
            return self.t0 + (1 / self.w) * np.log(D)

    def calc_LL(self, event_time_deltas, survival_time, hidden_states, init_h):
        """Calculates the likelihood of the given event time deltas and
        hidden-states."""
        LL = 0

        def log_u(tau, h):
            return self.bt + self.vt.dot(h) + self.wt * tau

        for t, h in zip(event_time_deltas,
                        [init_h] + hidden_states[:-1]):
            u_t_term = log_u(t, h)
            u_0_term = log_u(0, h)
            # TODO: Which 'h' will be used here in the second term?
            # TODO (should only count events which have been created by us, not merely events which changed the hidden state)
            LL_log = u_t_term
            LL_int = (1 / self.wt) * (np.exp(u_t_term) - np.exp(u_0_term))
            LL += LL_log - LL_int

        last_hidden_state = hidden_states[-1]
        LL -= (1 / self.wt) * (np.exp(log_u(survival_time, last_hidden_state)) -
                               np.exp(log_u(0, last_hidden_state)))
        return LL

    def calc_quad_loss(self, event_time_deltas, survival_time, hidden_states, init_h, q):
        """Calculates the loss incurred by the given event time deltas and hidden-states."""
        loss = 0

        def u_2(tau, h):
            return np.exp(2 * (self.bt + self.vt.dot(h) + self.wt * tau))

        for t, h in zip(event_time_deltas,
                        [init_h] + hidden_states[:-1]):
            u_t_term = u_2(t, h)
            u_0_term = u_2(0, h)
            # TODO: Which 'h' will be used here in the second term?
            # TODO: This can be made more numerically stable by using the identity:
            # a^2 - b^2 = (a + b) * (a - b)
            # Prevents squaring, at least.
            loss += (1 / (2 * self.wt)) * (u_t_term - u_0_term)

        last_hidden_state = hidden_states[-1]
        loss += (1 / (2 * self.wt)) * (u_2(survival_time, last_hidden_state) -
                                       u_2(0, last_hidden_state))
        return loss


class SigmoidCDFSampler(CDFSampler):
    """This is an sigmoidal intensity sampler.

    Additionally, it assumes that the sigmoid is multiplied by 'k' to scale the intensity.
    """

    @Deco.optioned()
    def __init__(self, vt, wt, bt, init_h, t_min, seed=42, k=1.0):
        self.k = k
        super().__init__(vt, wt, bt, init_h, t_min, seed=seed)

    def cdf(self, t):
        C = (1 + np.exp(self.c)) / (1 + np.exp(self.c + self.wt * (t - self.t0)))
        return 1 - C ** (self.k / self.wt)

    def generate_sample(self):
        D = (1 + np.exp(self.c)) * ((1 - self.u_unif) / self.Q) ** (- self.k / self.wt) - 1
        # print('D = ', D)
        if D <= 0:
            # This is the case when no event ever happens.
            return np.inf
        else:
            return self.t0 + (np.log(D) - self.c) / self.wt
