import redqueen.opt_model as OM
import warnings
import tensorflow as tf
import decorated_options as Deco
from exp_sampler import ExpCDFSampler

class ExpBroadcaster(OM.Broadcaster):

    @Deco.optioned()
    def __init__(self, src_id, seed, trainer, t_min=0):
        super(ExpBroadcaster, self).__init__(src_id, seed)
        self.init = False

        self.trainer = trainer

        params = Deco.Options(**self.trainer.sess.run({
            'Wm': trainer.tf_Wm,
            'Wh': trainer.tf_Wh,
            'Bh': trainer.tf_Bh,
            'Wt': trainer.tf_Wt,
            'Wr': trainer.tf_Wr,

            'wt': trainer.tf_wt,
            'vt': trainer.tf_vt,
            'bt': trainer.tf_bt,
            'init_h': trainer.tf_h
        }))

        self.cur_h = params.init_h

        # Hidden state parameters
        # self.Wh    = Wh
        # self.Wt    = Wt
        # self.Bh    = Bh
        # self.Wm    = Wm
        # self.cur_h = init_h

        # self.bt    = bt
        # self.wt    = wt
        # self.vt    = vt

        self.exp_sampler = ExpCDFSampler(_opts=params,
                                         t_min=t_min,
                                         seed=seed + 1)

    def update_hidden_state(self, src_id, time_delta):
        """Returns the hidden state after a post by src_id and time delta."""
        # Best done using self.sess.run here.
        r_t = self.state.get_wall_rank(self.src_id, self.sink_ids, dict_form=False)

        feed_dict = {
            self.trainer.tf_b_idx: np.asarray([self.trainer.src_embed_map[src_id]]),
            self.trainer.tf_t_delta: np.asarray([time_delta]).reshape(-1),
            self.trainer.tf_h: self.cur_h,
            self.trainer.tf_rank: np.asarray([np.mean(r_t)]).reshape(-1)
        }
        return self.trainer.sess.run(self.trainer.tf_h_next,
                                     feed_dict=feed_dict)

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            # Nothing special to do for the first event.

        self.state.apply_event(event)

        if event is None:
            # This is the first event. Post immediately to join the party?
            # Or hold off?
            return self.exp_sampler.generate_sample()
        else:
            self.cur_h = self.update_hidden_state(event.src_id, event.time_delta)
            next_post_time = self.exp_sampler.register_event(
                                    event.cur_time,
                                    self.cur_h,
                                    own_event=event.src_id == self.src_id)
            next_delta = (next_post_time - self.last_self_event_time)[0]
            # print(next_delta)
            assert next_delta >= 0
            return next_delta


OM.SimOpts.registerSource('ExpBroadcaster', ExpBroadcaster)

class ExpTrainer:

    @Deco.optioned()
    def __init__(self, Wm, Wh, Wt, Wr, Bh, vt, wt, bt, init_h, sess, sim_opts,
                 scope=None, t_min=0):
        """Initialize the trainer with the policy parameters."""

        num_broadcasters = len(sim_opts.other_sources) + 1

        self.src_embed_map = {x.src_id: idx + 1
                              for idx, x in enumerate(sim_opts.create_other_sources())}
        self.src_embed_map[sim_opts.src_id] = 0
        init_h = np.reshape(init_h, (-1, 1))
        Bh = np.reshape(Bh, (-1, 1))

        # TODO: Create all these variables on the CPU?
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("hidden_state"):
                self.tf_Wm = tf.get_variable(name="Wm", shape=Wm.shape,
                                             initializer=tf.constant_initializer(Wm))
                self.tf_Wh = tf.get_variable(name="Wh", shape=Wh.shape,
                                             initializer=tf.constant_initializer(Wh))
                self.tf_Wt = tf.get_variable(name="Wt", shape=Wt.shape,
                                             initializer=tf.constant_initializer(Wt))
                self.tf_Wr = tf.get_variable(name="Wr", shape=Wr.shape,
                                             initializer=tf.constant_initializer(Wr))
                self.tf_Bh = tf.get_variable(name="Bh", shape=Bh.shape,
                                             initializer=tf.constant_initializer(Bh))

                self.tf_h = tf.get_variable(name="h", shape=init_h.shape,
                                            initializer=tf.constant_initializer(init_h))
                self.tf_b_idx = tf.placeholder(name="b_idx", shape=1, dtype=tf.int32)
                self.tf_t_delta = tf.placeholder(name="t_delta", shape=1, dtype=tf.float32)
                self.tf_rank = tf.placeholder(name="rank", shape=1, dtype=tf.float32)

                self.tf_h_next = tf.nn.relu(
                    tf.transpose(
                        tf.nn.embedding_lookup(self.tf_Wm, self.tf_b_idx, name="b_embed")
                    ) +
                    tf.matmul(self.tf_Wh, self.tf_h) + self.tf_Bh +
                    self.tf_Wr * self.tf_rank +
                    self.tf_Wt * self.tf_t_delta,
                    name="h_next"
                )

            with tf.variable_scope("output"):
                self.tf_bt = tf.get_variable(name="bt", shape=bt.shape,
                                             initializer=tf.constant_initializer(bt))
                self.tf_vt = tf.get_variable(name="vt", shape=vt.shape,
                                             initializer=tf.constant_initializer(vt))
                self.tf_wt = tf.get_variable(name="wt", shape=wt.shape,
                                             initializer=tf.constant_initializer(wt))
                self.tf_cur_time = tf.placeholder(name="t", shape=1, dtype=tf.float32)
                self.u_t = tf.exp(
                    self.tf_vt * self.tf_h +
                    self.tf_cur_time * self.tf_wt +
                    self.tf_bt,
                    name="u_t"
                )


        sim_feed_dict = {
            self.tf_Wm: Wm,
            self.tf_Wh: Wh,
            self.tf_Wt: Wt,
            self.tf_Bh: Bh,

            self.tf_bt: bt,
            self.tf_vt: vt,
            self.tf_wt: wt,
        }

        self.sim_opts = sim_opts
        self.src_id = sim_opts.src_id
        self.sess = sess

    def initialize(self):
        """Initialize the graph."""
        self.sess.run(tf.global_variables_initializer())
        # No more nodes will be added to the graph beyond this point.
        # Recommended way to prevent memory leaks afterwards:
        # https://stackoverflow.com/questions/38694111/
        self.sess.graph.finalize()

    def _create_exp_broadcaster(self, seed):
        """Create a new exp_broadcaster with the current params."""
        return ExpBroadcaster(src_id=self.src_id, seed=seed, trainer=self)

    def run_sim(self, seed):
        """Run one simulation and return the dataframe.
        Will be thread-safe and can be called multiple times."""
        run_sim_opts = self.sim_opts.update({})
        exp_b = self._create_exp_broadcaster(seed=seed * 3)

        mgr = run_sim_opts.create_manager_with_broadcaster(exp_b)
        mgr.run_dynamic()
        return mgr.get_state().get_dataframe()
