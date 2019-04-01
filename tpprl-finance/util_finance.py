import datetime as D
import tensorflow as tf


def _now(raw=False):
    """Return the time now in red color."""
    templ = '\x1b[31m[{}]\x1b[0m' if not raw else '{}'
    return templ.format(D.datetime.now().isoformat(timespec='seconds'))


def variable_summaries(var, name=None):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    if name is None:
        name = var.name.split('/')[-1][:-2]

    with tf.name_scope('summaries-' + name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
