
import tensorflow as tf


class StopHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""



    def begin(self):
        self.loss = tf.losses.get_losses()
        self._step = -1

    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self.loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):


            loss_value = run_values.results[0]
            if loss_value < self._step * 0.00001 :
                run_context.request_stop()
