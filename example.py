import sys

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

tf.config.set_visible_devices([], 'GPU')

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "error"

    if case == "locallevel":
        model = tfp.sts.LocalLevel(name="ll")
        @tfd.JointDistributionCoroutine
        def joint_distribution():
            yield tfd.JointDistribution.Root(model.make_state_space_model(1, [1.]))
        print(joint_distribution.log_prob([[[0.]]]))
    elif case == "sequential":
        model = tfp.sts.AutoregressiveIntegratedMovingAverage(0, 0, 0, name="arima")
        joint_distribution = tfd.JointDistributionSequential([
            model.make_state_space_model(1, [1.]),
        ])
        print(joint_distribution.log_prob([[[0.]]]))
    elif case == "named":
        model = tfp.sts.AutoregressiveIntegratedMovingAverage(0, 0, 0, name="arima")
        joint_distribution = tfd.JointDistributionNamed(dict(
            e=model.make_state_space_model(1, [1.]),
            x=lambda e: tfd.Normal(0., 1.),
        ))
        print(joint_distribution.log_prob({"e": [[0.]], "x": [[0.]]}))
    else:
        model = tfp.sts.AutoregressiveIntegratedMovingAverage(0, 0, 0, name="arima")
        @tfd.JointDistributionCoroutine
        def joint_distribution():
            yield tfd.JointDistribution.Root(model.make_state_space_model(1, [1.]))
        print(joint_distribution.log_prob([[[0.]]]))
