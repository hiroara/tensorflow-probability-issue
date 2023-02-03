import sys

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

tf.config.set_visible_devices([], 'GPU')

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "error"

    if case == "successful":
        components = [tfp.sts.LocalLevel(name="ll")]
        components_params = {"ll/_level_scale": 1e-1}
    else:
        components = [tfp.sts.Autoregressive(order=1, name="ar")]
        components_params = {"ar/_coefficients": [1.], "ar/_level_scale": [1.]}

    s = tfp.sts.Sum(components, name="sum")

    ssm = s.make_state_space_model(
        num_timesteps=100,
        param_vals={
            "observation_noise_scale": 1e-1,
            **components_params,
        },
        name="ssm"
    )
    print(f"{ssm.name} is built successfully")
