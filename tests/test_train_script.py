import gin
import gin.tf

from imitation.scripts.train import train_and_plot


gin.parse_config_file("configs/classical_control.gin")
gin.bind_parameter('train_and_plot.env', 'CartPole-v1')
gin.bind_parameter('init_trainer.use_gail', False)  # False = use AIRL


def test_train_and_plot_no_crash():
    train_and_plot(n_epochs=2,
                   n_epochs_per_plot=1,
                   n_disc_steps_per_epoch=1,
                   n_gen_steps_per_epoch=1,
                   n_episodes_per_reward_data=2,
                   interactive=False)
    tf.reset_default_graph()
