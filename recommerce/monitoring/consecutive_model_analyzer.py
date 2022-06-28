from recommerce.monitoring.agent_monitoring.am_evaluation import Evaluator


def analyze_consecutive_models(saved_parameter_paths, monitor, marketplace_class, config_market, agent_class, continuous_action_space):
    agent_list = [(agent_class, [parameter_path]) for parameter_path in saved_parameter_paths]
    monitor.configurator.setup_monitoring(
        episodes=25,  # This is for performance reasons. Switch back to 100 if you want more details.
        plot_interval=25,
        marketplace=marketplace_class,
        agents=agent_list,
        separate_markets=True,
        support_continuous_action_space=continuous_action_space,
        config_market=config_market)
    rewards = monitor.run_marketplace()
    episode_numbers = [int(parameter_path[-9:][:5]) for parameter_path in saved_parameter_paths]
    Evaluator(monitor.configurator).evaluate_session(rewards, episode_numbers)
