from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3 import SAC


def create_model(env, args):
    print("Creating new SAC model")
    return SAC(MlpPolicy, env, verbose=1, learning_starts=10000, use_sde=True, use_sde_at_warmup=True)


def load_model(model_name, env, args):
    model = SAC.load(model_name, env=env, device="cuda")
    print("Model successfully loaded")
    return model


def get_model(env, args):
    model_to_load = args.load_model

    if model_to_load is not None:
        try:
            return load_model(model_to_load, env, args)
        except FileNotFoundError:
            print(f"Model {model_to_load} not found")

    return create_model(env, args)


def save_model(agent, name, args):
    agent.save(name)
