from psyneulink import *
from gym_forager.envs.forager_env import ForagerEnv
from gym_forager.envs.forager.randomexp import RandomAgent


num_trials = 4
env = ForagerEnv()
reward = 0
done = False


# Function used by PsyNeuLink Mechanism
r = RandomAgent(env.action_space)
def random_action(variable):
    return r.act(variable, None, None)

# Create PsyNeuLink Composition as agent
agent_mech = ProcessingMechanism(function=random_action)
agent_comp = Composition()

agent_comp.add_node(agent_mech)

def main():
    for _ in range(num_trials):
        observation = env.reset()
        while True:
            run_results = agent_comp.run(inputs={agent_mech:observation})
            action=run_results[0]
            observation, reward, done, _ = env.step(action)
            if done:
                break

if __name__ == "__main__":
    main()
