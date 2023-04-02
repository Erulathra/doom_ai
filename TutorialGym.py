from VizDoomEnv import VizDoomEnv

from rich import print

from matplotlib import pyplot as plt

def main():
    env = VizDoomEnv()
    test = env.reset()

    print(env.observation_space.shape)
    print(test.shape)

    env.close()


if __name__ == "__main__":
    main()
