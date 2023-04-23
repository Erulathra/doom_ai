# Doom AI

Welcome to my repository where I attempt to create an AI that can play the classic game
Doom using Python, PPO algorithm, Stable Baselines, VizDoom, and TensorFlow.

This project is created to help me familiarize with reinforcement learning
and prepare for writing my Engineering Thesis at the Łódź University of Technology.

### **WARNING**

Please note that all information presented in this readme should be treated as notes,
and I do not guarantee their accuracy or completeness.

## Environment

To start learning start ```TrainModel```

To test model start ```TestModel```

It's recommended to supervise learning output by TensorBoard:

```bash
tensorboard --logdir=<folder_with_logs>
```

and go to http://localhost:6006/

## Learning Output

* **ep_len_mean** - average of agent lifetime
* **ep_len_mean** - average of sum of rewards


* **fps** - how fast game is running


* **approx_kl** - difference between old and new agent. If its spikes massively,
  learning is unstable. It should be varying but not too big
* **clip** - if change between agents is big, PPO clips that change
    * **clip_fraction** - how many times PPO needs to clip value / all learning steps,
      for example 0.21 means that PPO needs to clip learning in 21% of all steps
    * **clip_range** - range of clip, set by scientist, how much PPO clips the learning value
* **entropy_lose** - how actor actions become less random (it should go towards 0)
* **explained_variance** - how well critic network can explain variance. (It should go upward)
  (Ideally it should be positive and going upward, because it means that critic model can
  predict reward)
* **learning_rate** - parameter set by scientist
* **loss** - current loss of reward
* **n_updates** - number of updates to networks so far
* **policy_gradient_loss** - (SUPER IMPORTANT) how well is agent is able to take actions to
  get high reward (It should decrease ( because it means that reward loss is decreasing))
* **value_loss** - how well is agent able to predict future based of its actions
  (It should decrease)

## Learning Parameters

* **learning_rate** - parameter set by scientist
* **clip_range** - range of clip, set by scientist, how much PPO clips the learning value
* **gae_lambda** - ???
* **n_steps** - ???

## Tips

* If approx_kl is too chaotic, you can increase clip_range and gae_lambda
* policy_gradient_loss is fluctuating in start but with time its going down

# Licence

Copyright (c) 2023 Szymon Świędrych

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.