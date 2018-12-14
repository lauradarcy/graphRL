# graphRL

[![PyPI version](https://badge.fury.io/py/graphRL.svg)](https://badge.fury.io/py/graphRL)

This repository contains a PIP package which is an OpenAI environment for simulating directed acyclic graphs for workflow scheduling.


## Installation

this package depends on [graph-tool](https://graph-tool.skewed.de/), which is a python graph manipulation module written in C++. it isn't a pip installable package, so needs to be installed separately before using this package. the only other dependency is openAI's gym module.

once graph-tool is installed, install this package via

```bash
pip3 install graphRL
```

## Usage

```python
import gym
import graphRL

env = gym.make('graphRL-v0')
```


## The Environment

Tbc.