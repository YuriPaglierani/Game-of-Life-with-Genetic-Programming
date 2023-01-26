import torch
import numpy as np
import gplearn.genetic as gp

class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def __repr__(self):
        return f"{self.value}"
    
    def __str__(self):
        return f"{self.value}"
    
    def add_left(self, node):
        self.left = node
        return self

    def add_right(self, node):
        self.right = node
        return self

class GameOfLife():
    def __init__(self, state:torch.tensor=None, M=None, N=None):
        if type(state)==torch.tensor:
            self.state = state.clone()
            self.M, self.N = state.shape
        elif M is not None and N is not None:
            self.M = M
            self.N = N
            self.state = torch.distributions.Bernoulli(0.3).sample((M, N)).int().requires_grad_(False)    
        else:
            raise ValueError("Either state or M and N must be provided")

        self.kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool).requires_grad_(False)
        self.neighbors = None
        self.name = "GameOfLife"
        self._neighbors()

    def _neighbors(self):
        self.neighbors = torch.nn.functional.conv2d(self.state.unsqueeze(0).unsqueeze(0).int(), 
                                                    self.kernel.unsqueeze(0).unsqueeze(0).int(), padding=1).squeeze(0).squeeze(0).requires_grad_(False)

    def step(self):
        self._neighbors()
        self.state = (self.neighbors == 3) | (self.neighbors == 2) & self.state

class FasterGameOfLife():
    def __init__(self, state:torch.tensor=None, M=None, N=None):
        if state is not None:
            self.state = state
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
            self.M, self.N = self.state.shape
        elif M and N:
            self.M = M
            self.N = N
            self.state = torch.distributions.Bernoulli(0.3).sample((M, N)).int().requires_grad_(False) 
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
        else:
            raise ValueError("Either state or M and N must be given")
        
        self.neighbors = None
        self.name = "GameOfLife"
        self._neighbors()

    def _neighbors(self):
        window_rows, window_cols = (3, 3)

        # use unfold to create a view of the padded state
        window_view = self.padded_state.unfold(0, window_rows, 1).unfold(1, window_cols, 1).reshape(-1, window_rows*window_cols)
        self.neighbors = window_view.sum(dim=1).reshape(self.M, self.N).requires_grad_(False) - self.state

    def step(self):
        self._neighbors()
        self.state = (self.neighbors == 3) | (self.neighbors == 2) & self.state
        self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)

# create the class FakeGameOfLife
class FakeGameOfLife:
    def __init__(self, state=None, M=None, N=None, rule:gp.SymbolicClassifier=None):
        if state is not None:
            self.state = state.clone()
            self.M, self.N = state.shape
        elif M and N:
            self.M = M
            self.N = N
            self.state = torch.distributions.Bernoulli(0.3).sample((M, N)).int().requires_grad_(False) 
        else:
            raise ValueError("Either state or M and N must be given")
        self.neighbors = None
        self.kernel = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.bool).requires_grad_(False)
        self.rule = rule
        self.name = "FakeGameOfLife"
        if rule is None:
            raise ValueError("Rule must be given")

    def _neighbors(self):
        self.neighbors = torch.nn.functional.conv2d(self.state.unsqueeze(0).unsqueeze(0).int(), 
                                                    self.kernel.unsqueeze(0).unsqueeze(0).int(), padding=1).squeeze(0).squeeze(0).requires_grad_(False)

    def step(self):
        self._neighbors()
        x = torch.cat((self.state.view(-1, 1), self.neighbors.view(-1, 1)), dim=1)
        y = self.rule.predict(x)
        self.state = torch.tensor(y).view(self.M, self.N).requires_grad_(False)

# create the class FakeGameOfLife
class FakeGameOfLife9Var:
    def __init__(self, state:torch.tensor=None, M=None, N=None, rule:gp.SymbolicClassifier=None):
        if state is not None:
            self.state = state
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
            self.M, self.N = self.state.shape
        elif M and N:
            self.M = M
            self.N = N
            self.state = torch.distributions.Bernoulli(0.3).sample((M, N)).int().requires_grad_(False) 
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
        else:
            raise ValueError("Either state or M and N must be given")
        self.rule = rule
        self.name = "FakeGameOfLife9Var"
        if rule is None:
            raise ValueError("Evolution rule must be given")

    def step(self):
        window_rows, window_cols = (3, 3)

        # use unfold to create a view of the padded state
        window_view = self.padded_state.unfold(0, window_rows, 1).unfold(1, window_cols, 1).reshape(-1, window_rows*window_cols)
        self.state = torch.tensor(self.rule.predict(window_view)).view(self.M, self.N)
        self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)

# create the class FakeGameOfLife
class FakeGameOfLife9VarFeature:
    def __init__(self, state:torch.tensor=None, M=None, N=None, featurizer:gp.SymbolicTransformer=None, rule:gp.SymbolicClassifier=None):
        if state is not None:
            self.state = state
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
            self.M, self.N = self.state.shape
        elif M and N:
            self.M = M
            self.N = N
            self.state = torch.distributions.Bernoulli(0.3).sample((M, N)).int().requires_grad_(False) 
            self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)
        else:
            raise ValueError("Either state or M and N must be given")
        self.rule = rule
        self.name = "FakeGameOfLife9VarFeature"
        self.featurizer = featurizer
        if rule is None:
            raise ValueError("Evolution rule must be given")

    def step(self):
        window_rows, window_cols = (3, 3)

        # use unfold to create a view of the padded state
        window_view = self.padded_state.unfold(0, window_rows, 1).unfold(1, window_cols, 1).reshape(-1, window_rows*window_cols)
        if self.featurizer is not None:
            window_view = self.featurizer.transform(window_view)
        self.state = torch.tensor(self.rule.predict(window_view)).view(self.M, self.N)
        self.padded_state = torch.nn.functional.pad(self.state, (1, 1, 1, 1), mode='constant', value=0).requires_grad_(False)