import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from classes_games import GameOfLife
import numpy as np
import pickle
import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# define a function to save the models
def save_model(model, name):
    with open(f"Models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# define a function to load the models
def load_model(name):
    with open(f"Models/{name}.pkl", "rb") as f:
        return pickle.load(f)

# function to save an image in the subfolder "Presentation"
def save_img(name):
    plt.savefig(f"Images/{name}.png") 
    
# naive plot of a tree
def plot_tree(node, ax, x, y, dx, dy):
    ax.axis("off")
    ax.text(x, y, node.value, ha="center", va="center", fontsize=20)
    if node.left:
        ax.plot([x-dx, x], [y-dy, y], c="k")
        plot_tree(node.left, ax, x-dx, y-dy, dx/2, dy)
    if node.right:
        ax.plot([x+dx, x], [y-dy, y], c="k")
        plot_tree(node.right, ax, x+dx, y-dy, dx/2, dy)

# function to make animations
def std_anim(game, figsize=(7, 3.5), autolayout=True, title=True, frames=100, interval=60):
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.autolayout"] = autolayout

    fig, ax = plt.subplots()
    
    # delete axis
    ax.axis("off")
    def update(i):
        ax.imshow(game.state)
        if title:
            ax.set_title(f"Step {i}")
        game.step()
        return [ax]

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    return HTML(anim.to_html5_video())

'''
this function takes a game of life and returns 2 numpy arrays
the first one contains the state of the game and the number of neighbors
the second one contains the state of the game after one step
'''
def set_IO(game:GameOfLife, unique=True):
    # s = np.array([game.state[i, j] for i in range(1, game.state.shape[0]-1) for j in range(1, game.state.shape[1]-1)], dtype=np.int8)
    s = game.state.flatten().numpy().astype(np.int8)
    ngbs = game.neighbors.flatten().numpy().astype(np.int8)
    s = np.concatenate((s.reshape(-1, 1), ngbs.reshape(-1, 1)), axis=1)
    game.step()
    y = game.state.flatten().numpy().astype(np.int8)
    if unique:
        unique_idx_train = np.unique(s, axis=0, return_index=True)[1]
        s = s[unique_idx_train]
        y = y[unique_idx_train]
        
    return s, y

# create the animation
def get_animation(t_games:tuple, fig, t_ax:tuple, interval=100, frames=100, metric=None):
    # check if the number of games and axes are the same
    def checktuple(t_games, t_ax, metric=metric):
        if len(t_games) != len(t_ax) and metric is None:
            raise ValueError("The number of games and axes must be the same")
        elif len(t_games)+1 != len(t_ax) and metric is not None:
            raise ValueError("The number of games + 1  and axes must be the same")
    
    # configure the subplots
    def set_config(game, ax, i):
            ax.clear()
            ax.set_xticks(np.arange(-.5, game.M-0.5, 1))
            ax.set_yticks(np.arange(-.5, game.N-0.5, 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlim(-0.5, game.M-0.5)
            ax.set_ylim(-0.5, game.N-0.5)
            ax.imshow(game.state, cmap="PuBu", alpha=0.7)
            ax.set_title(f"{game.name}: {i}")
            ax.grid(True, which='both', color='lightgrey', linewidth=1.5)
            
    def set_metric(metric, ax, frames=frames, i=0):
        if i == 0:
            ax.clear()
            ax.set_title(f"Metric")
            ax.set_xlabel("Step")
            ax.set_ylabel("Metric")
            ax.set_xlim(0, frames)
            ax.set_ylim(-0.5, 1.5)
        
        ax.scatter(i, metric, c="k")

    # animate the games
    def animate(i):
        if i == 0:
            checktuple(t_games, t_ax)
            for k in range(len(t_games)):
                set_config(t_games[k], t_ax[k], i)

            if metric is not None:
                set_metric(metric(t_games[0], t_games[1]), t_ax[-1], frames, i)

        for k in range(len(t_games)):
            t_games[k].step()
            set_config(t_games[k], t_ax[k], i)
        
        if metric is not None:
            set_metric(metric(t_games[0], t_games[1]), t_ax[-1], frames, i)

    return animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)

