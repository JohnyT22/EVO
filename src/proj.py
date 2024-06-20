# proj.py
# EVO projekt, 12.03.2024
# Autor: Jan Tomeček, FIT
# Založeno na tiny_gp_plus.py https://github.com/moshesipper/tiny_gp/tree/master
# 

# 30 behu kazde nastaveni
# min 2 pole
# min 2 nastaveni bloatu


from random import random, randint, seed
from statistics import mean
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import Image
from graphviz import Digraph 
import pickle
import pandas as pd
import os

import argparse
import pathlib

POP_SIZE        = 200    # population size
MIN_DEPTH       = 2      # minimal initial random tree depth
MAX_DEPTH       = 5      # maximal initial random tree depth
GENERATIONS     = 1000   # maximal number of generations to run evolution
TOURNAMENT_SIZE = 3      # size of tournament for tournament selection
XO_RATE         = 0.8    # crossover rate 
PROB_MUTATION   = 0.2    # per-node mutation probability 
BLOAT_CONTROL   = True   # True adds bloat control to fitness function
BLOAT_PENALTY   = 0.0001  # penalty for each extra node in a tree
ANT_START_ENERGY     = 20
ANT_FOOD_ENERGY_GAIN = 5
BEST_PARENTS_TO_SURVIVE = int(0.05*POP_SIZE) # best 5% of population survives

SHOW_WINDOW = True

OUTPUT_DIR = "output"
RUN_EXPERIMENT = 1


MAZE = [
    [0,1,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0],
    [0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,1,1,1,0,0],
    [0,0,0,0,0,1,1,1]]


rotationRigthMartix = [
    [0,-1],
    [1,0]
]
rotationLeftMatrix = [
    [0,1],
    [-1,0]
]

def mulMartixAndVector(martix, vector):
    return (martix[0][0]*vector[0] + martix[0][1]*vector[1], martix[1][0]*vector[0] + martix[1][1]*vector[1])
class Ant:
    def __init__(self):
        self.position = (0,0)
        self.direction = (1,0)
        self.energy = ANT_START_ENERGY
        self.path = [self.position]
        self.foodEaten = 0
    def restart(self):
        self.position = (0,0)
        self.direction = (1,0)
        self.energy = ANT_START_ENERGY
        self.path = [self.position]
        self.foodEaten = 0
    def move(self, maze):
        self.energy -= 1
        newPosX = self.position[0]
        newPosY = self.position[1]
        if(self.position[0] + self.direction[0] >=0 and self.position[0] + self.direction[0] < len(maze[0])):
            newPosX = self.position[0] + self.direction[0]
        if(self.position[1] + self.direction[1] >=0 and self.position[1] + self.direction[1] < len(maze)):
            newPosY = self.position[1] + self.direction[1]
        self.position = (newPosX, newPosY)
        if(self.position not in self.path): # potrava nebyla dosud snězena
            if maze[self.position[1]][self.position[0]] == 1:
                self.energy += ANT_FOOD_ENERGY_GAIN
                self.foodEaten += 1
        if(self.position not in [self.path[-1]]):
            self.path.append(self.position)  
    def turn(self, direction):
        self.energy -= 1
        if direction == "Left":
            self.direction = mulMartixAndVector(rotationLeftMatrix, self.direction)
        elif direction == "Right":
            self.direction = mulMartixAndVector(rotationRigthMartix, self.direction)
    
# end class Ant
            
def IfFoodAhead(ant: Ant, maze, x,y,_): 
    indexY = -1
    indexX = -1
    if ant.position[1]+ant.direction[1] < len(maze) and ant.position[1]+ant.direction[1] >= 0:
        indexX = ant.position[1]+ant.direction[1]
    if ant.position[0]+ant.direction[0] < len(maze) and ant.position[0]+ant.direction[0] >= 0:
        indexY = ant.position[0]+ant.direction[0]
    if indexY == -1 or indexX == -1: return [y]
    return [x] if (maze[indexY][indexX]==1 and (indexX, indexY) not in ant.path) else [y]
def Prog2(ant,maze,x,y,_): return [x,y]
def Prog3(ant,maze,x,y,z): return [x,y,z]
FUNCTIONS = [IfFoodAhead, Prog2, Prog3]
TERMINALS = ["Left", "Move", "Right"]


class DataSet():
    def __init__(self,maze, maxfood):
        self.maxfood = maxfood 
        self.maze = maze

def loadMazeFromFile(path=None) -> list:
    if not path:
        return MAZE # default value
    maze = []
    with open (path, 'rb') as fp:
        maze = pickle.load(fp)
    if maze == []:
        maze = MAZE # default value
    return maze

def generate_dataset(path_to_file=None) -> DataSet:
    maze = MAZE
    if(path_to_file) : maze = loadMazeFromFile(path_to_file)
    dataset = DataSet(maze, sum([sum(x,0) for x in maze],0))
    return dataset

class GPTree:
    def __init__(self, data = None, left = None, middle = None, right = None):
        self.data  = data
        self.left  = left
        self.middle = middle
        self.right = right
        
    def node_label(self): # return string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def draw(self, dot, count): # dot & count are lists in order to pass "by reference" 
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.middle:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.middle.draw(dot, count) 
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)
        
    def draw_tree(self, fname, footer, axarr):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        path = dot[0].render(filename=fname + ".gv", format="png")
        img = Image(filename = fname + ".gv.png")
        img = mpimg.imread(path)
        axarr[2].imshow(img)
        axarr[2].axis('off')
        if(SHOW_WINDOW):
            plt.draw()  
            plt.pause(0.01)
        #img = Image(filename = fname + ".gv.png")
        #display(img)

    def compute_tree(self,ant: Ant, dataset: DataSet): 
        while(ant.energy > 0 and ant.foodEaten < dataset.maxfood):
            self.interpreteTree(ant,dataset)
    
    def interpreteTree(self,ant: Ant, dataset: DataSet):
        if ant.energy <= 0 or self == None or ant.foodEaten >= dataset.maxfood:
            return
        if (self.data in FUNCTIONS): 
            nx = self.data(ant, dataset.maze, self.left, self.middle, self.right)
            for i in nx:
                i.interpreteTree(ant,dataset)
        elif (self.data in TERMINALS):
            if(self.data == "Move"):
                ant.move(dataset.maze)
            else:
                ant.turn(self.data)
        return
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.middle = GPTree()
            self.middle.random_tree(grow, max_depth, depth = depth + 1)
            self.right = None
            if self.data == Prog3:
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        if self == None: return
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.middle: self.middle.mutation()
        elif self.right: self.right.mutation() 
        
#    def depth(self):     
#        if self.data in TERMINALS: return 0
#        l = self.left.depth()  if self.left  else 0
#        r = self.right.depth() if self.right else 0
#        return 1 + max(l, r)

    def size(self): # tree size in nodes
        if self.data in TERMINALS: return 1
        l = self.left.size()  if self.left  else 0
        m = self.middle.size() if self.middle else 0
        r = self.right.size() if self.right else 0
        return 1 + l + m + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.middle: t.middle = self.middle.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.middle = second.middle
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second) 
            if self.middle and count[0] > 1: ret = self.middle.scan_tree(count, second) 
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree
# end class GPTree

def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append((t, Ant()))
        for i in range(int(POP_SIZE/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append((t, Ant())) 
    return pop

def error(individual, dataset: DataSet):
    individual[1].restart()
    individual[0].compute_tree(individual[1], dataset)
    return (dataset.maxfood - individual[1].foodEaten)/dataset.maxfood

def fitness(individual, dataset): 
    if BLOAT_CONTROL:
        return 1 / (1 + error(individual, dataset) + BLOAT_PENALTY*individual[0].size())
    else:
        return 1 / (1 + error(individual, dataset))
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])  

def prepare_plots():
    fig = plt.figure(figsize=(16,9), num = 'EVOLUTIONARY PROGRESS')
    plt.clf()
    gs = fig.add_gridspec(4,2)
    axarr = [None, None, None, None]
    axarr[0] = fig.add_subplot(gs[0,0])
    axarr[1] = fig.add_subplot(gs[1,0])
    axarr[2] = fig.add_subplot(gs[:,1])
    axarr[3] = fig.add_subplot(gs[2:,0])
    axarr[0].sharex(axarr[1])
    #fig, axarr = plt.subplots(2, sharex=True)
    #fig.canvas.setWindowTitle('EVOLUTIONARY PROGRESS')
    fig.subplots_adjust(hspace = 0.5)
    axarr[0].set_title('error', fontsize=14)
    axarr[1].set_title('mean size', fontsize=14)
    axarr[1].set_xlabel('generation', fontsize=18)
    plt.ion() # interactive mode for plot
    axarr[1].set_xlim(0, GENERATIONS)
    axarr[0].set_ylim(0, 1) # fitness range
    xdata = []
    ydata = [ [], [] ]
    line = [None, None]
    line[0], = axarr[0].plot(xdata, ydata[0], 'b-') # 'b-' = blue line    
    line[1], = axarr[1].plot(xdata, ydata[1], 'r-') # 'r-' = red line
    
    return axarr, line, xdata, ydata

def plot(axarr, line, xdata, ydata, gen, pop, errors, max_mean_size):
    xdata.append(gen)
    ydata[0].append(min(errors))
    line[0].set_xdata(xdata)
    line[0].set_ydata(ydata[0])
    sizes = [ind[0].size() for ind in pop]
    if mean(sizes) > max_mean_size[0]:
        max_mean_size[0] = mean(sizes)
        axarr[1].set_ylim(0, max_mean_size[0])
    ydata[1].append(mean(sizes))
    line[1].set_xdata(xdata)
    line[1].set_ydata(ydata[1])
    if(SHOW_WINDOW):
        plt.draw()  
        plt.pause(0.01)

#zdroj https://medium.com/@msgold/using-python-to-create-and-solve-mazes-672285723c96
# poupraveno
def draw_maze(ax, maze, path=None):
    #fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the border color to white
    #fig.patch.set_edgecolor('white')
    #fig.patch.set_linewidth(0)
    ax.clear()
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    # Major ticks
    ax.set_xticks([x for x in range(len(maze[0]))])
    ax.set_yticks([y for y in range(len(maze))])

    # Labels for major ticks
    ax.set_xticklabels(['' for x in range(len(maze[0]))])
    ax.set_yticklabels(['' for y in range(len(maze))])

    # Minor ticks
    ax.set_xticks([x-0.5 for x in range(len(maze[0]))], minor=True)
    ax.set_yticks([y-0.5 for y in range(len(maze))], minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=2)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)
    
    # Draw the solution path if it exists
    if path is not None:
        x_coords = [x[0] for x in path]
        y_coords = [y[1] for y in path]
        ax.plot(x_coords, y_coords, color='red', linewidth=1)
        
        for i in range(len(path)-1):
            dx = (x_coords[i+1] - x_coords[i]) * 0.2
            dy = (y_coords[i+1] - y_coords[i]) * 0.2
            ax.arrow(x_coords[i], y_coords[i], dx, dy, color='red', head_width=0.15, head_length=0.1)
        
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw entry and exit arrows
    ax.arrow(-0.2, 0, .1, 0, fc='green', ec='green', head_width=0.1, head_length=0.1)
    #ax.arrow(len(maze), len(maze[0])  - 1, 0.1, 0, fc='blue', ec='blue', head_width=0.1, head_length=0.1)


def main(path_to_data = None, runNumber = 0):      
    # init stuff
    seed() # init internal state of random number generator
    dataset = generate_dataset(path_to_data)
    population= init_population() 
    best_of_run = [GPTree(),Ant()]
    best_of_run_error = 1e20 
    best_of_run_gen = 0
    fitnesses = [fitness(ind, dataset) for ind in population]
    max_mean_size = [0] # track maximal mean size for plotting
    axarr, line, xdata, ydata = prepare_plots()
    draw_maze(axarr[3], dataset.maze)
    
    errors_through_generations = []
    # go evolution!
    for gen in range(GENERATIONS):        
        nextgen_population=[]
        for i in range(POP_SIZE-BEST_PARENTS_TO_SURVIVE):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1[0].crossover(parent2[0])
            parent1[0].mutation()
            nextgen_population.append(parent1)
        best_parents = sorted(population, key=lambda ind: fitness(ind, dataset), reverse=True)[:BEST_PARENTS_TO_SURVIVE]
        nextgen_population.extend(best_parents)
        population=nextgen_population
        fitnesses = [fitness(ind, dataset) for ind in population]
        errors = [error(ind, dataset) for ind in population]
        errors_through_generations.append(min(errors))
        if min(errors) < best_of_run_error:
            best_of_run_error = min(errors)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[errors.index(min(errors))])
            print(f"Gen: {gen:5d}, error: {best_of_run_error:.2f}, size: {best_of_run[0].size()}")
            best_of_run[0].draw_tree("best_of_run",\
                                "gen: " + str(gen) + ", error: " + str(round(best_of_run_error,3)), axarr)
            draw_maze(axarr[3], dataset.maze, best_of_run[1].path)
        plot(axarr, line, xdata, ydata, gen, population, errors, max_mean_size)
        if best_of_run_error <= 1e-5: break
    
    endrun = f"_________________________________________________\nEND OF RUN {runNumber} (bloat control was "
    endrun += "ON)" if BLOAT_CONTROL else "OFF)"
    print(endrun)
    s = "\n\nbest_of_run attained at gen " + str(best_of_run_gen) + " and has error=" + str(round(best_of_run_error,3))
    best_of_run[0].draw_tree("best_of_run",s, axarr)
    draw_maze(axarr[3], dataset.maze, best_of_run[1].path)
    if(SHOW_WINDOW):
        plt.show(block=True)
    else:
        if (not os.path.exists(f"./{OUTPUT_DIR}")):
            os.makedirs(f"./{OUTPUT_DIR}")
        plt.savefig(f"./{OUTPUT_DIR}/run_{runNumber}_PS{POP_SIZE}_MID{MIN_DEPTH}_MAD{MAX_DEPTH}_G{GENERATIONS}_TS{TOURNAMENT_SIZE}_COR{XO_RATE:.2f}_PM{PROB_MUTATION:.2f}_BC{BLOAT_CONTROL}_BP{BLOAT_PENALTY:.5f}.png", bbox_inches="tight", dpi = 600)
    return best_of_run_gen, best_of_run_error, best_of_run[0].size(), best_of_run[1].foodEaten, best_of_run[1].path, best_of_run[1].energy, errors_through_generations
    
if __name__== "__main__":
    parser = argparse.ArgumentParser(description='EVO Project \n Genetic Programing - Ant control program')
    parser.add_argument('filename',type=pathlib.Path, help='path to pickled input data')
    args = parser.parse_args()
    df = pd.DataFrame(columns=['runNumber','best_of_run_gen','best_of_run_error','size','foodEaten','path','energy', "errors"])
    for run in range(RUN_EXPERIMENT):
        best_of_run_gen, best_of_run_error, size, foodEaten, path, energy, errors_through_generations = main(args.filename, run)
        df.loc[run] = [run,best_of_run_gen, best_of_run_error, size, foodEaten, path, energy, errors_through_generations]
    df.to_csv(f"./{OUTPUT_DIR}/results.csv", index=False)