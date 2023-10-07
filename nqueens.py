import random
import matplotlib.pyplot as plt
import numpy as np
import checkerboard
import time


def main():
    m = 100  # size of population
    q = 8  # number of genes (i.e. queens)
    n = 100  # number of offspring
    k = 5  # size of tournament for parent selection
    pc = 0.8  # probability of crossover
    pm = 0.1  # probability of mutation
    maxiter = 10000  # maximum number of iterations
    pause = 0.01  # amount of time to pause between generations
    size = 800//q  # size of squares on board
    margin = 10  # size of margin on gui for graphics

    gui = checkerboard.draw(rows=q, cols=q, size=size, margin=margin)

    population = initialize(m, q)
    queens = checkerboard.add(gui, population[0, :], size=size, margin=margin)
    average = np.full(maxiter, np.nan)
    best = np.full(maxiter, np.nan)
    for i in range(maxiter):
        parents = select_parents(population, n, k)
        children = crossover(parents, pc)
        mutants = mutation(children, pm)
        population, score = select_survivors(population, mutants)
        average[i] = score.mean()
        best[i] = score[0]
        print('generation:{0} average:{1:0.2f} best:{2:0.2f}'.format(i + 1, average[i], best[i]))

        if i >= 2 and best[i] < best[i - 1]:  # only if the best solution has improved
            checkerboard.move(gui, queens, population[0, :], size=size, margin=margin)
            time.sleep(pause)

        plt.figure(1)
        plt.imshow(population, cmap='Set3', extent=[0, 1, 0, 1])
        plt.title('Population')
        plt.show(block=False)
        if i == 0:
            input('Waiting until you press Enter...')

        if score[0] == 0:
            print("SUCCESS!")
            print(*population[0, :])
            break

    plt.figure(2)
    plt.plot(np.arange(1, maxiter + 1), best, label='best')
    plt.plot(np.arange(1, maxiter + 1), average, label='average')
    plt.legend()
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show(block=False)

    while True:
        key = gui.checkKey()
        if key == 'Escape':
            break  # exit the loop when 'Esc' is pressed
    gui.close()


def initialize(m, q):
    population = np.random.randint(0, q, size=(m, q))
    score = fitness(population)
    order = np.argsort(score)
    population = population[order, :]
    return population


def fitness(population):
    cols = np.sum(np.diff(np.sort(population, axis=1)) == 0, axis=1)
    diags = np.zeros_like(cols)

    for i in range(1, population.shape[1]):
        potential = abs(population - np.roll(population, i)) == i
        actual = np.sum(potential[:, i:], axis=1)
        diags += actual

    return cols + diags


def select_parents(population, n, k):
    players = np.random.randint(population.shape[0], size=(n, k))
    winners = players.min(axis=1)
    return population[winners, :]


def crossover(parents, pc):
    parent1 = parents[::2, :]
    parent2 = parents[1::2, :]

    mask = np.random.rand(*parent1.shape) < 0.5
    nocrossover = np.random.rand(parent1.shape[0]) > pc
    mask[nocrossover, :] = False

    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    children = np.r_[child1, child2]

    return children


def mutation(children, pm):
    """Perform random resetting on children with probability pm."""
    newvalues = np.random.randint(0, children.shape[1], size=children.shape)
    mask = np.random.rand(*children.shape) <= pm
    children[mask] = newvalues[mask]

    return children


def select_survivors(population, mutants):
    candidates = mutants
    score = fitness(candidates)
    order = np.argsort(score)
    candidates = candidates[order, :]
    score = score[order]

    survivors = candidates[:population.shape[0], :]
    score = score[:population.shape[0]]
    return survivors, score


if __name__ == "__main__":
    main()
