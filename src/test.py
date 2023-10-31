import subprocess as sub
import numpy as np
import matplotlib.pyplot as plt


def plot_results(numbers, bins):
    plt.hist(numbers, bins=bins, edgecolor='k', alpha=0.65)
    plt.show()


def generate_random_normal_number():
   mu    = 0
   sigma = 1.3

   random_number = np.random.normal(mu, sigma)
   # rount to integers
   number = int(random_number)
   return number


def main():
    random_numbers = []
    indices        = []
    for i in range(1000):
        rand_number = generate_random_normal_number()
        random_numbers.append(int(rand_number))
        indices.append(i)

    plot_results(random_numbers, 30)

main()
