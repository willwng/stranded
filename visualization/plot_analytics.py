import csv
import numpy as np
import matplotlib.pyplot as plt


def main():
    analytics = []
    with open('../output/analytics.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            analytics.append([float(x) for x in row])

    analytics = np.array(analytics)
    t = np.arange(analytics.shape[0])
    total_energy = analytics[:, 4]
    kinetic_energy = analytics[:, 5]
    potential_energy = analytics[:, 6]

    plt.plot(t, total_energy, label='Total Energy')
    plt.plot(t, kinetic_energy, label='Kinetic Energy')
    plt.plot(t, potential_energy, label='Potential Energy')
    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    main()
