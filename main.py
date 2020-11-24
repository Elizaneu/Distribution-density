import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from matplotlib.widgets import Slider, Button


# Parameters
VALMIN = 0.01
VALMAX = 4
VALINIT = 0.5


# Generate and concatenate samples
def generate_data(seed=17):
    rand = np.random.RandomState(seed)
    x = []
    dat = rand.normal(0, 0.3, 1000)
    x = np.concatenate((x, dat))
    dat = rand.normal(3, 1, 1000)
    x = np.concatenate((x, dat))
    return x


# Histogram and scatter of sample
def scatter_histogram(x):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(np.arange(len(x)), x, c='red')
    plt.xlabel('Sample no.')
    plt.ylabel('Value')
    plt.title('Scatter plot')
    plt.subplot(122)
    hist = plt.hist(x_train, bins=50)
    plt.title('Histogram')
    fig.subplots_adjust(wspace=.3)
    plt.show()
    return hist


# Kernel density estimation
def kde(val):
    model = KernelDensity(kernel='gaussian', bandwidth=val)
    model.fit(x_train)
    log_dens = model.score_samples(x_test)
    return log_dens


# Update plot while using slider
def update(val):
    graph_axes.clear()
    log_dens = kde(val)
    graph_axes.plot(x_test, np.exp(log_dens))
    return


def kde_manual_calc(val):
    x_test = np.linspace(-1, 7, 2000)
    density = sum(np.abs(x_train - x_test) < val)
    return density, x_test


# Update plot using kde_handle function
def update_manual(val):
    graph_axes.clear()
    kde, x_test = kde_manual_calc(val)
    graph_axes.plot(x_test, kde)
    return


# Optimal estimate with Gaussian kernel
def optimal_estimate():
    bandwidth = np.arange(0.05, 2, .05)
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(kde, {'bandwidth': bandwidth})
    grid.fit(x_train)
    kde = grid.best_estimator_
    log_dens = kde.score_samples(x_test)
    plt.fill(x_test, np.exp(log_dens), c='green')
    plt.title('Optimal estimate with Gaussian kernel')
    plt.show()
    print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))


def on_button_clear_clicked(event):
    graph_axes.clear()
    plt.show()


if __name__ == "__main__":
    x_train = generate_data()[:, np.newaxis]
    x_test = np.linspace(-1, 7, 2000)[:, np.newaxis]
    fig, graph_axes = plt.subplots()
    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.4)
    axes_slider_bandwidth = plt.axes([0.18, 0.25, 0.25, 0.075])
    slider_bandwidth = Slider(axes_slider_bandwidth,
                          label='bandwidth',
                          valmin=VALMIN,
                          valmax=VALMAX,
                          valinit=VALINIT,
                          valfmt='%1.2f')

    axes_button_clear = plt.axes([0.06, 0.15, 0.25, 0.075])
    button_clear = Button(axes_button_clear, 'Очистить')
    button_clear.on_clicked(on_button_clear_clicked)

    slider_bandwidth.on_changed(update_manual)
    plt.show()

    optimal_estimate()
