import numpy as np
import re
# import click
import matplotlib
import sys

matplotlib.use('Agg')
from matplotlib import pylab as plt


# @click.command()
# @click.argument('files', nargs=-1, type=click.Path(exists=True))
def draw_train_loss(files, imsave_path=''):
    """

    :param files:
    :param imsave_path: if this path exists, save the image to it
    :return:
    """
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    # ax2.set_ylabel('accuracy %')
    for i, log_file in enumerate(files):
        loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind = parse_log_eddie(
            log_file)
        disp_results(fig, ax1, ax2, loss_iterations, losses,
                     accuracy_iterations, accuracies,
                     accuracies_iteration_checkpoints_ind, color_ind=i)
    if imsave_path:
        plt.savefig(imsave_path)
    else:
        plt.show()



def parse_log(log_file):
    with open(log_file, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    accuracy_pattern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* accuracy = (?P<accuracy>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    accuracies = []
    accuracy_iterations = []
    accuracies_iteration_checkpoints_ind = []

    for r in re.findall(accuracy_pattern, log):
        iteration = int(r[0])
        accuracy = float(r[1]) * 100

        if iteration % 10000 == 0 and iteration > 0:
            accuracies_iteration_checkpoints_ind.append(
                len(accuracy_iterations))

        accuracy_iterations.append(iteration)
        accuracies.append(accuracy)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)

    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind


def parse_log_eddie(log_file_path):
    # find losses
    with open(log_file_path, 'r') as log_file:
        log = log_file.read()

    loss_pattern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    losses = []
    loss_iterations = []

    for r in re.findall(loss_pattern, log):
        loss_iterations.append(int(r[0]))
        losses.append(float(r[1]))

    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    # find accuracies
    accuracies = []
    accuracy_iterations = []
    outputs = ['dsn1_loss = ', 'dsn2_loss = ', 'dsn3_loss = ', 'dsn4_loss = ',
               'dsn5_loss = ',
               'fuse_loss = ']
    outputs = ['fuse_loss = ']
    with open(log_file_path, 'r') as log_file:
        lines = log_file.readlines()
        for line_num, line in enumerate(lines):
            idx = line.find(', Testing net ')
            if idx != -1:
                # Sample line:
                # I0213 19:28:35.089439  5539 solver.cpp:346] Iteration 0, Testing net (#0)
                str1 = 'Iteration '
                idx1 = line.find(str1) + len(str1)
                accuracy_iterations.append(int(line[idx1:idx]))
                for i in range(len(outputs)):
                    # In case extra lines sneaked into the log from other
                    # cpp sources
                    j = 1
                    idx1 = -1
                    while idx1 == -1 and j < 3:
                        line2 = lines[line_num + j + i]  # skipping lines
                        idx1 = line2.find(outputs[i])
                        j += 1
                    idx1 = idx1 + len(outputs[i])
                    idx2 = line2.find(' (* 1 = ')
                    if i == len(outputs) - 1:
                        accuracies.append(float(line2[idx1:idx2]))

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)
    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)
    accuracies_iteration_checkpoints_ind = []
    return loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations,
                 accuracies, accuracies_iteration_checkpoints_ind, color_ind=0):
    modula = len(plt.rcParams['axes.color_cycle'])
    ax1.plot(loss_iterations, losses, color=plt.rcParams['axes.color_cycle'][
        (color_ind * 2 + 0) % modula])
    ax1.plot(accuracy_iterations, accuracies,
             plt.rcParams['axes.color_cycle'][(color_ind * 2 + 1) % modula])
    # ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind],
    #          accuracies[accuracies_iteration_checkpoints_ind], 'o',
    #          color=plt.rcParams['axes.color_cycle'][
    #              (color_ind * 2 + 1) % modula])

if __name__ == '__main__':
    log_file_path = [sys.argv[1]]
    draw_train_loss(log_file_path, imsave_path=sys.argv[2])
    # draw_train_loss([
    #     '/Users/eddie/Documents/Projects/Repositories/gygonet/temp-log.txt'])
