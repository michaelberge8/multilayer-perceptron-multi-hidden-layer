import matplotlib.pyplot as plt
import neural_network as nn
import graphics as g
import random
import time
import math

'''
    File name: main.py
    Author: Michael Berge
    Date created: 7/19/2018
    Date modified: 5/2/2020
    Python Version: 3.8.1
'''

def main():
    num_i = 2
    num_h1 = 5
    num_h2 = 5
    num_o = 1
    args = [num_i, num_h1, num_o, num_h2]

    neural_network = nn.NeuralNetwork(args)
    start_time = time.time()
    plt.title("Training Data Improvement")
    plt.xlabel("Iterations")
    plt.ylabel("Guess")

    l1 = []
    l2 = []
    l3 = []
    l4 = []

    # Number of training iterations
    epoch = 1000

    # Graphics object
    gr = g.Graphics(args)

    for i in range(epoch):
        file = open("training_data.txt", "r")
        arr = []

        for j in range(4):
            str_ = file.readline()
            str_split = str_.split(" ")
            arr.append(str_split)
        random.shuffle(arr)

        for j in range(4):
            input_ = [int(arr[j][0]), int(arr[j][1])]
            target = [int(arr[j][2].strip())]
            neural_network.train(input_, target, args, gr)

        if i % (epoch / 100) == 0:
            l1.append(neural_network.feed_forward([0, 0], args))
            l2.append(neural_network.feed_forward([1, 1], args))
            l3.append(neural_network.feed_forward([0, 1], args))
            l4.append(neural_network.feed_forward([1, 0], args))
        file.close()

        # Print progress bar
        print_progress_bar(i + 1, epoch, prefix='Progress:', suffix='Complete', length=50)

    # Calculate and display training time
    display_time(start_time)

    # testing data for the network
    print("[0, 0]: " + str(round(neural_network.feed_forward([0, 0], args)[0])))
    print("[1, 1]: " + str(round(neural_network.feed_forward([1, 1], args)[0])))
    print("[0, 1]: " + str(round(neural_network.feed_forward([0, 1], args)[0])))
    print("[1, 0]: " + str(round(neural_network.feed_forward([1, 0], args)[0])))

    # Plot points and display graph
    x = []
    for i in range(epoch):
        if i % (epoch / 100) == 0:
            x.append(i)
    plt.plot(x, l1, "black")
    plt.plot(x, l2, "black")
    plt.plot(x, l3, "black")
    plt.plot(x, l4, "black")
    plt.show()

# Prints the progress bar for training data iterations
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='#'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New line on Complete
    if iteration == total:
        print()

# Displays the time from start_time to the time the function was called
def display_time(start_time):
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("\nTime Elapsed: ", end="")
    if time_elapsed > 60:
        print("{:02d}".format(math.floor(time_elapsed / 60)), end="")
        print(":{:02d}".format(round(time_elapsed % 60)), end="\n\n")
    else:
        print("00:{:02d}".format(round(time_elapsed % 60)), end="\n\n")

if __name__ == "__main__":
    main()