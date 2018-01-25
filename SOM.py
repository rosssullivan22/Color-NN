from SOMObject import SOM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os

project_root = os.path.abspath(os.path.dirname(__file__))

# Prepare Data
colors = ["Black", "Blue", "Brown", "Green", "Orange", "Pink", "Purple", "Red", "Yellow"]
data_set = open(project_root + '\\colors_dataset.csv', 'r')
data = []

def one_hot_to_index(array):
    for i in range(len(array)):
        if array[i] == 1:
            return i


def one_hot_to_color(array):
    return colors[one_hot_to_index(array)]


def color_to_index(color=str):
    color = color.lower()
    if color == 'black':
        return 0
    if color == 'blue':
        return 1
    if color == 'brown':
        return 2
    if color == 'green':
        return 3
    if color == 'orange':
        return 4
    if color == 'pink':
        return 5
    if color == 'purple':
        return 6
    if color == 'red':
        return 7
    if color == 'yellow':
        return 8


def decimal_to_binary(number=int):
    i = 0
    binary_number = [0, 0, 0, 0, 0, 0, 0, 0]

    while number > 0:
        binary_number[i] = number % 2
        i += 1
        number = int(number / 2)

    return binary_number


def color_from_data_array(array):
    r = array[0] / 255
    g = array[1] / 255
    b = array[2] / 255
    return (r, g, b)


def binary_array_to_decimal(array):
    number = 0
    factor = 1
    for element in array:
        number += element * factor
        factor = factor * 2
    return number


def number_to_one_hot(number):
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    arr[number] = 1
    return arr


def preprocessing(x):
    training_data = []
    training_labels = []

    for arr in x:
        training_data.append([arr[0], arr[1], arr[2]])
        training_labels.append(number_to_one_hot(arr[3]))

    return training_data, training_labels


def shuffle_data(array1, array2):
    n = len(array1)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        array1[i], array1[j] = array1[j], array1[i]
        array2[i], array2[j] = array2[j], array2[i]
    return array1, array2


def test_network_accuracy():
    total_tested = 0
    correct = 0

    for i in range(len(data)):
        index = som.make_color_prediction(data[i])
        actual = one_hot_to_color(labels[i])

        if colors[index].lower() == actual.lower():
            correct += 1
        total_tested += 1
        if i % 500 == 0:
            print(i, 'out of', len(data), 'complete')

    acccuracy = correct / total_tested
    print('Accuracy =', acccuracy)


def predict_color_using_network(color):
    test_color, test_label = preprocessing([color])
    index = som.make_color_prediction(test_color)

    actual = one_hot_to_color(test_label[0])
    print('Predicted color:', colors[index], 'Actual:', actual)


def map_colors():
    mapping_colors = []
    colors_counted = 0

    for i in range(len(colors)):
        color_r = 0
        color_g = 0
        color_b = 0
        colors_counted = 0
        for j in range(len(data)):
            if i == one_hot_to_index(labels[j]):
                color_r += data[j][0]
                color_g += data[j][1]
                color_b += data[j][2]
                colors_counted += 1

        average_r = color_r / colors_counted
        average_g = color_g / colors_counted
        average_b = color_b / colors_counted

        mapping_colors.append([average_r, average_g, average_b, i])

    mapping_colors, mapping_label = preprocessing(mapping_colors)
    som.map_colors(mapping_colors, colors)

def plot(x):
    print('Preparing data to be plotted...')

    # Fit train data into SOM lattice
    mapped = som.map_vects(x)
    mappedarr = np.array(mapped)
    x1 = mappedarr[:, 0]
    y1 = mappedarr[:, 1]

    print('Plotting data points...')

    plt.figure(1, figsize=(12, 6))
    plt.subplot(121)

    for i in range(len(x1)):
        color = color_from_data_array(data[i])
        plt.scatter(x1[i], y1[i], c=color)

    plt.title('Colors')
    plt.show()


for line in data_set.readlines():
    elements = line.split(',')
    red = float(elements[0])
    green = float(elements[1])
    blue = float(elements[2])
    label = color_to_index(elements[3][:-1])
    data_entry = [red, green, blue, label]
    data.append(data_entry)

data, labels = preprocessing(data)
# data, labels = shuffle_data(data, labels)
# data = data[:500]


print('Preprocessing finished...')

m = 50
n = 50
input_size = 3
iterations = 100

som = SOM(m, n, input_size, iterations)
if som.trained is False:
    som.train(data)


map_colors()
test_network_accuracy()
plot(data)

# predict_color_using_network([10, 10, 220, 1])
