from SOMObject import SOM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


# Prepare Data
colors = ["Black", "Blue", "Brown", "Green", "Orange", "Pink", "Purple", "Red", "Yellow"]
data_set = open('C:\\Users\\Ross\\Desktop\\Python Color NN\\colors_dataset.csv', 'r')
data = []

def one_hot_to_color(array):
    for i in range(len(array)):
        if array[i] == 1:
            return colors[i]


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
    r = binary_array_to_decimal(array[0:8]) / 255
    g = binary_array_to_decimal(array[8:16]) / 255
    b = binary_array_to_decimal(array[16:24]) / 255
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
        training_data.append(decimal_to_binary(arr[0]) + decimal_to_binary(arr[1]) + decimal_to_binary(arr[2]))
        training_labels.append(number_to_one_hot(arr[3]))

    return training_data, training_labels


def shuffle_data(array1, array2):
    n = len(array1)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        array1[i], array1[j] = array1[j], array1[i]
        array2[i], array2[j] = array2[j], array2[i]
    return array1, array2


for line in data_set.readlines():
    elements = line.split(',')
    red = int(float(elements[0]))
    green = int(float(elements[1]))
    blue = int(float(elements[2]))
    label = color_to_index(elements[3][:-1])
    data_entry = [red, green, blue, label]
    data.append(data_entry)


data, labels = preprocessing(data)
data, labels = shuffle_data(data, labels)

#Shuffle and duplicate data for more entries
# d = data
# l = labels
# for i in range(10):
#     data2, labels2 = shuffle_data(d, l)
#     data = data + data2
#     labels = labels + labels2

print('Preprocessing finished...')

m = 50
n = 50
input_size = 24
iterations = 65

som = SOM(m, n, input_size, iterations)
if som.trained is False:
    som.train(data)


# cents = som.get_centroids()
# print(cents)



print('Preparing data to be plotted...')

# Fit train data into SOM lattice
mapped = som.map_vects(data)
mappedarr = np.array(mapped)
x1 = mappedarr[:,0]
y1 = mappedarr[:,1]

print('Plotting data points...')

## Plot
plt.figure(1, figsize=(12, 6))
plt.subplot(121)

for i in range(len(x1)):
    color = color_from_data_array(data[i])
    plt.scatter(x1[i], y1[i],c=color)

# Adding text
# for i, m in enumerate(mapped):
#     if i % (x1 / 20) == 0:
#         color = one_hot_to_color(labels[i])
#         plt.text(m[0], m[1], color, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.title('Colors')
plt.show()





