import tensorflow as tf
import random

# Prepare Data
colors = ["Black", "Blue", "Brown", "Green", "Orange", "Pink", "Purple", "Red", "Yellow"]
data_set = open('C:\\Users\\Ross\\Desktop\\Python Color NN\\colors_dataset.csv', 'r')
data = []

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

for line in data_set.readlines():
    elements = line.split(',')
    red = int(float(elements[0]))
    green = int(float(elements[1]))
    blue = int(float(elements[2]))
    label = color_to_index(elements[3][:-1])
    data_entry = [red, green, blue, label]
    data.append(data_entry)


def decimal_to_binary(number=int):

    i = 0
    binary_number = [0, 0, 0, 0, 0, 0, 0, 0]

    while number > 0 :
        binary_number[i] = number % 2
        i += 1
        number = int(number / 2)

    return binary_number


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



data, labels = preprocessing(data)


# Network
x = tf.placeholder('float', [None, 24])
y = tf.placeholder('float')

hm_epochs = 10

def network_model(x):

    # print(x)

    hidden_one = {'w' : tf.Variable(tf.random_normal([24 , 100])) , 'b' : tf.Variable(tf.random_normal([100]))}
    hidden_two = {'w': tf.Variable(tf.random_normal([100, 50])), 'b': tf.Variable(tf.random_normal([50]))}
    output_layer = {'w': tf.Variable(tf.random_normal([50, 9])), 'b': tf.Variable(tf.random_normal([9]))}

    h1 = tf.add(tf.matmul(x , hidden_one['w']), hidden_one['b'])
    h1 = tf.nn.relu(h1)

    h2 = tf.add(tf.matmul(h1 , hidden_two['w']), hidden_two['b'])
    h2 = tf.nn.relu(h2)

    output = tf.add(tf.matmul(h2 , output_layer['w']), output_layer['b'])
    return output


def shuffle_data(array1, array2):
    n = len(array1)
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        array1[i], array1[j] = array1[j], array1[i]
        array2[i], array2[j] = array2[j], array2[i]
    return array1, array2

def train_neural_network(training_data, training_labels):
    prediction = network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(len(training_data)):
            epoch_loss = 0

            epoch_x = [training_data[epoch]]
            epoch_y = [training_labels[epoch]]

            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
            epoch_loss += c
            if epoch % 10000 == 0:
              print("Training Cycle:", epoch, "  Loss:", epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        testing_data, testing_labels = shuffle_data(training_data, training_labels)

        print('Accuracy:', accuracy.eval({x: testing_data, y: testing_labels}))

        print("Saving...")
        saver = tf.train.Saver()
        saver.save(sess, 'C:\\Users\\Ross\\Desktop\\SeniorProject\\Python Module\\model.ckpt')
        print("Model Saved!")





# Uncomment to train network

#Shuffle and duplicate data for more entries
# data, labels = shuffle_data(data, labels)
# d = data
# l = labels
# for i in range(50):
#     data2, labels2 = shuffle_data(d, l)
#     data = data + data2
#     labels = labels + labels2
#
# train_neural_network(data, labels)

def use_neural_network(data):
    prediction = network_model(x)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, "model.ckpt")

        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: [data]}), 1))
        print('Result: ' + colors[result[0]])

def create_test_color_vector(r,g,b):
    return decimal_to_binary(r) + decimal_to_binary(g) + decimal_to_binary(b)

#Uncomment to use network with test color
# test_color = create_test_color_vector(0, 174 , 250)
# use_neural_network(test_color)