import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

n = 100     # Number of individuals
k = 2       # Dimension of feature space
B = 10      # Bound on the coefficients of the linear classifier
tau = 0.05  # Tolerance
M = 1       # 
G = 1

def generate_data():
    center_class1 = 3*np.random.uniform(size=(1, k))       # Center of class -1 will be -1 * this
    center_class1[0, 0] = 0

    labels_A = np.random.choice((-1, 1), size=n)
    individuals_base_A = np.random.normal(size=(n, k))
    individuals_A= individuals_base_A + center_class1 * np.reshape(labels_A, (n, 1))
    individuals_A[:,0] = np.abs(individuals_A[:,0])

    labels_B = np.random.choice((-1, 1), size=n)
    individuals_base_B = np.random.normal(size=(n, k))
    individuals_B = individuals_base_B + center_class1 * np.reshape(labels_B, (n, 1))
    individuals_B[:,0] = -np.abs(individuals_B[:,0])

    individuals = np.concatenate((individuals_A, individuals_B), 0)
    labels = np.concatenate((labels_A, labels_B), 0)

    return individuals, labels, center_class1, individuals_A, individuals_B, labels_A, labels_B

def generate_comparison_class():
    C = []
    S = np.zeros((n ** 2, 2))
    ctr = 0
    for i in range(n):
        for j in range(n):
            S[ctr][0] = i
            S[ctr][1] = j
            ctr += 1
    C.append(S)
    return C

tf.reset_default_graph()

C = generate_comparison_class()
S = tf.constant(C[0], dtype=tf.int32)

individuals = tf.placeholder(shape=(2 * n, k), dtype=tf.dtypes.float32)
labels = tf.placeholder(shape=(2 * n,), dtype=tf.dtypes.float32)

# initial_classifier = tf.random.uniform(shape=[k], minval=-B, maxval=B)
initial_classifier = tf.constant([1., 2.])
classifier = tf.Variable(initial_value=initial_classifier, trainable=True)

def classify(individual):
    return tf.math.sign(tf.reduce_sum(tf.math.multiply(classifier, individual)))

fairness_loss_running = 0.
ctr = 0
def update_loss(ctr, fairness_loss_running):
    treatment_gap = tf.abs(classify(individuals[S[ctr, 0]]) - classify(individuals[S[ctr, 1]]))
    distance = tf.norm(individuals[S[ctr, 0], 1:] - individuals[S[ctr, 1], 1:])
    fairness_loss_running += treatment_gap - distance
    return ctr + 1, fairness_loss_running

_, fairness_loss = tf.while_loop(
    lambda ctr, fairness_loss_running: tf.less(ctr, n ** 2),
    update_loss,
    [ctr, fairness_loss_running]
)

fairness_loss /= n ** 2

classifications = tf.map_fn(classify, individuals, dtype=tf.float32)
classification_loss = tf.reduce_mean(tf.math.square(classifications - labels))

feasible_iterates = tf.zeros((1, k))

fairness_optimizer = tf.train.GradientDescentOptimizer(learning_rate=tau / M ** 2)
classification_optimizer = tf.train.GradientDescentOptimizer(learning_rate=tau / (G * M))

fairness_minimize = fairness_optimizer.minimize(fairness_loss)
classification_minimize = classification_optimizer.minimize(classification_loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

individuals_all, labels_all, center_class1, A, B, la, lb = generate_data()
individuals_all = individuals_all.astype(float)
labels_all = labels_all.astype(float)


plt.scatter(A[la == 1,0], A[la == 1,1], c="red", marker="o")
plt.scatter(A[la == -1,0], A[la == -1,1], c="red", marker="x")
plt.scatter(B[lb == 1,0], B[lb == 1,1], c="blue", marker="o")
plt.scatter(B[lb == -1,0], B[lb == -1,1], c="blue", marker="x")
plt.scatter(center_class1[0, 0], center_class1[0, 1], c="black", marker="D")
plt.savefig("scatter.png")

m = sess.run(classifier)
x = np.linspace(-2, 2, 100)
y = -m[0] * x / m[1]
plt.plot(x, y)
plt.savefig("scatter.png")

f = 0
for i in range(100):
    if f > 4 * tau / 5:
        # grads = tf.gradients(fairness_loss, classifier)
        # classifier -= (tau / M ** 2) * grads
        print(np.shape(individuals_all))
        print(individuals_all.dtype)
        _, m, c, f = sess.run([fairness_minimize, classifier, classification_loss, fairness_loss], 
                 feed_dict={individuals:individuals_all, labels:labels_all})
        print("fairness step", c, f)
    else:
        tf.concat([feasible_iterates, tf.reshape(classifier, (1, k))], 0)
        # grads = tf.gradients(classification_loss, classifier)
        # classifier -= (tau / (G * M)) * grads
        print(np.shape(individuals_all))
        print(individuals_all.dtype)
        _, m, c, f = sess.run([classification_minimize, classifier, classification_loss, fairness_loss], 
                 feed_dict={individuals:individuals_all, labels:labels_all})
        print("classification step", c, f)
    x = np.linspace(-2, 2, 100)
    y = -m[0] * x / m[1]
    plt.plot(x, y)
    plt.savefig("scatter.png")
