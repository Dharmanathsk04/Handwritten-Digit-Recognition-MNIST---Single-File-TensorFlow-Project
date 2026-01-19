

import tensorflow as tf
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = x_train / 255.0
x_test = x_test / 255.0



model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

prediction = model.predict(x_test)

plt.imshow(x_test[0], cmap='gray')
plt.title("Predicted Digit: " + str(prediction[0].argmax()))
plt.axis('off')
plt.show()
