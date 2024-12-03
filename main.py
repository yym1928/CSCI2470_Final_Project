import pickle
import tensorflow as tf
from cnn import CNNModel

def r2_score(y_true, y_pred):
    # Total sum of squares
    ss_total = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    # Residual sum of squares
    ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
    # R^2 score
    r2 = 1 - (ss_residual / (ss_total + tf.keras.backend.epsilon()))  # Add epsilon to avoid division by zero
    return r2

def main():
    with open('../data/data.p', 'rb') as f:
        data = pickle.load(f)

    images_train = data['train_images']
    images_test = data['test_images']
    labels_train = data['train_labels'].astype('float32')
    labels_test = data['test_labels'].astype('float32')

    model = CNNModel()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), r2_score]
    )

    model.fit(
        images_train,
        labels_train,
        batch_size=32,
        epochs=20,
    )

    test_loss, test_mae, r2 = model.evaluate(images_test, labels_test)
    print(f"Test Loss: {test_loss}")
    print(f"Test MAE: {test_mae}")
    print(f"Test r2: {r2}")

if __name__ == '__main__':
    main()