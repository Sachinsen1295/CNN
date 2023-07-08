from keras.models import Sequential
from keras.layers import Dense,Flatten,MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



classifier = Sequential()

classifier.add(Conv2D(32,(3,3), input_shape= (64,64,3), activation='relu'))
classifier.add(Conv2D(64,(3,3), activation='tanh'))
#classifier.add(Conv2D(20,(3,3), activation='tanh'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Conv2D(20,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=128, activation='relu'))


classifier.add(Dense(units=1, activation='sigmoid'))

classifier.compile(optimizer='adam',loss='BinaryCrossentropy',metrics=['accuracy'])


train_datagen =ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen =ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
    '/Users/sachinsen/Documents/DL Project/dogcat/cats_and_dogs_filtered/train',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    '/Users/sachinsen/Documents/DL Project/dogcat/cats_and_dogs_filtered/validation',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: training_set,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).repeat()  # Repeat the dataset indefinitely

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_set,
    output_signature=(
        tf.TensorSpec(shape=(None, 64, 64, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

model = classifier.fit(
    train_dataset,
    steps_per_epoch=100,
    epochs=1,
    validation_data=test_dataset,
    validation_steps=20,
    shuffle=True
)

classifier.save('model.h5')
print('Model saved')
