from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def keras_model(image_x, image_y):
    num_of_classes = 6
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(image_x, image_y, 1)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.2),
        Dense(num_of_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "guitar_learner.keras"  # Adjusted filepath to end with .keras
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list

def main():
    image_x, image_y = 200, 200
    batch_size = 64
    train_dir = "chords"

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="training")

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_x, image_y),
        color_mode="grayscale",
        batch_size=batch_size,
        seed=42,
        class_mode='categorical',
        subset="validation")

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(train_generator, epochs=5, validation_data=validation_generator, callbacks=callbacks_list)
    scores = model.evaluate(validation_generator)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('guitar_learner.h5')

main()
