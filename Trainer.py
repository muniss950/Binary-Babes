from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def keras_model(image_x, image_y):
#     num_of_classes = 3
#     model = Sequential([
#         # Flatten(),

#         Conv2D(32, (3, 3), activation='relu', input_shape=(image_x, image_y, 1)),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),

#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),

#         Conv2D(256, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),

#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),

#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(pool_size=(2, 2)),
#         Dropout(0.25),


#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_of_classes, activation='softmax')
#     ])

#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     filepath = "guitar_learner.keras"  # Adjusted filepath to end with .keras
#     checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
#     callbacks_list = [checkpoint]

#     return model, callbacks_list
def keras_model(image_x, image_y):
    num_of_classes = 3
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(image_x, image_y, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_of_classes, activation='softmax')  # Adjusted to match the number of classes
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "guitar_learner.keras"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    return model, callbacks_list
def main():
    image_x, image_y = 200, 200
    batch_size = 64
    train_dir = "chords"

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

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
    model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=callbacks_list)
    scores = model.evaluate(validation_generator)
    print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('test_new_guitar_learner1211345.h5')

main()
