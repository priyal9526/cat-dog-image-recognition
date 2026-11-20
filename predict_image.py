
import tensorflow as tf
import numpy as np

# STEP 1: LOAD TRAINED MODEL

model = tf.keras.models.load_model(
    "cat_dog_cnn.keras"
)


# STEP 2: SELECT IMAGES

image_paths = [
    # "test_images/dog.jpg",
    # "test_images/cats.jpg"
    "test_images/cat1.jpg"
]


# STEP 3: PREDICT IMAGES

for image_path in image_paths:

    print("\n-----------------------------------")
    print("Testing Image:", image_path)

    # LOAD AND RESIZE IMAGE

    image = tf.keras.utils.load_img(
        image_path,
        target_size=(128, 128)
    )

    # CONVERT IMAGE INTO ARRAY

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # ADD BATCH DIMENSION

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # MODEL PREDICTION

    prediction = model.predict(
        image_array,
        verbose=0
    )

    # GET DOG PROBABILITY

    dog_probability = float(prediction[0][0])

    # GET CAT PROBABILITY

    cat_probability = 1 - dog_probability

    # PREDICT CLASS

    if dog_probability >= 0.5:

        predicted_class = "DOG"

    else:

        predicted_class = "CAT"

    # DISPLAY RESULTS

    print("Prediction:", predicted_class)

    print(
        "CAT Confidence:",
        round(cat_probability * 100, 2),
        "%"
    )

    print(
        "DOG Confidence:",
        round(dog_probability * 100, 2),
        "%"
    )
