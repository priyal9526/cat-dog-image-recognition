# ==========================================
# MODEL EVALUATION
# File: evaluate_model.py
# ==========================================

import tensorflow as tf
# STEP 1: LOAD TRAINED MODEL
model = tf.keras.models.load_model(
    "cat_dog_cnn.keras"
)


# STEP 2: LOAD TEST DATASET
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=False
)


# STEP 3: EVALUATE MODEL

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


# STEP 4: DISPLAY RESULT

print("\n===== FINAL TEST RESULT =====")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy * 100, "%")