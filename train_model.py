#Step 1 — Libraries import karo
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 2 — Project configuration
TRAIN_PATH = "dataset/train"
VALIDATION_PATH = "dataset/validation"
TEST_PATH = "dataset/test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10

SEED = 42

# Step 3 — Training dataset load karo
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=True,
    seed=SEED
)

# Step 4 — Validation aur test dataset load karo

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=False
)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=False
)

# Step 5 — Dataset preprocessing aur performance

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)
# Step 6 — Data Augmentation

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])

# Step 7 — CNN Architecture create karo

model = models.Sequential()

# Input Layer:
model.add(
    layers.Input(shape=(128, 128, 3))
)

model.add(data_augmentation)

# Normalization Layer:
model.add(
    layers.Rescaling(1.0 / 255)
)

# First Convolution Block:
model.add(
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Second Convolution Block:
model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Third Convolution Block:
model.add(
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )
)

model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
    )
)

# Flatten Layer:
model.add(
    layers.Flatten()
)

# Dense Layer:
model.add(
    layers.Dense(
        128,
        activation="relu"
    )
)

# Dropout Layer:
model.add(
    layers.Dropout(0.5)
)

# Output Layer:
model.add(
    layers.Dense(
        1,
        activation="sigmoid"
    )
)

# Step 8 — Model Compile karo

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Step 9 — Model summary check karo
model.summary()

# Step 10 — Callbacks create karo
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_cat_dog_model.keras",
    monitor="val_loss",
    save_best_only=True
)

# Step 11 — Model train karo
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Step 12 — Model save karo
model.save("cat_dog_cnn.keras")

print("Model saved successfully!")

# Step 13 — Accuracy graph

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# Step 14 — Loss graph

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()






# import os
# import tensorflow as tf
#
# DATASET_PATH = r"D:\NensiProject\ml\cnnwork\cat_dog_image_recognition\dataset"
# deleted_count = 0
#
# for root, dirs, files in os.walk(DATASET_PATH):
#     for file in files:
#         file_path = os.path.join(root, file)
#         try:
#             # Image ko raw bytes mein padhein
#             img_bytes = tf.io.read_file(file_path)
#             # TensorFlow ke exact wahi function se decode karein jo training mein error de raha hai
#             img = tf.io.decode_image(img_bytes, channels=3)
#
#             # Check karein ki image khali (0 size) toh nahi hai
#             if img.shape[0] == 0 or img.shape[1] == 0:
#                 raise ValueError("Empty image")
#
#         except Exception as e:
#             print(f"Removing problematic image: {file_path}")
#             os.remove(file_path)
#             deleted_count += 1
#
# print(f"\nTensorFlow Safai poori hui! Total {deleted_count} kharab images delete ki gayi hain.")