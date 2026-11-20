import os
import shutil
import random

# -----------------------------------
# Project Settings
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Possible locations of original dataset
POSSIBLE_ROOTS = [
    os.path.join(BASE_DIR, "raw_dataset"),
    os.path.join(BASE_DIR, "PetImages"),
    r"D:\PetImages"
]

OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset")

TRAIN_COUNT = 800
VALIDATION_COUNT = 200
TEST_COUNT = 100

random.seed(42)

# -----------------------------------
# Find Dataset Root Automatically
# -----------------------------------

def find_class_folder(class_name):

    for root_folder in POSSIBLE_ROOTS:

        if not os.path.exists(root_folder):
            continue

        for root, dirs, files in os.walk(root_folder):

            for folder in dirs:

                if folder.lower() == class_name.lower():

                    return os.path.join(root, folder)

    return None


# -----------------------------------
# Create Output Folders
# -----------------------------------

folders = [
    "train/cats",
    "train/dogs",
    "validation/cats",
    "validation/dogs",
    "test/cats",
    "test/dogs"
]

for folder in folders:

    os.makedirs(
        os.path.join(OUTPUT_FOLDER, folder),
        exist_ok=True
    )


# -----------------------------------
# Copy Images
# -----------------------------------

def prepare_class(class_name):

    source_folder = find_class_folder(class_name)

    if source_folder is None:

        print(f"\n{class_name.upper()} folder not found!")
        return False

    print(f"\nPreparing {class_name} images...")
    print("Source:", source_folder)

    images = [
        f for f in os.listdir(source_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    total_required = (
        TRAIN_COUNT +
        VALIDATION_COUNT +
        TEST_COUNT
    )

    if len(images) < total_required:

        print(f"Not enough {class_name} images!")
        print("Available:", len(images))
        print("Required:", total_required)

        return False

    selected = images[:total_required]

    train_images = selected[:TRAIN_COUNT]

    val_images = selected[
        TRAIN_COUNT:TRAIN_COUNT + VALIDATION_COUNT
    ]

    test_images = selected[
        TRAIN_COUNT + VALIDATION_COUNT:
    ]

    # -----------------------------------
    # Copy Training Images
    # -----------------------------------

    for img in train_images:

        shutil.copy2(
            os.path.join(source_folder, img),
            os.path.join(
                OUTPUT_FOLDER,
                "train",
                class_name.lower() + "s",
                img
            )
        )

    # -----------------------------------
    # Copy Validation Images
    # -----------------------------------

    for img in val_images:

        shutil.copy2(
            os.path.join(source_folder, img),
            os.path.join(
                OUTPUT_FOLDER,
                "validation",
                class_name.lower() + "s",
                img
            )
        )

    # -----------------------------------
    # Copy Testing Images
    # -----------------------------------

    for img in test_images:

        shutil.copy2(
            os.path.join(source_folder, img),
            os.path.join(
                OUTPUT_FOLDER,
                "test",
                class_name.lower() + "s",
                img
            )
        )

    print("Training:", len(train_images))
    print("Validation:", len(val_images))
    print("Testing:", len(test_images))

    return True


# -----------------------------------
# Run Dataset Preparation
# -----------------------------------

print("========== DATASET PREPARATION ==========")

cat_ok = prepare_class("Cat")
dog_ok = prepare_class("Dog")

print("\n========== RESULT ==========")

if cat_ok and dog_ok:

    print("Cats dataset prepared successfully.")
    print("Dogs dataset prepared successfully.")

    total_images = (
        TRAIN_COUNT +
        VALIDATION_COUNT +
        TEST_COUNT
    ) * 2

    print("Total dataset images:", total_images)

else:

    print("Dataset preparation incomplete.")
    print("Please check the original Cat and Dog folders.")

print("Output folder:", OUTPUT_FOLDER)