import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings

# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", category=UserWarning)

# Paths to dataset
train_dir = r'C:\Users\LENOVO\OneDrive\Documents\adovi intern project\train_data'
val_dir = r'C:\Users\LENOVO\OneDrive\Documents\adovi intern project\validation_data'
test_dir = r'C:\Users\LENOVO\OneDrive\Documents\adovi intern project\test_data'

# Data augmentation for training; rescaling for validation/testing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# Verify class indices (optional)
print("Class indices:", train_generator.class_indices)

# Build the CNN model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Option 1: Use .keras extension
model_checkpoint = ModelCheckpoint('pneumonia_model.keras', monitor='val_loss', save_best_only=True)

# Option 2: Use HDF5 format explicitly
# model_checkpoint = ModelCheckpoint('pneumonia_model.h5', monitor='val_loss', save_best_only=True, save_format='h5')

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate on the test data
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc:.2f}')

# Generate predictions
test_generator.reset()
preds = model.predict(test_generator)
pred_labels = np.round(preds)

# Extract true labels
true_labels = test_generator.classes

# Calculate evaluation metrics
precision = precision_score(true_labels, pred_labels)
recall = recall_score(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, preds)

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'AUC-ROC: {roc_auc:.2f}')

# Classification report
print(classification_report(true_labels, pred_labels, target_names=['Normal', 'Pneumonia']))

# Plot training history
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the final model (optional)
model.save("final_pneumonia_cnn_model.keras")

