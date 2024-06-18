import os
import scipy.io
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pandas_tfrecords as pdtfr
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Activation, BatchNormalization, Dropout, Conv2D, Flatten, Dense, MaxPooling2D, GlobalAveragePooling2D, Permute
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support

# Define the directory containing the .mat files
directory_path = os.getcwd()
data_path = os.path.join(directory_path, 'Raw_IQ_Dataset')

train_path = os.path.join(data_path, 'Training')
test_path = os.path.join(data_path, 'Testing') 

# Check if TFRecord files already exist
train_tfrecord_path = os.path.join(directory_path,'Train', 'train.tfrecord')
valid_tfrecord_path = os.path.join(directory_path, 'Train', 'valid.tfrecord')
test_tfrecord_path = os.path.join(directory_path, 'Train', 'test.tfrecord')

force_data_pass = False # Set to false to force to write new tfrecords


def getData(filePath):
    # Initialize lists to store data and labels
    data = []
    labels = []

    # Load each .mat file and extract the IQ data and labels
    for folder in os.listdir(filePath):
        for filename in sorted(os.listdir(os.path.join(filePath, folder))):
            if filename.endswith('.mat'):
                file_path = os.path.join(filePath,folder,filename)
                mat_data = scipy.io.loadmat(file_path)
                
                # varname is the key in the dict for the IQ data
                varname = list(mat_data)[-1]
                iq_data = mat_data[varname].flatten()

                # Label is defined as the folder the data is in i.e. DME, NB, NoJAM
                label = folder
                
                data.append(iq_data)
                labels.append(label)
    return data,labels

data,labels = getData(train_path)
# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode string labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(data))
train_data, train_labels = data[:train_size], encoded_labels[:train_size]
val_data, val_labels = data[train_size:], encoded_labels[train_size:]

# Get Test Data
data,labels = getData(test_path)
# Convert lists to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode string labels to integers
encoded_labels = label_encoder.fit_transform(labels)

test_data = data
test_labels = encoded_labels

# Print the shapes of the datasets
print(f"Train data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Validation labels shape: {val_labels.shape}")
print(f"Test labels shape: {test_labels.shape}")
print(f'Train first value and label: {train_labels[0]},{train_data[0][0]}')
print(f'Valid first value and label: {val_labels[0]},{val_data[0][0]}')
print(f'Test first value and label: {test_labels[0]},{test_data[0][0]}')

# Function to create a TFRecord example with complex data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(iq_data, label):
    iq_data_real = np.real(iq_data).astype(np.float32)
    iq_data_imag = np.imag(iq_data).astype(np.float32)
    feature = {
        'iq_data_real': _bytes_feature(iq_data_real.tobytes()),
        'iq_data_imag': _bytes_feature(iq_data_imag.tobytes()),
        'label':  _int64_feature([label])
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def write_tfrecord(file_path, data, labels):
    with tf.io.TFRecordWriter(file_path) as writer:
        for iq_data, label in zip(data, labels):
            example = serialize_example(iq_data, label)
            writer.write(example)

if not (os.path.exists(train_tfrecord_path) and os.path.exists(valid_tfrecord_path) and os.path.exists(test_tfrecord_path) and force_data_pass):
    # Write training and validation data to TFRecord files
    write_tfrecord(train_tfrecord_path, train_data, train_labels)
    write_tfrecord(valid_tfrecord_path, val_data, val_labels)
    write_tfrecord(test_tfrecord_path, test_data, test_labels)
# Function to parse the TFRecord with complex data
def _parse_function(proto):
    feature_description = {
        'iq_data_real': tf.io.FixedLenFeature([], tf.string),
        'iq_data_imag': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    iq_data_real = tf.io.decode_raw(parsed_features['iq_data_real'], tf.float32)
    iq_data_imag = tf.io.decode_raw(parsed_features['iq_data_imag'], tf.float32)
    iq_data = tf.stack([iq_data_real, iq_data_imag], axis=-1)
    iq_data = iq_data[np.newaxis]
    label = parsed_features['label']
    return iq_data, label

# Create a dataset from the TFRecord file
def create_dataset(tfrecord_files, BATCH_SIZE, NUM_EPOCHS, cycle_length):
    dataset = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(_parse_function),
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.repeat(NUM_EPOCHS)# Maybe be over generating data
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_TIME_POINTS = 16368 #Should be 16368 , number of samples IQ generated 
NUM_FEATURES = 2
NUM_FILTERS_PER_FILTER = 16
TRAIN_SIZE = 4800
VALID_SIZE = 1200
TEST_SIZE = 1500
# Create a list of TFRecord files (in this case, we have only one for train and one for valid)
train_tfrecord_files = [train_tfrecord_path]
valid_tfrecord_files = [valid_tfrecord_path]

# Create training and validation datasets
train_dataset = create_dataset(train_tfrecord_files, BATCH_SIZE, NUM_EPOCHS, cycle_length=6)
val_dataset = create_dataset(valid_tfrecord_files, BATCH_SIZE, NUM_EPOCHS, cycle_length=6)

# Function to plot the complex IQ data
def plot_complex_data(data):
    # Plot real and imaginary parts separately
    fig, axs = plt.subplots(2, figsize=(10, 6))
    axs[0].plot(data[:,0])
    axs[0].set_title('Real Part')
    axs[1].plot(data[:,1])
    axs[1].set_title('Imaginary Part')
    plt.tight_layout()
    plt.show()

# Visualize the first sample in the training dataset
sample = train_dataset.take(1)  # Assuming train_data contains the complex IQ data
line, _ = list(sample)[0]
# plot_complex_data(line[0])
Force_Retrain = True

model_path = os.path.join(directory_path, 'Models', 'Standard_Data.keras')
if os.path.exists(model_path) and not Force_Retrain:
    # Load the model for prediction
    model = load_model(model_path)
else:
    # Define the CNN-LSTM model with batch normalization, dropout, and LSTM layers
    inputs = Input(shape=(1,NUM_TIME_POINTS, NUM_FEATURES), batch_size=BATCH_SIZE)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding='same')(inputs)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)  # Six signal classification categories
    model = Model(inputs=inputs, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_dataset,steps_per_epoch=TRAIN_SIZE//BATCH_SIZE,
               epochs=NUM_EPOCHS, validation_data=val_dataset, 
               verbose = 1, validation_steps = VALID_SIZE//BATCH_SIZE)

    # Save the model
    model.save(model_path)

# Provide a model summary
model.summary()

# Create a list of TFRecord files for testing
test_tfrecord_files = [test_tfrecord_path]

# Create test dataset
test_dataset = create_dataset(test_tfrecord_files, BATCH_SIZE, 1 , cycle_length=6)

# Make predictions using the loaded model
predictions = model.predict(test_dataset, batch_size = BATCH_SIZE, steps = TEST_SIZE//BATCH_SIZE)
predicted_labels = np.argmax(predictions, axis=1)

# Get true labels from the test dataset
true_labels = []
print(f'Test Size: {TEST_SIZE}')
print(f'Predicted Labels: {len(predicted_labels)}')
print(f'Take size: {(TEST_SIZE//BATCH_SIZE)*BATCH_SIZE} ')

for _, label in test_dataset.unbatch():
    true_labels.append(label.numpy())
true_labels = np.array(true_labels)

print(f'Length of True Labels: {len(true_labels)}')

true_labels = true_labels[0:(TEST_SIZE//BATCH_SIZE)*BATCH_SIZE]

print(f'Corrected True labels length: {len(true_labels)} ')

# Evaluate the model's performance
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Create a confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.savefig('Figures/CM_Un_Normalized.pdf')
plt.show()
