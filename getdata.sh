#!/bin/bash

# URL of the file to be downloaded
URL="https://zenodo.org/records/4629685/files/Raw_IQ_Dataset.zip?download=1"

# Destination filename
FILENAME="Raw_IQ_Dataset.zip"

# Function to check if unzip is installed
check_unzip() {
  if ! command -v unzip &> /dev/null; then
    echo "unzip could not be found, installing unzip..."
    apt-get update
    apt-get install -y unzip
  else
    echo "unzip is already installed."
  fi
}

# Check and install unzip if necessary
check_unzip

# Check if the file already exists
if [ -f "$FILENAME" ]; then
  echo "File $FILENAME already exists. Skipping download."
else
  # Download the file
  echo "Downloading the file from $URL..."
  wget -O $FILENAME $URL

  # Check if the download was successful
  if [ $? -ne 0 ]; then
    echo "Download failed!"
    exit 1
  fi
fi

# Extract the downloaded zip file
echo "Extracting the file $FILENAME..."
unzip -o $FILENAME

# Check if the extraction was successful
if [ $? -ne 0 ]; then
  echo "Extraction failed!"
  exit 1
fi
rm $FILENAME

echo "Download and extraction completed successfully."
