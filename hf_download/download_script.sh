#!/bin/bash

# Check if enough arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <repo-type> <model-or-dataset-name> [local-dir]"
  exit 1
fi

REPO_TYPE=$1
NAME=$2
MAX_RETRIES=5  # Maximum number of retries
RETRY_DELAY=10  # Delay between retries in seconds

# Set resource type based on --repo-type value
case $REPO_TYPE in
  model)
    # Download model
    ;;
  dataset)
    # Download dataset
    ;;
  *)
    echo "Error: --repo-type must be 'model' or 'dataset'."
    exit 1
    ;;
esac

# If the third argument is provided, use it as the local directory; otherwise, use the name as the local directory
if [ -z "$3" ]; then
  LOCAL_DIR=$(basename "$NAME")
else
  LOCAL_DIR=$3
fi

# Function to download with retry mechanism
download() {
  local attempt=1

  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Attempt $attempt: Downloading $REPO_TYPE from $NAME to $LOCAL_DIR..."
    huggingface-cli download \
      --repo-type "$REPO_TYPE" \
      --resume-download "$NAME" \
      --local-dir "$LOCAL_DIR" \
      --local-dir-use-symlinks False

    if [ $? -eq 0 ]; then
      echo "Download completed successfully."
      return 0
    else
      echo "Error during download. Retrying in $RETRY_DELAY seconds..."
      sleep $RETRY_DELAY
      attempt=$((attempt + 1))
    fi
  done

  echo "Failed to download after $MAX_RETRIES attempts."
  return 1
}

# Execute the download function
download
