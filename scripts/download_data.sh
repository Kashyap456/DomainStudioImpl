# !/bin/bash

FILE_ID="16PxCGtP5qLaneN8zUdYDWn3_Ok7P3qyT"
ZIP_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

curl -L -o domain_set.zip $ZIP_URL
unzip domain_set.zip
rm -rf domain_set.zip
rm -rf __MACOSX

echo "Downloaded and unzipped domain_set.zip"