# !/bin/bash

FILE_ID="1cAilBhSDkdU7i2cJRexLKVcrcLtXH4vv"
ZIP_URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

curl -L -o domain_set.zip $ZIP_URL
unzip domain_set.zip
rm -rf domain_set.zip
rm -rf __MACOSX

echo "Downloaded and unzipped domain_set.zip"