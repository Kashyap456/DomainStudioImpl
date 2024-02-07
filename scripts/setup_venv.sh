# !/bin/bash

if [ ! -d "domain_studio" ]; then
  python3 -m venv domain_studio
  echo "Created Python virtual environment 'domain_studio'"
fi

source domain_studio/bin/activate
echo "Activated the 'domain_studio' virtual environment"

pip install -r requirements.txt
echo "Installed dependencies from requirements.txt"
