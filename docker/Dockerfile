FROM python:3.10.8-slim

# Copy the current version of the repo in the image
COPY . /atlas-ftag-tools_repo
# Install and remove the folder afterwards
RUN python3 -m pip install /atlas-ftag-tools_repo && rm -rf /atlas-ftag-tools_repo
