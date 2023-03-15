FROM python:3.11.2-slim

# Copy the current version of the repo in the image
COPY . /atlas-ftag-tools_repo
# Install and remove the folder afterwards
RUN pip install /atlas-ftag-tools_repo && rm -rf /atlas-ftag-tools_repo
