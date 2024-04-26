"""
Copyright (c) 2024 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

import boto3
import tarfile
import io
import pathlib

# Recursive function that extracts tarball contents, and if contents include tarballs, call this function again
def extract_tarball(tar, member, dataset, parent_path='', files: list[str] = []):
    # Check if the member is a file and not a tarball
    if len(files) > 0 and member.isfile() and os.path.basename(member.name) not in files:
        return
    if member.isfile() and not member.name.endswith(('.tar.gz', '.tgz', '.tar.bz2', '.tar', '.tbz2')):
        file = tar.extractfile(member)
        content = file.read().decode('utf-8')
        data = {
            "path": parent_path + member.name,
            "content": content
        }
        dataset.append(data)
    elif member.isfile():
        print(f"Ignoring nested archive: {member.name}")
        return

def load_dataset_s3(bucket_name: str, key: str, files: list[str] = []):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)

    # We will determine the mode to use after we get the object
    # Assuming the key name reflects the file type
    mode = 'r'
    if key.endswith(('.tar.gz', '.tgz')):
        mode = 'r:gz'
    elif key.endswith('.tar.bz2'):
        mode = 'r:bz2'

    with io.BytesIO(response['Body'].read()) as file_obj:
        with tarfile.open(fileobj=file_obj, mode=mode) as tar:
            # Create a list to hold datasets
            dataset = []
            for member in tar.getmembers():
                extract_tarball(tar, member, dataset,'', files)
    
    return dataset

def load_dataset_local(file_path: str):
    # Determine the mode based on the file extension
    mode = 'r'
    if file_path.endswith(('.tar.gz', '.tgz')):
        mode = 'r:gz'
    elif file_path.endswith('.tar.bz2'):
        mode = 'r:bz2'
    elif file_path.endswith('.tar'):
        mode = 'r'

    # Read the content of the tar file
    with tarfile.open(file_path, mode=mode) as tar:
        # Create a list to hold datasets
        dataset = []
        for member in tar.getmembers():
            # Skip the file if it is the experiment description
            if 'experiment_description.txt' in member.name:
                continue
            # Recursively extract files or nested tarballs
            extract_tarball(tar, member, dataset)
    
    return dataset

def get_file_content(dataset, path):
    for data in dataset:
        if path in data['path']:
            return data['content']
            break
    return ""

def extract_dataset(archive_file, output_dir):
    dataset = load_dataset_local(archive_file)
    
    for data in dataset:
        path = pathlib.Path(f'{output_dir}/{data["path"]}')
        path.parents[0].mkdir(parents=True, exist_ok=True)
        
        f = open(str(path), "w")
        f.write(data['content'])
        f.close()
        

def load_file_s3(bucket_name: str, key: str) -> str:
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    file_content = response['Body'].read()
    text = file_content.decode('utf-8')
    return text