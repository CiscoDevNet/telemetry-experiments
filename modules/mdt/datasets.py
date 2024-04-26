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

import os
import pathlib
import re

from .utils import *
from IPython.display import display, Markdown, HTML
import ipywidgets as widgets
import modules.mdt.utils as utils

class Datasets:

    DATASETS_FOLDER = "datasets"

    def __init__(self, dataset_archive = None, datasets_dir = None):        
        
        self._datasets_dir = datasets_dir
        if self._datasets_dir is None:
            self.find_datasets()
        self._datasets = {}
        self._dataset_dropbox = None
        self._device_dropbox = None
        self._file_dropbox = None
        self.list_datasets()

    def find_datasets(self, curpath=None):
        if curpath is None:
            curpath = pathlib.Path(__file__).parent
        dirs = [d for d in os.listdir(curpath.resolve()) if os.path.isdir(os.path.realpath(os.path.join(curpath, d)))]
        if Datasets.DATASETS_FOLDER in dirs:
            self._datasets_dir = os.path.join(curpath.resolve(), Datasets.DATASETS_FOLDER)
            return
        if curpath.name=="":
            raise Exception(f"Datasets folder '{Datasets.DATASETS_FOLDER}' not found in hierarchy!")
        self.find_datasets(curpath.parent)

    def list_datasets(self):
        for dataset_path in [ f.path for f in os.scandir(self._datasets_dir) if f.is_dir() ]:        
            dataset = os.path.basename(dataset_path)
            yang_models_dir = os.path.join(dataset_path, "yang_models")
            if not os.path.exists(yang_models_dir):
                continue
            notes = ""
            try:
                with open(os.path.join(dataset_path, "notes.md"), 'r') as notesdata:
                    notes = notesdata.read()
            except:
                pass

            self._datasets[dataset] = {
                'path': dataset_path,
                'notes': notes,
                'devices': [ os.path.basename(f.path) for f in os.scandir(yang_models_dir) if f.is_dir() and not os.path.basename(f.path).startswith('.')]
            }


    def jupyter_select_dataset_device(self, filter = None, select_file = True):

        display(HTML("<h1>Available Datasets:</h1><hr>"))
        dataset_list = sorted(list(self._datasets))
        for dataset in dataset_list:
            display(HTML(f"<h2>{dataset}</h2>"))
            display(Markdown(self._datasets[dataset]["notes"]))
            display(HTML("<hr>"))

        self._dataset_dropbox = widgets.Dropdown(
            description = "Dataset:",
            options = dataset_list,
            value = dataset_list[0] if len(dataset_list) > 0 else None,
            disabled=False,
            layout=ITEM_LAYOUT,
            style=ITEM_STYLE
        )

        device_list = []
        if len(self._datasets) > 0:
            device_list = sorted(self._datasets[self._dataset_dropbox.value]["devices"])

        self._device_dropbox = widgets.Dropdown(
            description = "Device:",
            options = device_list,
            value = device_list[0] if len(device_list) > 0 else None,
            disabled = False,
            layout=ITEM_LAYOUT,
            style=ITEM_STYLE
        )

        def on_dataset_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                device_list = sorted(self._datasets[self._dataset_dropbox.value]["devices"])
                self._device_dropbox.options = device_list
                self._device_dropbox.value = device_list[0]

        self._dataset_dropbox.observe(on_dataset_change)

        files = sorted(self.list_files(filter=filter))
        self._file_dropbox = widgets.Dropdown(
            description="Data file:",
            layout=ITEM_LAYOUT,
            style=ITEM_STYLE,
            options = files,
            value = files[0] if len(files) > 0 else None,
            disabled = False
        )

        def on_device_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                files = sorted(self.list_files())
                self._file_dropbox.options = files
                self._file_dropbox.value = files[0] if len(files) > 0 else None

        self._device_dropbox.observe(on_device_change)
        
        if select_file:
            jupyter_display_options(self._dataset_dropbox, self._device_dropbox, self._file_dropbox)
        else:
            jupyter_display_options(self._dataset_dropbox, self._device_dropbox)


    def list_files(self, dataset=None, device=None, filter=None):        
        if dataset is None:
            data = self._dataset_dropbox.value
        if device is None:
            device = self._device_dropbox.value
        if filter is None:
            filter = "^(merged|preprocessed|crft).*[.]csv"

        pattern = re.compile(filter)
        files = []
        if data is not None and device is not None:
            device_path = os.path.join(self._datasets_dir, self._dataset_dropbox.value, "yang_models", self._device_dropbox.value)
            files = [ os.path.basename(f.path) for f in os.scandir(device_path) if pattern.match(os.path.basename(f.path)) ]
            
        return files


    def jupyter_select_input_file(self, description="Input file:", filter="^(merged|preprocessed|crft).*[.]csv"):
        files = sorted(self.list_files(filter=filter))
        select_file = widgets.Dropdown(
            description=description,
            layout=ITEM_LAYOUT,
            style=ITEM_STYLE,
            options = files,
            value = files[0] if len(files) > 0 else None,
            disabled = False
        )

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                files = sorted(self.list_files())
                select_file.options = files
                select_file.value = files[0] if len(files) > 0 else None

        self._device_dropbox.observe(on_change)
        return select_file

    def get_datasets_dir(self):
        return self._datasets_dir

    def get_dataset(self):
        return self._dataset_dropbox.value

    def get_device(self):
        return self._device_dropbox.value
    
    def get_filename(self):
        return self._file_dropbox.value
    
    def get_dataset_dir(self):
        return os.path.join(self._datasets_dir, self._dataset_dropbox.value)

    def get_device_yang_dir(self):
        return os.path.join(self.get_dataset_dir(), 'yang_models', self.get_device())

    def get_input_data_file(self, input_file):
        input_path = os.path.join(self.get_device_yang_dir(), input_file)
        return input_path, input_file
