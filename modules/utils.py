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

import json
import jsonschema
import logging
import pandas as pd
from IPython.display import display, Markdown, HTML
import ipywidgets as widgets


def load_json(json_str, strict=False):
    # By default we want to parse RFC8259 compliant JSON, this may contain escaped control characters within the JSON value strings i.e. \n.
    # However, to be compliant, JSON values must be double quoted, which means in Python they will be handled as string literals, which means the backslash must also be escaped. 
    # The documented way to handle this is to disable strict mode in the Python JSON decoder (default is True), this explicitly allows control characters within literal strings.
    try:
        json_data = json.loads(json_str, strict=strict)
        return json_data, True            
    except ValueError as err:
        print(f"string not in JSON format: {err}")
    
    return None, False
    
def load_validated_json(json_str, schema, strict=False):
    json_data, loaded = load_json(json_str, strict=strict)
    if loaded == False:
        return None, loaded
        
    return validate_json(json_data, schema)
    
    
def validate_json(json_data, schema):
    if schema is None:
        return json_data, True
    
    try:
        jsonschema.validate(instance=json_data, schema=schema)
        return json_data, True
    except jsonschema.exceptions.ValidationError as err:
        print(f"json object failed schema validation: {err}")
       
    return None, False

def logger(name, level : int = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter('%(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger

def displayDictionary(data, showFields : list = None):
    df = None
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    displayDataFrame( df, showFields=showFields)
    
def displayDataFrame(df, showFields : list = None):
    df_style = df.style.hide(axis='index')
    if showFields is not None:
        hideColumns = []
        for column in df:
            hide = True
            for showField in showFields:
                if column == showField:
                    hide = False
                    break
            
            if hide:
                hideColumns.append(column)
        
        if len(hideColumns) > 0:
            df_style = df_style.hide(axis='columns', subset=hideColumns)
    df_style = df_style.set_properties(**{'text-align': 'left','vertical-align': 'text-top', 'white-space': 'pre-wrap'})
    df_style = df_style.set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])])
    display(df_style)    
