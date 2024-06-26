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

import logging

class Logger:
    def __init__(self, level : int = logging.DEBUG):
        self._level = level
        
    def debug(self, message):
        if self._level < logging.INFO:
            print(message)
        
    def info(self, message):
        if self._level < logging.WARN:
            print(message)
    
    def warn(self, message):
        if self._level < logging.ERROR:
            print(message)
            
    def error(self, message):
        print(message)
    