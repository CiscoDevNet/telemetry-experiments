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

from abc import ABC, abstractmethod
from typing import Tuple

class LlmInterface(ABC):
    @abstractmethod
    def invokePrompt(self, prompt : str) -> str:
        return None
 
    @abstractmethod
    def invokeRagSearch(self, collection, query) -> str:
        return None, None
    
    @abstractmethod
    def getRagChunks(self) -> Tuple[str, str]:
        return None
    
class RagChunk(dict):
    def __init__(self, content, score):
        #self = {'Score': score, 'Content': content}
        super().__init__({'Similarity Score': score, 'Content': content})
                
    def getContent(self):
        return self['Content']
        
    def getSimilarityScore(self):
        return self['Similarity Score']
