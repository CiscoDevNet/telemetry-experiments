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

from openai import AzureOpenAI
from .llm import LlmInterface
import os
from typing import Tuple

class AzureLlm(LlmInterface):
    def __init__(self, logger, apiKey):

        self._client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-03-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        self._logger = logger
        self._logger.info("LLM Endpoint: " + os.getenv("AZURE_OPENAI_ENDPOINT"))
    
    def invokePrompt(self, prompt : str)  -> str: 
        response = self._client.chat.completions.create(
            model=os.getenv("AZURE_MODEL"),
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def invokeRagSearch(self, collection, query) -> str:
        raise NotImplementedError
    
    def getRagChunks(self) -> Tuple[str, str]:
        raise NotImplementedError