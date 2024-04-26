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

from string import Template
import os
import abc
import re
from .llm import LlmInterface
from ..utils import load_validated_json

class JsonCompletion(metaclass=abc.ABCMeta):
    def __init__(self, logger, llm : LlmInterface, schema=None):
        self._schema = schema
        self._llm = llm
        self._max_attempts = int(os.getenv('LLM_MAX_JSON_LOAD_ATTEMPTS', "3"))
        self._logger = logger
        
    def promptJsonResponse(self, prompt): 
        loaded = False
        loadAttempts = 0
        data = None
        
        while loaded == False:
            loadAttempts += 1
            response = self._llm.invokePrompt(prompt)
            taggedJson = re.search(r"(?:\[JSON\])([.\s\S]*)(?:\[JSON\])", response)
            
            if taggedJson is not None:
                data, loaded = load_validated_json(taggedJson.group(1), self._schema)
                
            if loaded == False:
                # TODO decide error handling if the llm response parsing fails, currently we just log and not raise an exception up the call stack, this means 
                # the diagnoser will continue to attempt to diagnose other entities and change points within the investigation.
                # As a side effect, failing to parse an llm response for this diagnosis isn't reported back to the backend, we should determine what status to leave the
                # investigation in if one or more diagnosis responses are not parsed.  Note, currently, if we encounter a genie api call failure then this is passed as an 
                # exception up the call stack, resulting in an abort of the complete investigation diagnose attempt with a subsequent error stored in the investigation 
                # status in the database.
                
                self._logger.debug(f"failed to parse llm response as json to specified schema: {response}")    
                
                if loadAttempts >= self._max_attempts:
                    return None
        
        return data
    
    def setSchema(self, schema):
        self._schema = schema
        
class DiagnosisPrompt(JsonCompletion):
    _template = Template(
"""$context

$task

### $feature_desc \n \n
$features""")
   
    def __init__(self, logger, llm : LlmInterface, schema=None, context="", task="", feature_desc=""):
        super().__init__(logger, llm, schema)
        self._logger = logger
        self._context = context
        self._feature_desc = feature_desc
        self._task = task
        self.reset()
                        
    def run(self, features):
        self._prompt = self._template.substitute(context=self._context, task=self._task, feature_desc=self._feature_desc, features=features)
        self._logger.debug(f'Prompt:\n {self._prompt}')
        diagnosisResponse = self.promptJsonResponse(self._prompt)
        if diagnosisResponse is not None:
            self._data.update(diagnosisResponse)
    
    def reset(self):
        self._prompt = None
        self._data = {}
        
    def getDiagnosis(self):
        return f'ISSUE:\n{self.getIssue()}\n\nRESOLUTION:\n{self.getResolution()}'
    
    def getIssue(self):
        return self._data.get("description", "NONE")
    
    def isIssue(self):
        return self._data.get("issue", True)
    
    def getResolution(self):
        return self._data.get("resolution", "")
    
    def get(self) -> str:
        if self._prompt is None:
           return ""
        
        return self._prompt
    
class MdtDiagnosisPrompt(DiagnosisPrompt):
    _context = """You are a networking expert.
You are diagnosing a network issue based on telemetry information received from a Cisco router running IOS-XR 7.3.1.
"""
    _task = """
Below, in the "List of sensor paths" section you find a list of YANG sensor path counters which
have changed the most while the issue occurred. 
Each line shows the name of the sensor path counter and the absolute amount that the sensor path counter 
has changed separated by the word CHANGE.
The sensor path counters are descriptive of the issue.

Perform the following two steps one after the other:

1. First, create a 'description', explain what is the issue with this router in a single paragraph. Be technical and specific.
2. Second, create a 'resolution', detailing your suggested next steps to fix the issue in a single paragraph. Be technical and specific.

Your response must only contain RFC8259 compliant JSON in the following format wrapped within [JSON] tags, see 'Example' below
### Example
[JSON]
{
    "description": "description of a detected issue or NONE",
    "resolution": "your resolution to a detected issue"
}
[JSON]
"""
    _schema = {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "resolution": {"type": "string"},
        },
        "required": [
            "description",
            "resolution"
        ]
    }
    
    _feature_desc = "List of sensor paths"

    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, schema=self._schema, context=self._context, task=self._task,feature_desc=self._feature_desc)
                        
    def run(self, file, features):
        self._logger.info('LLM Prompt: MDT Sensor Path Diagnosis')
        super().run(features)
        
class SyslogDiagnosis(DiagnosisPrompt):
    _task_is_issue = """Below, in the 'syslog capture' section you will find a capture of the most important log entries from the specified 'syslog file name'.

Perform the following steps:

1) Determine whether the "syslog capture" is an issue containing anomalous information that could point to a potential problem, or whether the "syslog capture" just shows a system operating normally.
2) If you determine the system is operating normally, then set 'issue' to false your response and do not perform steps 3 below (see "Example 1" below), otherwise continue on to next 3.
3) Create a technical 'description' of what this issue is (see "Example 2" below). Be sure to keep technical details in the description (such as host names, IP addresses, interfaces, ...). Also consider the "syslog file name" to determine the host name. 

Your response must only contain RFC8259 compliant JSON in the following format wrapped within [JSON] tags.
### Example 1:
[JSON]
{
    "issue": false
}
[JSON]

### Example 2:
[JSON]
{
    "issue": true,
    "description": "A technical description of the issue",
}
[JSON]
"""
    _schema_is_issue = _schema = {
        "type": "object",
        "properties": {
            "issue": {"type": "boolean"},
            "description": {"type": "string"}
        },
        "required": [
            "issue",            
        ]
    }

    _task_resolution = """Below, in the 'syslog capture' section you will find a capture of the most important log entries used to create the provided 'issue description'.

Perform the following steps:

1) Create a 'resolution', composed of a single paragraph detailing your suggested next steps to fix the provided 'issue description' for the 'syslog capture'. Be technical and specific.

Your response must only contain RFC8259 compliant JSON in the following format wrapped within [JSON] tags, see 'Example' below
### Example:
[JSON]
{
    "resolution: "Suggested next steps to fix the issue"
}
[JSON]
"""
    _schema_resolution = _schema = {
        "type": "object",
        "properties": {
            "resolution": {"type": "string"}
        },
        "required": [
            "resolution",            
        ]
    }
    
    _feature_desc = "Syslog capture"
    
    def __init__(self, logger, llm : LlmInterface, context):
        super().__init__(logger, llm, context=context, task=self._task_is_issue, feature_desc=self._feature_desc)
        
    def taskIsIssue(self, file, features):
        self.reset()
        self._schema = self._schema_is_issue
        self._task = self._task_is_issue
        super().run(features + "\n\n### syslog file name: " + file)                
        
    def taskResolution(self, features):
        if self.isIssue():
            self._schema = self._schema_resolution
            self._task = self._task_resolution
            super().run(features + "\n\n### issue description: " + self.getIssue())
            
    def run(self, file, features):
        self.taskIsIssue(file, features)
        self.taskResolution(features)
        
        
class NetworkDiagnosisPrompt(SyslogDiagnosis):
    _context = """You are a networking expert.
You are diagnosing a network issue based on syslog received from a Cisco router running IOSv 15.9."""
    
    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._context)
                        
    def run(self, file, features):
        self._logger.info('LLM Prompt: Diagnose Network Syslog features')
        super().run(file, features)
        
class KurbernetesPodDiagnosisPrompt(SyslogDiagnosis):
    _context = """You are an IT expert.
You are diagnosing a issue based on a syslog received from a Kubernetes pod."""
   
    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._context)
                        
    def run(self, file, features):
        self._logger.info('LLM Prompt: Diagnose Kubernetes Syslog features')
        super().run(file, features)        
    
class GenericDiagnosisPrompt(SyslogDiagnosis):
    _context = """You are an IT expert.
You are diagnosing an issue based on a syslog received."""
        
    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._context)
                        
    def run(self, file, features):
        self._logger.info('LLM Prompt: Diagnose Syslog features')
        super().run(file, features)     
    
        
class FileTypeDetectionPrompt(JsonCompletion):
    _template = Template(
"""You are a file type analyzer. Given the content of the file with name "$file_name" between three backticks (```), provide the file type. You can choose between: "syslog", "configuration". If you are not sure, use "unknown".
Keep in mind that syslog files always contain logs and timestamps. Configuration files contain system configuration and settings.
```
$file_content
```

Answer only using JSON format following the template below. When providing the answer, write RFC8259 compliant JSON between [JSON] tags. Meaning:
[JSON]
<Your RFC8259 compliant JSON Answer here>
[JSON]

The JSON template:
[JSON]
{
     "type": "syslog" // or "configuration" or "unknown"
}
[JSON]

""")
    
    _schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string"}
        },
        "required": [
            "type"
        ]
    }

    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._schema)
        self._logger = logger
        self._data = None
                
    def run(self, file_name, file_content):
        self._prompt = self._template.substitute(file_name=file_name, file_content=file_content)
        self._logger.debug(f'File Type Detection Prompt:\n {self._prompt}')
        
        self._data = self.promptJsonResponse(self._prompt)
        
    def getData(self):
        if self._data is not None:
            return self._data
        
        return ""
    
    def getType(self):
        return self._data["type"]
    
class IssuesDetectionPrompt(JsonCompletion):
    _template = Template(
"""You are an IT operations engineer. Given the log file included between three backticks (```) and the file name below, perform the following tasks in order.

1) Indicate whether the file just shows a system operating normally or whether the file contains any anomalous information that could point to a potential problem. DO NOT consider "SSH RSA Key Size compliance violation" as issues.

2) If there is an issue, give a description of what this issue is. Also, be sure to keep technical details in the description (such as host names, IP addresses, interfaces, ...).
Also consider the file name to determine the host name.
Log File:
```$file_content```

File Name: $file_name

Answer only using JSON format and use the template below. When providing the answer, write RFC8259 compliant JSON between [JSON] tags. Meaning:
[JSON]
<Your RFC8259 compliant JSON Answer here>
[JSON]

The JSON template:
[JSON]
{
    "issue": true, # only true or false are accepted here
    "description": "issue description", # only a string is accepted here
}
[JSON]

""")
    
    _schema = {
        "type": "object",
        "properties": {
            "issue": {"type": "boolean"},
            "description": {"type": "string"}
        },
        "required": [
            "issue",
            "description"
        ]
    }

    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._schema)
        self._logger = logger
        self._data = None
                
    def run(self, file_name, file_content):
        self._prompt = self._template.substitute(file_name=file_name, file_content=file_content)
        self._logger.debug(f'Issues Detection Prompt:\n {self._prompt}')
        
        self._data = self.promptJsonResponse(self._prompt)
        
    def getData(self):
        if self._data is not None:
            return self._data
        
        return ""
    
    def getIssue(self):
        return self._data["issue"]

    def getDescription(self):
        return self._data["description"]

class SummarizeIssuesPrompt(JsonCompletion):
    _template = Template(
"""You are an IT operations engineer who summarizes issues. Summarize the issues in the list included between three backticks (``` ```):

```$issues```

Answer only using JSON format and use the template below. When providing the answer, write RFC8259 compliant JSON between [JSON] tags. Meaning:
[JSON]
<You JSON Answer here>
[JSON]

The JSON template:
[JSON]
{
    "summary" : "write the summary of the issues here" # write only a string here
}
[JSON]
""")
    
    _schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": [
            "summary"
        ]
    }

    def __init__(self, logger, llm : LlmInterface):
        super().__init__(logger, llm, self._schema)
        self._logger = logger
        self._data = None
                
    def run(self, issues):
        self._prompt = self._template.substitute(issues=issues)
        self._logger.debug(f'Summarize Issues Prompt:\n {self._prompt}')
        
        self._data = self.promptJsonResponse(self._prompt)
        
    def getData(self):
        if self._data is not None:
            return self._data
        
        return ""
    
    def getSummary(self):
        return self._data["summary"]
