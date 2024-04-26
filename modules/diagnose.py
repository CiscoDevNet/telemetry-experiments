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

from modules.llm.prompt import *
from .logger import Logger
from modules.llm.llm import LlmInterface
from typing import List
from enum import Enum

class Diagnose():
    
    initialDiagnosisField = 'Initial Diagnosis'
    enrichedDiagnosisField = 'Enriched Diagnosis'
    
    def __init__(self, logger : Logger, llm : LlmInterface):
        self._logger = logger
        self._llm = llm
        self._initialDiagnosisField = 'Initial Diagnosis'
        self._issueDescriptionField = 'Description'
        self._issueStateField = 'Issue'
        self._resolutionField = 'Resolution'
        self._inputSource = 'Source'
        self._inputFile = 'File'
        self._inputIdentifier = 'Event'
        self._inputFeatures = 'Features'
        self._inputEntity = 'Entity'
        self._inputType = 'Type'
        self._entityTypes = Enum('Types', ['NETWORK_DEVICE', 'KUBERNETES_POD', 'GENERIC'])
        
        self._filteredOutput = list()
        
    def getNetworkDeviceType(self) -> str:
        return self._entityTypes.NETWORK_DEVICE.name
    
    def getPodType(self) -> str:
        return self._entityTypes.KUBERNETES_POD.name
    
    def getGenericType(self) -> str:
        return self._entityTypes.GENERIC.name
    
    def setInputSource(self, name):
        self._inputSource = name
    
    def setInputIdentifier(self, name):
        self._inputIdentifier = name
    
    def setInputFeatures(self, name):
        self._inputFeatures = name
    
    def setInputType(self, name):
        self._inputType = name
        
    def setInputFile(self, name):
        self._inputFile = name
        
    def setOutputInitialDiagnosis(self, name=None):
        if name is not None:
            self._initialDiagnosisField = name
            
        self._filteredOutput.append(self._initialDiagnosisField)
        
    def setOutputIssueState(self, name=None):
        if name is not None:
            self._issueStateField = name
            
        self._filteredOutput.append(self._issueStateField)
        
    def setOutputIssueDesc(self, name=None):
        if name is not None:
            self._issueDescriptionField = name
            
        self._filteredOutput.append(self._issueDescriptionField)
        
    def setOutputResolution(self, name=None):
        if name is not None:
            self._resolutionField = name
            
        self._filteredOutput.append(self._resolutionField)
    
    def diagnosisPrompt(self) -> DiagnosisPrompt:
        return self._diagnosisPrompt
    
    def applyFieldFilter(self, data, diagnosis) -> dict:
        for field in data.keys():
            if len(self._filteredOutput) > 0:
                if field in self._filteredOutput:
                    diagnosis[field] = data[field]
            else:
                diagnosis[field] = data[field]
                
        return diagnosis
    
    def diagnoseEntityIssue(self, diagnosis) -> dict:
        
        data = {
            self._initialDiagnosisField: "",
        }    
                
        self._diagnosisPrompt = None
        
        # Select the supported diagnosis prompt based upon the source.
        if diagnosis[self._inputSource].upper() == 'MDT':
            self._diagnosisPrompt = MdtDiagnosisPrompt(self._logger, self._llm)
        elif diagnosis[self._inputSource].upper() == 'SYSLOG':
            match diagnosis[self._inputType]:
                case self._entityTypes.NETWORK_DEVICE.name:
                    self._diagnosisPrompt = NetworkDiagnosisPrompt(self._logger, self._llm)
                case self._entityTypes.KUBERNETES_POD.name:
                    self._diagnosisPrompt = KurbernetesPodDiagnosisPrompt(self._logger, self._llm)
                case _:
                    self._diagnosisPrompt = GenericDiagnosisPrompt(self._logger, self._llm)
        
        self._diagnosisPrompt.run(diagnosis[self._inputFile], diagnosis[self._inputFeatures])
        data[self._issueStateField] = self._diagnosisPrompt.isIssue()
        data[self._initialDiagnosisField] = self._diagnosisPrompt.getDiagnosis()
        data[self._issueDescriptionField] = self._diagnosisPrompt.getIssue()
        data[self._resolutionField] = self._diagnosisPrompt.getResolution()
                
        return self.applyFieldFilter(data, diagnosis)
        

    def run(self, retrieved_issues, inject=False):
        
        if inject:
            diagnosis_output = retrieved_issues
        else:
            diagnosis_output = []
            
        for issue in retrieved_issues:    
            
            diagnosis = {
                self._inputSource: issue[self._inputSource],
                self._inputType: issue[self._inputType],
                self._inputFile: issue.get(self._inputFile, "")
            }
            
            if type(issue[self._inputFeatures]) is list:
                diagnosis[self._inputFeatures] = '\n'.join(str(s) for s in issue[self._inputFeatures])
            else:
                diagnosis[self._inputFeatures] = issue[self._inputFeatures]  
                    
            diagnosis = self.diagnoseEntityIssue(diagnosis)
            
            if inject:
                for field in diagnosis.keys():
                    issue[field] = diagnosis[field]
            else:
                diagnosis_output.append(diagnosis)
                
        return diagnosis_output