from transformers import GPT2LMHeadModel,GPT2TokenizerFast
import torch
from log import LogClass
from Config import ConfigClass


class GPT2Class:
    def __init__(self):
        # Getting the Config Data
        self.configObj = ConfigClass("params.yaml")
        self.configData = self.configObj.Loading_Config()
        self.logFileName = self.configData["LoggingFileName"]
        self.Model_name = self.configData['Loading']['Model_name']
        self.tokenizer_name = self.configData['Loading']['TokenizerDir']
        self.max_length = self.configData['Generate']['max_length']
        self.do_sample = self.configData['Generate']['do_sample']
        self.top_p = self.configData['Generate']['top_p']
        self.top_k = self.configData['Generate']['top_k']
        # calling the Logger to Log
        self.loggerObj = LogClass(self.logFileName)
        # Loading the Model
        self.loggerObj.Logger("Loading Model")
        self.Model = torch.load(self.Model_name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.tokenizer_name)
        self.loggerObj.Logger("Model & Tokenizer Loaded")

    def Encoding(self,text):
        try:
            input_ids = self.tokenizer.encode_plus(text, return_tensors="pt", add_special_tokens=True)
            return input_ids
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Encoding method of GPT2Class : "+str(e))
            return "Exception Occured in Encoding method of GPT2Class : "+str(e)

    def Generate_output(self,input_ids):
        try:
            # here we are applying the Top k & Top P algorithm for generating Output
            outputs = self.Model.generate(**input_ids, max_length=self.max_length, do_sample=self.do_sample, top_p=self.top_p, top_k=self.top_k)
            return outputs
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Generate_output method of GPT2Class : "+str(e))
            return  "Exception Occured in Generate_output method of GPT2Class : "+str(e)

    def Decoding(self,outputs):
        try:
            string  = self.tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return string[:string.rindex(".")]
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Decoding method of GPT2Class : "+str(e))
            return "Exception Occured in Decoding method of GPT2Class : "+str(e)


    def Prediction(self,text):
        try:
            # Encoding the Input_text data
            input_ids = self.Encoding(text)

            # Generating the output Vector
            outputs = self.Generate_output(input_ids)

            # Decoding the output Generated vector into the string
            string = self.Decoding(outputs)

            return  string
        except Exception as e:
            self.loggerObj.Logger("Exception Occured in Prediction method of GPT2Class : "+str(e))
            return "Exception Occured in Prediction method of GPT2Class : "+str(e)




