from flask import Flask,jsonify,request,render_template
from GPT2LM import GPT2Class
from log import LogClass
from Config import ConfigClass


app = Flask(__name__)

# calling the Classes
ModelObj = GPT2Class()
configObj = ConfigClass("params.yaml")
configData = configObj.Loading_Config()
LogObj = LogClass(configData['LoggingFileName'])

@app.route("/",methods = ["GET"])
def HomePage():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def PredictionRoute():
    if request.json is not None:
        text = request.json['text']
        LogObj.Logger("Get the Text from User: "+str(text))
        generated_output = ModelObj.Prediction(text)
        dic = {
            "Input Text" : text,
            "Generated Output" : generated_output
        }
        LogObj.Logger("Get prediction Successfully")
        return  jsonify(dic)




if __name__ == "__main__":
    app.run()