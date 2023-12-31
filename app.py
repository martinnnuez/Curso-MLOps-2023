import pickle
import pandas as pd
import numpy as np
import gradio as gr
import pathlib
#plt = platform.system()
#if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

with open('lin_reg.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features
    
def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])

def main(PULocationID,DOLocationID,trip_distance):
    """request input, preprocess it and make prediction"""
    input_data = {
    "PULocationID": PULocationID,
    "DOLocationID": DOLocationID,
    "trip_distance": trip_distance
    }
    features = prepare_features(input_data)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return result

#create input and output objects
#input
input1 = gr.inputs.Number()
input2 = gr.inputs.Number()
input3 = gr.inputs.Number()

#output object
output = gr.outputs.Textbox() 

intf = gr.Interface(title = "New York taxi duration prediction",
                    description = "The objective of this project is to predict the duration of a taxi trip in the city of New York.",
                    fn=main, 
                    inputs=[input1,input2,input3], 
                    outputs=[output], 
                    live=True,
                    enable_queue=True
                    )
intf.launch()