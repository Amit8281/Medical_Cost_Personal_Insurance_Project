#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
import pandas as pd
import gradio as gr

# Load the trained model
model = pickle.load(open('XGBRegressor.pkl', 'rb'))

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # Prepare the input data as a DataFrame
    data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # Perform the prediction
    prediction = model.predict(data)
    return prediction

# Create the input components
input_components = [
    gr.inputs.Number(label="Age"),
    gr.inputs.Number(label="Sex : (0=Female, 1=Male)"),
    gr.inputs.Number(label="BMI"),
    gr.inputs.Number(label="Number of Children"),
    gr.inputs.Number(label="Smoker : (0 = NO, 1 = YES)"),
    gr.inputs.Number(label="Region : (0=northeast, 1=northwest, 2=southeast, 3=southwest)")
]

# Create the interface
interface = gr.Interface(
    fn=predict_insurance_cost,
    inputs=input_components,
    outputs="number",
    title="Insurance Cost Predictor",
    description="Predict whether the insurance cost will be high or low based on the given inputs."
)

# Launch the interface
interface.launch()


# In[ ]:





# In[ ]:




