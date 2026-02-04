
import gradio as gr
import pandas as pd
from xgboost import XGBRegressor
import os

MODEL_PATH = "xgb_best_model.json"

model = XGBRegressor()
model.load_model(MODEL_PATH)

def predict_dv(penetration, softening_point, ductility, concentration, time):
    df = pd.DataFrame(
        [[penetration, softening_point, ductility, concentration, time]],
        columns=["Penetration", "Softening point", "Ductility", "Concentration", "Time"]
    )
    return float(model.predict(df)[0])

demo = gr.Interface(
    fn=predict_dv,
    inputs=[
        gr.Number(label="Penetration"),
        gr.Number(label="Softening point"),
        gr.Number(label="Ductility"),
        gr.Number(label="Concentration"),
        gr.Number(label="Time")
    ],
    outputs=gr.Number(label="Predicted DV"),
    title="SGA–XGBoost GUI"
)

if __name__ == "__main__":
    demo.launch()
