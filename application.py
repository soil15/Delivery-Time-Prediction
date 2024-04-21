from flask import Flask, request, render_template, jsonify
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.pipelines.prediction_pipeline import CustomData, PredictionPipeline
from src.logger import logging

application = Flask(__name__, template_folder='templates')

app = application


@app.route("/")
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        # return render_template("form.html")

        return render_template("form.html")

    else:
        
        data = CustomData(
            Delivery_person_Age = int(request.form.get('Age')),
            Order_Date  = request.form.get('Order_Date'),
            Delivery_person_Ratings = float(request.form.get('Rating')),
            Time_Orderd = request.form.get('Time_Orderd'),
            Time_Order_picked = request.form.get('Time_order_picked'),
            Weather_conditions = request.form.get('Weather_conditions'),
            Road_traffic_density = request.form.get('Road_traffic_density'),
            Vehicle_condition = int(request.form.get('v_condition')),
            Type_of_order = request.form.get('Type_of_order'),
            Type_of_vehicle = request.form.get('Type_of_vehicle'),
            multiple_deliveries = float(request.form.get('Multiple_deliveries')),
            Festival = request.form.get('Festival'),
            City = request.form.get('City')
            )
        
        df = data.get_custom_data_as_df()
        
        pred_pipeline_obj = PredictionPipeline()

        logging.info('Input DataFrame columns : {}'.format(list(df.columns)))

        result = pred_pipeline_obj.predict(df)
        
        print('Result => {}'.format(result))

        return (render_template('result.html', final_result=result))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
    
