# Plant 🌱 Disease 🐛 Detection 🔎

Plant Disease Detection is an innovative machine learning project that harnesses the power of Convolutional Neural Networks (CNN) and deep learning techniques to identify and classify diseases in plants. The primary objective is to offer farmers and agricultural experts a valuable tool for swift plant health diagnosis, facilitating timely intervention and minimizing the risk of crop loss.

Project Structure 

The project comprises essential components:

- `Plant_Disease_Detection.ipynb`: Jupyter Notebook with the code for model training.
- `main_app.py`: Streamlit web application for plant disease prediction.
- `plant_disease_model.h5`: Pre-trained model weights.
- `requirements.txt`: List of necessary Python packages.

## Installation 🚀

To run the project locally, follow these steps:


2. Navigate to the project directory:

```bash
cd Plant-Disease-Detection
```

3. Install the required packages:


pip install -r requirements.txt


4. Run the Streamlit web application:


streamlit run main_app.py


## Usage 🌿

Once the application is running, open your web browser and navigate to [http://localhost:8501](http://localhost:8501). Upload an image of a plant leaf, and the system will predict if it is affected by any disease.

## Model Training 🧠

The model was trained using the `Plant_Disease_Detection.ipynb` notebook. It employs a Convolutional Neural Network architecture to classify plant images into different disease categories. The trained model weights are saved in `plant_disease_model.h5`.

## Web Application 🌐

The web application (`main_app.py`) empowers users to interact with the trained model. Upload plant images, and the application provides real-time predictions regarding the health of the plant.

 Requirements 🛠️

- Keras==2.8.0
- numpy==1.21.4
- streamlit==1.18.0
- opencv-python-headless==4.5.3
- tensorflow==2.7.0

## final output images
<img width="1920" height="900" alt="before final output result " src="https://github.com/user-attachments/assets/207bceb5-049a-4e6c-9437-ef2c69259f3d" />
<br><br><br><br>
<img width="1908" height="916" alt="final output result ss" src="https://github.com/user-attachments/assets/9db131d2-9a9b-4542-831a-a54fd5f6259b" />
