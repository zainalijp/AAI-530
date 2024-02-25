## SmartSensory: An IoT Time Series Prediction Model for Behavioral Monitoring  

### Repository Overview

This repository contains smartphone sensor data from 60 distinct user CSV files, merged for comprehensive analysis and performance evaluation across various models in diverse iterations. It showcases our approach to data preparation and manipulation within the "ExtraSensory_Combined_User_Data" framework, which details our data cleaning practices and the assorted model variations explored. While the majority of our models are individually archived, initial models tested during the exploratory data analysis phase are exceptions. For insights into our most effective models and access to the streamlined version of our algorithms and code, please refer to "ExtraSensory_Combined_User_Data_v1.0.ipynb". 

---

### Dataset

The ExtraSensory Dataset is a publicly available dataset that was created in 2015-2016 by Yonatan Vaizman and Katherine Ellis, under the supervision of professor Gert Lanckriet at the University of California, San Diego’s Department of Electrical and Computer Engineering. The collection of this data was facilitated through the ExtraSensory mobile application, created by its founders, allowing users to grant access to smartphone sensor recordings and incorporating a Bluetooth linked smartwatch for additional data support.  
More information regarding the dataset and the ExtraSensory application can be found [here.](http://extrasensory.ucsd.edu/) 
For the context of this research, we refer to the original paper, which can be found [here](https://ieeexplore.ieee.org/document/8090454). 

The original dataset contained 60 user files, each named with a unique user identifier (UUID) and containing a timestamp and a combined total of approximately 300,000 raw data points collected from smartphone and smartwatch sensor readings over a period of time. Additionally, each user file contained 51 boolean labels, which were originally confirmed or denied by users via the ExtraSensory Application and later cleaned to determine correlation between sensor readings and label responses. In our study, we combined all 60 user files and used a small subset of all the available variables based on a 0% missing value threshold, intentionally excluding smartwatch data to broaden user appeal and to minimize the dataset’s noise, size and computational demands for testing purposes. Below is a brief description of the variables for the entire dataset: 

- `timestamp`: This is represented as the standard number of seconds since the epoch.
- `raw_acc`: Raw version of accelerometer from the phone
- `proc_gyro`: Processed version of gyroscope from the phone
- `raw_magnet`: Raw version of magnetometer from the phone
- `watch_acceleration`: Accelerometer from the watch
- `watch_heading`: Heading from the compass on the watch
- `location: location services`: These features were extracted offline for every example from the sequence of latitude-longitude-altitude updates from the example's minute. 
- `location_quick_features`: Calculated on the phone when data was collected. 
- `audio_naive`: Averages and std dev of the 13 MFCCs from the 20 second microphone recording per sample. 
- `discrete: phone-state`: Binary indicators for the state of the phone
- `lf_measurements`: Various sensors that were recorded in low-frequency 
- `label`: Ground truth labels representing the relevance of examples (1 for relevant, 0 for not relevant)
- `label_source`: Where the original labeling came from in the mobile application interface

---

### Repository Structure

```
AAI-530/
│
├── .vscode        # LSTM attempted version training folder  
│
├── LSTM_Model        # LSTM attempted version model folder
│
├── .DS_Store        # LSTM attempted version
│
├── Business_data_processing.0.1.ipynb        # Preprocessing for business dashboard
│
├── ExtraSensory_CNN_LSTM_Model_v1.h5        # Saving attempted CNN LSTM model
│
├── ExtraSensory_CNN_LSTM_Model_v2_bs_20.h5        # Saving attempted CNN LSTM model 
│
├── ExtraSensory_Combined_User_Data_v0.1.ipynb        # Combined data processing: version 1
│
├── ExtraSensory_Combined_User_Data_v0.2.ipynb        # Combined data processing: version 2
│
├── ExtraSensory_Combined_User_Data_v0.3.ipynb        # Combined data processing: version 3
│
├── ExtraSensory_Combined_User_Data_v0.4.ipynb        # Combined data processing: version 4
│
├── ExtraSensory_Combined_User_Data_v0.5.ipynb        # Combined data processing: version 5
│
├── ExtraSensory_Combined_User_Data_v0.6.ipynb        # Combined data processing: version 6 
│
├── ExtraSensory_Combined_User_Data_v0.7.html         # Combined data processing: version 7
│
├── ExtraSensory_Combined_User_Data_v0.7.ipynb        # Combined data processing: version 7
│
├── ExtraSensory_Combined_User_Data_v0.9.ipynb        # Combined data processing: cleaning
│
├── ExtraSensory_Combined_User_Data_v0.91ipynb        # Combined data processing: cleaning
│
├── ExtraSensory_Combined_User_Data_v0.92.ipynb       # Combined data processing: cleaning
│
├──LSTM_next_model.h5        # CNN and LSTM attempted version: parameter testing 
│
├──fifth_try_2.h5        # CNN and LSTM attempted version: parameter testing 
│
├──fifth_try_3.h5        # CNN and LSTM attempted version: parameter testing 
│
└──first_try.h5        # CNN and LSTM attempted version: parameter testing 
│
├──fourth_try_2.h5        # CNN and LSTM attempted version: parameter testing
│
├──requirements.txt        # Attempting to synch requirements between collaborators
│
├──second_try.h5        # CNN and LSTM attempted version: parameter testing 
│
├──subset.csv        # Data subset for testing  
│
├──subset_of_combined_csv_data.csv        # Data subset for testing   
│
├──tf_model.h5        # Tensorflow model attempt 
│
├──tf_model_v2.h5        # Tensorflow model attempt 
│
├──tf_model_v3.h5        # Tensorflow model attempt
│
└──third_try.h5        # CNN and LSTM attempted version: parameter testing
│
├──third_try_2.h5        # CNN and LSTM attempted version: parameter testing
│
```

---

### Project Summary 

In this project, we explored a range of machine learning and deep learning techniques, with an emphasis on forecasting time series, to develop an IoT application aimed at monitoring behavior in critical situations. Our approach involved evaluating our algorithms on a select group of variables, identified by a predefined threshold for missing values, across data from all 60 participants. We achieved notable success by employing classifier chains and Long Short-Term Memory (LSTM) networks for producing predictions of labels based on sensor data. Additionally, we included a comparable linear approach that used these label predictions to forecast sensor input.   

---

### Contributing

For any questions, comments, issues or contributions, please feel free to open an issue on the Github repository. 

---

### Authors

- **Halladay Kinsey**
- **Zain Ali**
- **Hani Jandali**

---

### Clone the Repository

To clone this repository, open your terminal or command prompt and run the following command:

```bash
git clone https://github.com/zainalijp/AAI-530
```
---

### Acknowledgments

We would like to express our gratitude to the UCSD ExtraSensory dataset creators, and to our professors and mentors for their guidance throughout the project.

---

