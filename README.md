## SmartSensory: An IoT Time Series Prediction Model for Behavioral Monitoring  

### Repository Overview

This repository contains smartphone sensor data from 60 distinct user CSV files, merged for comprehensive analysis and performance evaluation across various models in diverse iterations. It showcases our approach to data preparation and manipulation within the "ExtraSensory_Combined_User_Data" framework, which details our data cleaning practices and the assorted model variations explored. While the majority of our models are individually archived, initial models tested during the exploratory data analysis phase are exceptions.  

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
├── Business_data_processing.0.1.ipynb
# Jupyter notebook for preprocessing data for business analytics dashboard. It likely contains data cleaning, transformation, and preparation steps to facilitate business insight generation.

├── Data Subsets
# Contains various CSV files that are subsets of the main dataset. These files are used for different aspects of the analysis, such as model predictions, feature extraction, and more detailed analyses on smaller, more manageable portions of the data.

├── ExtraSensory_Combined_User_Data_v1.0.ipynb
# The main Jupyter notebook for the project, containing the latest and most refined version of the data processing and machine learning models. It likely includes comprehensive data analysis, model training, and evaluation.

├── Older Version & Unused Code
# A folder that stores older versions of the data processing notebooks and possibly experimental code that was not used in the final version. This serves as an archive for tracking the project's progression and storing potential alternative approaches.

├── README.md
# The README file provides an overview of the project, including a description of its purpose, structure, and instructions on how to navigate and use the repository.

├── Saved Models
# This folder contains the saved machine learning models, specifically the weights and architecture of neural networks like CNN and LSTM. These files can be loaded to replicate the results or to continue training without starting from scratch.

└── requirements.txt
# A text file listing all the Python libraries and their versions required to run the notebooks and scripts in the repository. Ensures consistency and ease of setup for new users or environments.

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

### Dashboard

To view the visual insights generated by our data processing and predictive models, please refer to the links below. 


Predictive insights for individual users can be found [here](https://public.tableau.com/app/profile/zain.ali5503/viz/UserView_17086251674130/FullDashboard?publish=yes) 

Collective dataset and business analytic insights can be found [here](https://public.tableau.com/app/profile/halladay.kinsey/viz/AAI530FinalProjectBusinessAnalytics/Dashboard1)
