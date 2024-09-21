# Analysis and Forecasting of Carbon Emission in SAARC Countries using Attention-based LSTM

## Abstract

Climate change and global warming are urgent environmental issues demanding immediate action to safeguard future generations. The major contributor to the greenhouse effect, carbon dioxide (CO2), primarily originates from industrial and transportation fossil fuel combustion. International agreements, like the Paris Agreement, call for a 30-35% reduction in CO2 emissions compared to 2005 levels. This research aims to predict CO2 emissions and raise awareness among SAARC nations and governments about the increasing trend. We introduce a novel predictive framework using Attention-based Long Short-Term Memory (A-LSTM) for CO2 emissions analysis. The Attention mechanism assigns variable weights to input data, facilitating indirect connections between LSTM outputs and pertinent inputs. This enhances resource allocation in the A-LSTM model, overcoming computational constraints. We integrate input parameters encompassing CO2 emissions from land-use changes, oil, natural gas, and coal combustion to forecast CO2 emissions and correlate them with population and per capita GDP. 

Our comparative analysis conclusively demonstrates the superior performance of A-LSTM models over baseline LSTM models when applied to the CO2 emission dataset sourced from the Our World in Data (OWID) and World Bank Indicator database. Specifically, the LSTM model registers a MAPE of 24.968 and an RMSE of 0.34, whereas the Attention-based LSTM model showcases a marked improvement of 57% with a considerably lower MAPE of 10.5902 and an RMSE of 0.107.

The proposed A-LSTM pipeline is:


![image](https://github.com/user-attachments/assets/0ade88ea-c192-415c-bcbd-c4f7f3bd4843)

---

## Project Directory Structure

```
CO2-PREDICTION-SAARC-COUNTRIES/
│
├── Analysis Data/
│   ├── cumulative-co2-emissions-all-countries.csv
│   ├── cumulative-co2-emissions-SAARC.csv
│   ├── GDP_data_yearwise_all_countries.csv
│   ├── input_data_analysis.xlsx
│   └── owid-co2-data-raw-saarc.csv
│
├── Input Data/
│   ├── final_cleaned_data_yearly.csv
│   └── final_input_data_clean_monthly.csv
│
├── model1/
│
├── Results/
│   ├── Comparative study/
│   ├── LSTM/
│   └── LSTM_attention/
│
├── .gitattributes
├── Analysis_and_Forecasting_of_Carbon_Emission_in_SAARC.ipynb
├── Co2_project_cleaning.ipynb
├── Co2_project_training-wotemp.ipynb
├── Co2_project_training.ipynb
├── presentation.pptx
└── selected_parameters_description.csv
```

---

## Setup and Requirements

The project requires Python 3.7 or later and the following Python packages:

- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Keras**
- **TensorFlow**
- **scikit-learn**

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include:

```
numpy
pandas
matplotlib
seaborn
keras
tensorflow
scikit-learn
```

---

## How to Run the Code

1. **Data Preprocessing:**

   `Co2_project_cleaning.ipynb` was used to clean and preprocess the dataset. This step includes parsing dates, resampling from yearwise values to monthwise, converting dates to `datetime` format, etc. Final preprocessed data has been stored in `final_input_data_clean_monthly.csv` for direct use. 

2. **Model Training:**

   To reproduce the results, open and execute `Co2_project_training-wotemp.ipynb`. This notebook contains the code for training both the baseline LSTM model and the Attention-based LSTM model. You can modify hyperparameters such as the learning rate, batch size, and number of epochs to experiment with different training configurations.

3. **Results and Evaluation:**

   The results folder contains detailed output for both the LSTM and A-LSTM models, including comparative studies, training loss, validation loss, and accuracy plots. 

   - **Mean Absolute Error (MAE)**
   - **Mean Squared Error (MSE)**
   - **Mean Absolute Percentage Error (MAPE)**
   - **Root Mean Square Error (RMSE)**

   These metrics are evaluated in both the training and validation datasets, and they provide insights into the model performance.

The steps are presented in the following diagram:

![image](https://github.com/user-attachments/assets/43c61277-5b2a-4ee6-8702-ab5a82802291)

---

## Results Summary

### Key Metrics:

1. **Mean Absolute Error (MAE):**
   The MAE indicates how much inaccuracy is present in the predictions on average. A lower MAE reflects a more accurate model.
   \[
   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y_i}|
   \]

2. **Mean Squared Error (MSE):**
   The MSE penalizes larger errors and is useful when you want to heavily weigh larger deviations between predicted and actual values.
   \[
   MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2
   \]

3. **Mean Absolute Percentage Error (MAPE):**
   MAPE expresses forecast errors as a percentage, offering a more interpretable metric for how far off predictions are from actual values.
   \[
   MAPE = \frac{100%}{n} \sum_{i=1}^{n} \frac{|y_i - \hat{y_i}|}{y_i}
   \]

4. **Root Mean Squared Error (RMSE):**
   RMSE provides a measure of how much prediction errors vary. It’s a widely used metric in forecasting applications to understand prediction uncertainty.
   \[
   RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y_i})^2}
   \]

### Comparative Study:

The Attention-based LSTM (A-LSTM) significantly outperforms the baseline LSTM model:
- LSTM Model: **MAPE** = 24.968, **RMSE** = 0.34
- A-LSTM Model: **MAPE** = 10.5902, **RMSE** = 0.107

This highlights a 57% improvement in MAPE for the A-LSTM model, showcasing its superior forecasting capability.

---

## Conclusion

In conclusion, our study explored the critical task of forecasting carbon dioxide (CO2) emissions to address the pressing issues of climate change and global warming. The proposed Attention-based Long Short-Term Memory (A-LSTM) model significantly outperformed the traditional LSTM model, achieving remarkable improvements in predictive accuracy. 
These results bear noteworthy implications, particularly for predictive modeling in the context of CO2 emissions and sustainable development. The incorporation of attention mechanisms within LSTM architectures opens up exciting avenues for future research in forecasting other climate-related variables and enhancing resource allocation in machine learning models.

---

## Future Research Directions

This study provides a solid foundation for expanding research in several key areas:
- **Incorporation of more diverse climate variables** such as methane (CH4) emissions, and industrial pollution data.
- **Expanding the predictive model to other regions** beyond SAARC countries.
- **Exploring different neural architectures** such as transformers, which may further enhance predictive accuracy.
  
--- 

## Contact Information

For any queries, please reach out at:  `harshit_2101mc20@iitp.ac.in`
