# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess


data = pd.read_csv("tsa.csv")

# Convert Date to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Set frequency to business days
data = data.asfreq('B')

# Use Closing Price
X = data['Close']

# Plot closing prices
plt.figure(figsize=(12,6))
plt.plot(X)
plt.title("Stock Closing Price Over Time")
plt.show()

# ACF and PACF
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X.dropna(), lags=len(X)//20, ax=plt.gca())
plt.title("ACF of Closing Price")

plt.subplot(2,1,2)
plot_pacf(X.dropna(), lags=len(X)//20, ax=plt.gca())
plt.title("PACF of Closing Price")
plt.tight_layout()
plt.show()

# Fit ARMA(1,1)
arma11_model = ARIMA(X, order=(1,0,1), enforce_stationarity=False, enforce_invertibility=False).fit()
phi_arma11 = arma11_model.params['ar.L1']
theta_arma11 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi_arma11])
ma1 = np.array([1, theta_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=500)

plt.plot(ARMA_1)
plt.title("Simulated ARMA(1,1) Process")
plt.show()

plot_acf(ARMA_1)
plt.title("ARMA(1,1) ACF")
plt.show()
plot_pacf(ARMA_1)
plt.title("ARMA(1,1) PACF")
plt.show()

# Fit ARMA(2,2)
arma22_model = ARIMA(X, order=(2,0,2), enforce_stationarity=False, enforce_invertibility=False).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=500)

plt.plot(ARMA_2)
plt.title("Simulated ARMA(2,2) Process")
plt.show()

plot_acf(ARMA_2)
plt.title("ARMA(2,2) ACF")
plt.show()
plot_pacf(ARMA_2)
plt.title("ARMA(2,2) PACF")
plt.show()

# Forecast next 30 days with ARMA(2,2)
forecast = arma22_model.get_forecast(steps=30)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

plt.figure(figsize=(12,6))
plt.plot(X, label='Observed')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
plt.title("30-Day Forecast of Closing Prices")
plt.legend()
plt.show()
```
OUTPUT:
Original data:
<img width="1079" height="543" alt="image" src="https://github.com/user-attachments/assets/66c5732e-10aa-4c32-a6e5-1c54314d8add" />
Autocorrelation:
<img width="1081" height="225" alt="image" src="https://github.com/user-attachments/assets/99596b2f-cc73-4425-a67d-c1d3f12d4960" />


Partial Autocorrelation
<img width="1079" height="260" alt="image" src="https://github.com/user-attachments/assets/992a3719-8595-431b-90f6-2dc8a88504bd" />

SIMULATED ARMA(1,1) PROCESS:
<img width="588" height="427" alt="image" src="https://github.com/user-attachments/assets/3a6c54aa-2138-43d2-b776-379925f2f597" />

Autocorrelation:
<img width="600" height="439" alt="image" src="https://github.com/user-attachments/assets/b17f31b1-bab1-4e07-94a1-83237e1ff1c9" />

Partial Autocorrelation:
<img width="692" height="439" alt="image" src="https://github.com/user-attachments/assets/cba8f4a7-eba2-40f9-bee8-accbda4c2671" />

SIMULATED ARMA(2,2) PROCESS:
<img width="663" height="437" alt="image" src="https://github.com/user-attachments/assets/3f27dd5a-3330-49c8-b521-aa4146defcda" />



Partial Autocorrelation
<img width="636" height="426" alt="image" src="https://github.com/user-attachments/assets/7718dc45-a6e7-44bd-ad1c-c8c07516ed5c" />



Autocorrelation
<img width="654" height="435" alt="image" src="https://github.com/user-attachments/assets/59a6530a-f6bb-4de3-986f-9d650acc0641" />


RESULT:
Thus, a python program is created to fir ARMA Model successfully.
