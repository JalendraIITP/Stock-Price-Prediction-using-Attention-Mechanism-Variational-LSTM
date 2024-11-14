# Stock Price Prediction using Attention Mechanism Variational LSTM (AMV-LSTM)

This repository provides a robust approach to stock price prediction using an **Attention Mechanism Variational Long Short-Term Memory (AMV-LSTM)** neural network. By integrating the attention mechanism, this model enhances the predictive capabilities of traditional LSTM networks, making it well-suited for time-series forecasting tasks like stock price prediction.

---

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Installation](#installation)
- [Working Procedure](#working-procedure)
- [Results](#results)
- [References](#references)

---

## Introduction

Predicting stock prices is a challenging task due to the volatility and complexity of financial markets. This project leverages an **Attention LSTM model**, which allows the model to focus on relevant past data points, thereby improving prediction accuracy.

---

## Model Architecture

The **Attention Mechanism Variational LSTM (AMV-LSTM)** model architecture comprises:
- **LSTM Layers** Variational LSTM Layers inspired by Peephole LSTM
- **Attention Mechanism** to allow the model to focus on relevant data points in the input sequence
- **Fully Connected Layers** for final stock price prediction
<table>
  <tr>
    <td>
      <img src="https://github.com/JalendraIITP/Stock-Price-Prediction-using-Attention-Mechanism-Variational-LSTM/blob/master/Structure_of_LSTM.png" alt="LSTM" width="600"><br>
      <img src="https://github.com/JalendraIITP/Stock-Price-Prediction-using-Attention-Mechanism-Variational-LSTM/blob/master/Structure_of_VLSTM.png" alt="VLSTM" width="600">
    </td>
    <td>
      <img src="https://github.com/JalendraIITP/Stock-Price-Prediction-using-Attention-Mechanism-Variational-LSTM/blob/master/Structure_of_Attention.png" alt="Attention Block" width="600">
    </td>
  </tr>
</table>

The attention mechanism improves the model’s ability to capture dependencies in the time-series data, essential for accurate stock price predictions.

### Mathematical Equations
- **VLSTM**
  1. **Forget Gate**:
   \[
   f_t = \sigma(W_{x_f} \cdot \tilde{x}_t + W_{h_f} \cdot h_{t-1} + b_f)
   \]

2. **Input Gate**:
   \[
   i_t = (1 - f_t) \odot g_t
   \]

3. **Intermediate Gate**:
   \[
   g_t = \sigma(W_{cg} \odot c_{t-1})
   \]

4. **Cell State Update**:
   \[
   \tilde{c}_t = \tanh(W_{xc} \cdot \tilde{x}_t + W_{hc} \odot h_{t-1} + b_c)
   \]

5. **Cell State**:
   \[
   c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
   \]

6. **Output Gate**:
   \[
   o_t = \sigma(W_{xo} \cdot \tilde{x}_t + W_{ho} \cdot h_{t-1} + b_o)
   \]

7. **Hidden State**:
   \[
   h_t = o_t \odot \tanh(c_t)
   \]
- **Attention**
  8. **Attention Scores**:
   \[
   a_t = V_a \cdot \tanh(W_{ax} \cdot x_t + b_a)
   \]

9. **Probability Distribution**:
   \[
   p_t = \text{Softmax}(a_t)
   \]

10. **Weighted Input**:
    \[
    \tilde{x}_t = p_t \odot x_t
    \]
---
## Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:

- **Python** 3.x
- Libraries:
  - **PyTorch**
  - **NumPy**
  - **Pandas**
  - **Matplotlib**

### Installation

1. **Clone this repository**:

    ```bash
    git clone https://github.com/JalendraIITP/Stock-Price-Prediction-using-Attention-LSTM
    cd Stock-Price-Prediction-using-Attention-LSTM
    ```

2. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Dataset

Select any stock (e.g., AAPL, GOOGL) from Yahoo Finance and define the start and end dates for the data range you wish to use for training.

## Working Procedure

To train and test the model, follow these steps:

1. **Train the Model**

   Run the `train.py` script with the following options:
   - `StockName`: Specify the stock symbol (e.g., AAPL, GOOGL).
   - `startDate` and `endDate`: Define the date range for training data (e.g., 2000-01-01 to 2020-12-31).
   - `Model_Name`: Choose one of the following models:
      - `LSTM` - Standard LSTM
      - `VLSTM` - Variational LSTM
      - `AMVLSTM` - Attention Mechanism Variational LSTM

   Example command:

    ```bash
    python train.py --StockName GOOGL --startDate 2004-08-19 --endDate 2020-12-31 --Model_Name AMVLSTM
    ```

2. **Test the Trained Model**

   Run the `test.py` script to evaluate the model on new data:

    ```bash
    python test.py --StockName GOOGL --startDate 2021-01-01 --endDate 2024-10-31 --Model_Name AMVLSTM
    ```

## Results

<table>
  <tr>
    <th style="border: 1px solid; padding: 8px;">Model</th>
    <th style="border: 1px solid; padding: 8px;">R2 Score</th>
    <th style="border: 1px solid; padding: 8px;">MSE</th>
    <th style="border: 1px solid; padding: 8px;">MAE</th>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">LSTM</td>
    <td style="border: 1px solid; padding: 8px;">0.77856</td>
    <td style="border: 1px solid; padding: 8px;">115.9576</td>
    <td style="border: 1px solid; padding: 8px;">8.6093</td>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">Variational LSTM</td>
    <td style="border: 1px solid; padding: 8px;">0.8166</td>
    <td style="border: 1px solid; padding: 8px;">99.1781</td>
    <td style="border: 1px solid; padding: 8px;">7.8780</td>
  </tr>
  <tr>
    <td style="border: 1px solid; padding: 8px;">AMV-LSTM</td>
    <td style="border: 1px solid; padding: 8px;">0.9779</td>
    <td style="border: 1px solid; padding: 8px;">11.9392</td>
    <td style="border: 1px solid; padding: 8px;">2.6519</td>
  </tr>
</table>

## References

This is the implementation of the paper:
**A Novel Variant of LSTM Stock Prediction Method Incorporating Attention Mechanism**  
_Shuai Sang and Lu Li_

I attempted to implement this paper to gain a better understanding of Attention and LSTM; any feedback on mistakes is welcome.
