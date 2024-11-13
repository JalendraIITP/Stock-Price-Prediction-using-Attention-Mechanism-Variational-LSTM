# Stock Price Prediction using Attention Mechanism Variational LSTM (AMV-LSTM)

This repository provides a robust approach to stock price prediction using an **Attention Mechanism Variational Long Short-Term Memory (AMV-LSTM)** neural network. By integrating the attention mechanism, this model enhances the predictive capabilities of traditional LSTM networks, making it well-suited for time-series forecasting tasks like stock price prediction.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Installation](#installation)
- [Working Procedure](#working-procedure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)

---

## Introduction

Predicting stock prices is a challenging task due to the volatility and complexity of financial markets. This project leverages an **Attention LSTM model**, which allows the model to focus on relevant past data points, thereby improving prediction accuracy.

## Features

- **Attention Mechanism**: Enhances the model’s ability to focus on critical historical data points for better predictions.
- **LSTM Variants**: Supports standard LSTM, Variational LSTM (VLSTM), and Attention Mechanism Variational LSTM (AMV-LSTM).
- **Visualization Tools**: Tools for visualizing stock price trends and model predictions.
- **User-Friendly**: Comprehensive scripts for training, evaluating, and visualizing the model’s performance.

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

---

## Model Architecture

The **Attention Mechanism Variational LSTM (AMV-LSTM)** model architecture comprises:
- **LSTM Layers** Variational LSTM Layers inspired by Peephole LSTM
- **Attention Mechanism** to allow the model to focus on relevant data points in the input sequence
- **Fully Connected Layers** for final stock price prediction

The attention mechanism improves the model’s ability to capture dependencies in the time-series data, essential for accurate stock price predictions.

---

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
