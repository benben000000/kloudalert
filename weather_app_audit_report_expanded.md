# Weather Anomaly Prediction App: Architecture Audit & Proof-of-Concept Guide (Expanded)

## Executive Summary

This expanded report provides a comprehensive audit of the proposed architecture for a weather anomaly prediction mobile app. The app aims to predict heavy rainfall, light rainfall, lightning barrages, heat index spikes, and heat waves 15–45 minutes in advance using a hybrid architecture combining Liquid Neural Networks (LNNs), Vector Quantized Transformers (VQ-T), and edge computing.

**Overall Verdict:** The architecture is **highly plausible and technically feasible**, but it leans toward being **over-engineered** for a proof-of-concept (PoC). While the combination of VQ-T for tokenization and LNNs for continuous-time adaptation is theoretically sound, implementing both simultaneously in a single mobile deployment pipeline will introduce significant complexity. However, the core premise—predicting localized weather anomalies with a 15–45 minute lead time—is entirely viable, as evidenced by existing commercial products like Flash Weather AI [1].

This update specifically addresses your requirements for **location-aware, multi-station alert routing** and the **"when will it end?" duration timer** feature.

## Architecture Component Assessment

### 1. The Core Hybrid Model: VQ-T7 Tokenization + Liquid Neural Networks

The proposed architecture uses a Vector Quantized Transformer (VQ-T) to process raw data streams into tokens, followed by a Liquid Neural Network (LNN) for continuous-time inference.

**Plausibility and Viability:**
This hybrid approach is at the cutting edge of time-series forecasting. The VQ-TR (Vector Quantized Transformer) architecture maps large sequences to a discrete set of latent representations, allowing for linear time complexity instead of the quadratic complexity typical of standard Transformers [2]. This is highly viable for compressing high-frequency sensor data (like lightning pulses) into manageable tokens. LNNs, specifically Liquid Time-Constant (LTC) networks, are exceptional for continuous-time series data. They possess minimal resource requirements and can adapt to new inputs after training, making them ideal for edge deployment [3].

**Feasibility and Over-engineering:**
While scientifically sound, combining these two complex architectures is over-engineered for an initial PoC. 
*   **The Fix:** For the PoC, bypass the VQ-T7 Tokenizer and the "DenseNet Feature Extraction" layer. Instead, feed the normalized sensor data (Temperature, Humidity, Pressure, Lightning pulses) directly into a simplified LNN or a standard Long Short-Term Memory (LSTM) network. LSTMs are heavily documented for precipitation nowcasting (predicting rain 15-45 minutes ahead) and are computationally cheaper to train than a custom VQ-Transformer [4].

### 2. Location-Aware Multi-Station Alert Routing

You noted that alerts should be based on the user's location relative to multiple AWS (Automated Weather Station) stations.

**Plausibility and Viability:**
This is a standard and highly viable approach in modern weather alerting systems. Companies like Earth Networks and Tempest use this exact method, allowing users to set a custom radius for alerts based on their GPS coordinates [5]. 

**Feasibility:**
*   **The Math:** The Haversine formula is perfectly suited for calculating the great-circle distance between the user's moving GPS coordinates and the fixed coordinates of the AWS stations [6].
*   **The Logic:** When the user is on their way home, the app should continuously calculate the distance to the nearest 3-5 AWS stations.
*   **Data Weighting:** Instead of just picking the absolute closest station, the app should use **Inverse Distance Weighting (IDW)**. If a storm is brewing, the stations closest to the user's path will have a higher impact on the LNN's input vector.
*   **Implementation:** In Flutter or React Native, this can be handled entirely on-device using the Haversine formula, ensuring low latency even if the user is moving through areas with poor cell service.

### 3. The "When Will It End?" Duration Timer

You requested a timer that determines the duration of the current weather anomaly (e.g., predicting when heavy rain will stop).

**Plausibility and Viability:**
Predicting the duration of an anomaly is a more complex task than simply predicting its onset, but it is entirely viable using Convolutional LSTM (ConvLSTM) networks or extended LNNs. The seminal work on precipitation nowcasting by Shi et al. (2015) demonstrated that spatiotemporal sequence forecasting can effectively predict the evolution and dissipation of rain clouds [7].

**Feasibility:**
*   **The Model Output:** Instead of outputting a single "Rain starts in 15 mins" prediction, the LNN must be trained to output a **probability curve** over the next 45-60 minutes (e.g., 18 time steps of 2.5 minutes each).
*   **Calculating Duration:** The app logic will look at this probability curve. If the model predicts high rain probability from `t+5` to `t+25`, the app calculates the duration as 20 minutes.
*   **The User Interface:** This translates directly to the requested countdown timer: *"Heavy rain expected to end in 23 minutes."*
*   **Updates:** This timer should update every 2-5 minutes as the LNN receives new real-time sensor data, recalculating the expected end time.

### 4. The "Critical Fix": Fusing VQ-RES of LNNs with AWS Data

The architecture diagram highlights a "Critical Fix" that fuses the VQ-RES of LNNs with AWS environmental data to classify Dry Lightning and Heavy Rain.

**Plausibility and Viability:**
Fusing discrete vector representations with continuous physical measurements is a robust multimodal approach. It allows the model to recognize complex patterns (like a sudden drop in pressure combined with specific lightning pulse frequencies) that traditional threshold-based alerts miss.

**Feasibility:**
Building a custom fusion layer is difficult. For the PoC, utilize **feature concatenation**. Simply pass the sensor readings alongside derived features (e.g., rate of change in temperature) into a single input vector for the LNN. This achieves the same goal of multimodal fusion without requiring complex architectural modifications.

## Data Sources and APIs

To make this system work, high-fidelity, low-latency data is required.

| Data Type | Recommended API/Source | Granularity | Cost/Tier |
| :--- | :--- | :--- | :--- |
| **General Weather** | Open-Meteo API | 15-minute intervals | Free (Non-commercial) [8] |
| **Minute-by-Minute Rain** | Google Maps Weather API / WeatherAPI | 1-minute intervals | Paid Tier [9] |
| **Lightning Strikes** | Blitzortung.org API / Xweather | Real-time / Near real-time | Free / Paid |
| **Historical Training** | Open-Meteo Historical API | Hourly/Daily | Free (Non-commercial) |

*Note: For a PoC, Open-Meteo's 15-minute historical data is sufficient to train the LNN to predict the next 15-45 minute block. For the live app, minute-by-minute APIs will be necessary to beat the weather.*

## Proof-of-Concept (PoC) Roadmap

To validate the concept without getting bogged down in VQ-T7 implementation details, follow this phased approach:

### Phase 1: Data Ingestion & Baseline Model (Weeks 1-2)
1.  **Data Collection:** Use the Open-Meteo Historical API to download 3 years of hourly weather data (Temperature, Humidity, Pressure, Rainfall) for a specific geographic region prone to storms.
2.  **Baseline Training:** Train a standard LSTM network using PyTorch or TensorFlow to predict the `precipitation` value for the next 3 time steps (45 minutes ahead).
3.  **Lightning Proxy:** If lightning data is hard to source historically, use `pressure_msl` (Mean Sea Level Pressure) and `temperature_2m` as proxies. Rapid drops in pressure and spikes in temperature are strong predictors of incoming storms.

### Phase 2: LNN Integration & Edge Testing (Weeks 3-4)
1.  **Switch to LNN:** Replace the LSTM with a Liquid Time-Constant (LTC) network using the `ncps` library.
2.  **Duration Logic:** Modify the LNN to output a sequence of 18 probabilities (for the next 45 minutes). Write a script that calculates the "expected duration" of the rain based on this sequence.
3.  **Quantization:** Convert the trained LNN model to TFLite format.
4.  **Edge Test:** Write a simple Python script simulating an edge device that takes a stream of numbers and runs the TFLite model to output a probability score for "Heavy Rain in 30 mins" and "Rain ends in 20 mins".

### Phase 3: Mobile App Skeleton (Weeks 5-6)
1.  **Framework:** Use Flutter or React Native.
2.  **Location:** Implement the Haversine formula to calculate the distance between the user's GPS coordinates and a simulated "storm center" (or multiple AWS station coordinates).
3.  **Alert Logic:** If the TFLite model outputs a probability > 0.8, trigger a local push notification using Firebase Cloud Messaging (FCM) or Expo Notifications.
4.  **Timer UI:** Display the calculated duration as a countdown timer on the app's main screen.

## Conclusion

The architecture is brilliant but ambitious. The integration of VQ-Transformers with Liquid Neural Networks represents a novel approach to nowcasting. However, to ensure it is viable and plausible as a Proof-of-Concept, it is highly recommended to strip away the Vector Quantization layer initially. Focus entirely on training an LNN on historical sensor data to predict a 45-minute precipitation window and its duration. If the LNN succeeds where traditional LSTMs fail (due to irregular sampling or changing dynamics), you will have proven the core value proposition of the app.

## References

[1] Flash Weather AI. "Home - Flash Weather AI." https://flashweather.ai/
[2] Rasul, K., et al. "VQ-TR: Vector Quantized Attention for Time Series Forecasting." ICLR 2024. https://openreview.net/pdf?id=IxpTsFS7mh
[3] Hasani, R., et al. "Closed-form continuous-time neural networks." Nature Machine Intelligence, 2022. https://www.nature.com/articles/s42256-022-00556-7
[4] Viteri López, A.S. "TITAN-LSTM: A Weather Radar Nowcasting Tool." Journal of Geophysical Research, 2026. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2026JH001337
[5] Earth Networks. "What Lightning Alerts Should I Use?" https://www.earthnetworks.com/what-lightning-alerts-should-i-use/
[6] Product Teacher. "The Haversine Formula for Geospatial Distances." https://www.productteacher.com/quick-product-tips/haversine-formula-for-product-teams
[7] Shi, X., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." NeurIPS 2015. https://proceedings.neurips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html
[8] Open-Meteo. "Open-Meteo.com: Free Open-Source Weather API." https://open-meteo.com/
[9] Google for Developers. "Get minute forecast (Experimental) | Weather API." https://developers.google.com/maps/documentation/weather/minute-forecast
