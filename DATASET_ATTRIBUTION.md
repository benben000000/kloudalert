# DATASET ATTRIBUTION & METEOROLOGICAL DATA CREDITS

This document provides formal credit, attribution, and license documentation for all data sources integrated into the **KloudAlert / PIMCAN-v4 Neural Weather Engine**.

---

## 1. KloudTech SEA Weather Station Telemetry Network
* **Provider**: KloudTech SEA Production API (`api.kloudtechsea.com`)
* **Scope**: 17 Automatic Weather Stations (AWS) deployed across Central Luzon, Bataan, Nueva Ecija, Pampanga, Bulacan, Aurora, and Makati City.
* **Data Captured**: 1-hour and 10-minute continuous telemetry (Air Temperature, Relative Humidity, Barometric Pressure, Precipitation Rate, Wind Speed, Solar Radiation, Romps Heat Index).
* **Attribution Statement**: *"Telemetry data provided by KloudTech SEA Weather Observation Network."*

---

## 2. Himawari-9 Geostationary Satellite Imagery (AHI)
* **Provider**: Japan Meteorological Agency (JMA) & National Institute of Information and Communications Technology (NICT)
* **Endpoint**: JMA / NICT Himawari Science Data API (`himawari8.nict.go.jp`)
* **Scope**: Geostationary Himawari-9 Advanced Himawari Imager (AHI) satellite scans over Bataan / Central Luzon ($14.0^\circ\text{N} - 15.5^\circ\text{N}, 120.0^\circ\text{E} - 121.5^\circ\text{E}$).
* **Bands Used**: 
  - Band 13: Clean Thermal Infrared ($10.4\,\mu\text{m}$) Cloud-Top Brightness Temperature ($T_b$)
  - Band 8: Upper-Level Water Vapor ($6.2\,\mu\text{m}$)
* **Attribution Statement**: *"Satellite telemetry provided by Japan Meteorological Agency (JMA) and NICT Himawari Data Feed."*

---

## 3. RainViewer Doppler Radar Network (DOST-PAGASA Composite)
* **Provider**: RainViewer Global Weather Radar API (`api.rainviewer.com`) & DOST-PAGASA Composite Radar Feed
* **Scope**: Doppler radar reflectivity tiles ($\text{dBZ}$) over Subic, Tagaytay, and Bataan/Luzon regional grids.
* **Conversion Applied**: Marshall-Palmer Radar Reflectivity Relation $Z = 200 \cdot R^{1.6}$.
* **Attribution Statement**: *"Doppler radar reflectivity data provided by RainViewer API using DOST-PAGASA regional radar composites."*

---

## 4. Blitzortung / Limaps.org Lightning Detection Network
* **Provider**: Blitzortung.org & Limaps Community Lightning Detection Network (`data.blitzortung.org` / `limaps.org`)
* **Scope**: Real-time and historical lightning stroke records over Luzon & Bataan bounding box ($12.0^\circ\text{N} - 19.0^\circ\text{N}, 119.5^\circ\text{E} - 124.5^\circ\text{E}$).
* **Features Extracted**: Flash Count Density, Peak Current Amplitude ($\text{kA}$), Positive Polarity Ratios, Convective Severity Score.
* **Attribution Statement**: *"Lightning stroke telemetry provided by Blitzortung.org collaborative lightning detection network."*

---

## 5. Open-Meteo & ECMWF ERA5 Reanalysis
* **Provider**: Open-Meteo API & ECMWF (European Centre for Medium-Range Weather Forecasts)
* **Scope**: Global atmospheric reanalysis and weather model data.
* **Attribution Statement**: *"Historical atmospheric reanalysis data provided by ECMWF ERA5 via Open-Meteo API."*
