# **💧 National Water Resources Intelligence Dashboard**

An advanced, data-driven Streamlit application designed for the comprehensive analysis, visualization, and forecasting of national water resource data. This dashboard integrates Google's Gemini AI to provide intelligent insights, automated data processing, and in-depth reporting for policymakers, researchers, and water resource managers in India.

## **✨ Key Features**

This dashboard is organized into several thematic tabs, each offering a unique set of tools and analyses:

**Core Functionality:**

* **🤖 AI-Powered Data Ingestion:** Automatically maps columns from user-uploaded CSVs to the required schema using Google's Gemini AI, significantly reducing manual data preparation.  
* **⚙️ Manual Fallback Mode:** An AI-disable toggle allows the app to run without an API key, providing a manual interface for column mapping.  
* **📊 Automated Data Quality Reports:** Instantly generates a report on missing values, duplicates, and potential data anomalies upon file upload.  
* **🌍 Interactive Filtering:** Dynamically filter the entire dashboard's data by State, District, River Basin, specific station, or a custom time range.

**Analytical & Visualization Tabs:**

* **🗺️ Unified Map View:**  
  * Visualize groundwater and rainfall station locations across India.  
  * Switch between a "Points of Interest" view and a "Heatmap" to see station density.  
  * Analyze historical water stress levels on the map by year.  
* **📊 At-a-Glance Dashboard:**  
  * View key metrics like average water levels, rainfall, and water quality parameters.  
  * Analyze agency contributions with an interactive donut chart.  
* **⚖️ Policy & Governance:**  
  * Identify regional groundwater stress hotspots by state or river basin.  
  * Define "critical" water levels with a dynamic percentile slider.  
  * View the top 5 most rapidly declining and improving stations.  
  * Generate AI-powered policy briefings and recommendations.  
* **🏛️ Strategic Planning (Single Station):**  
  * Analyze long-term historical trends for a specific station.  
  * Estimate sustainable yield and average annual recharge based on aquifer properties.  
  * Model supply vs. demand scenarios to identify potential water deficits or surpluses.  
* **🔬 Research Hub (Single Station):**  
  * Perform high-accuracy predictive forecasting using a SARIMAX model.  
  * Analyze correlations between different water parameters with AI-driven interpretations.  
  * Export forecast data to CSV.  
* **💧 Advanced Hydrology (Single Station):**  
  * Analyze water level fluctuation, volatility, and long-term trends using moving averages (EWMA).  
  * Conduct seasonal performance analysis by comparing pre- and post-monsoon water levels.  
  * Detect and analyze historical drought events based on custom percentile thresholds.  
* **📋 Generate Full Report:**  
  * Consolidate all key metrics, insights, and forecasts from your current selection into a single view.  
  * Download the consolidated report as a JSON file for easy sharing and record-keeping.

## **🛠️ Tech Stack**

* **Framework:** [Streamlit](https://streamlit.io/) \- For building the interactive web application.  
* **Core Language:** Python  
* **Data Analysis & Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)  
* **Data Visualization:** [Plotly](https://plotly.com/python/), [Pydeck](https://deckgl.readthedocs.io/en/latest/) (for heatmap visualization)  
* **Time-Series Forecasting:** [statsmodels](https://www.statsmodels.org/stable/index.html) (SARIMAX)  
* **Machine Learning:** [scikit-learn](https://scikit-learn.org/stable/) (Linear Regression for trend analysis)  
* **Generative AI:** [Google Gemini API](https://ai.google.dev/)

## **💾 Data Requirements**

To use the dashboard, you need to upload **three separate CSV files**. The application will intelligently try to assign them, but you must confirm the roles.

1. **Groundwater Stations File:** Contains the geographic and administrative details for each groundwater monitoring station.  
   * **Required Columns:** station\_name, latitude, longitude, state\_name, district\_name, agency\_name, basin  
2. **Rainfall Stations File:** Contains the geographic and administrative details for each rainfall monitoring station.  
   * **Required Columns:** station\_name, latitude, longitude, state\_name, district\_name, agency\_name, basin  
3. **Time-Series Data File:** Contains the chronological measurements from all stations (both groundwater and rainfall).  
   * **Required Columns:** station\_name, timestamp, groundwaterlevel\_mbgl, rainfall\_mm, temperature\_c, ph, turbidity\_ntu, tds\_ppm

***Note:*** The column names in your files do not need to match exactly. The AI mapping feature is designed to handle variations (e.g., lat vs latitude, station name vs station\_name). If AI is disabled, you will need to map your columns to these standard names manually in the sidebar.

## **🚀 Getting Started**

Follow these instructions to set up and run the dashboard on your local machine.

### **Prerequisites**

* Python 3.8 or newer  
* A Google Gemini API Key (optional, for AI features)

### **Installation & Setup**

1. **Clone the repository:**  
   git clone \[https://github.com/your-username/water-resources-dashboard.git\](https://github.com/your-username/water-resources-dashboard.git)  
   cd water-resources-dashboard

2. **Create a virtual environment (recommended):**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

3. Install the required libraries:  
   Create a requirements.txt file with the following contents:  
   streamlit  
   pandas  
   plotly  
   google-generativeai  
   statsmodels  
   numpy  
   pydeck  
   scikit-learn

   Then, install them using pip:  
   pip install \-r requirements.txt

4. Set up your Gemini API Key (Optional):  
   The application will prompt you to enter your API key in the sidebar. You can get a key from Google AI Studio.

### **How to Run the Application**

1. Open your terminal and navigate to the project directory.  
2. Run the following command:  
   streamlit run app.py

3. The application will open in a new tab in your default web browser.

## **📖 How to Use**

1. **Upload Data:** Use the sidebar uploader to select your three required CSV files.  
2. **Configure API Key:** If you want to use the AI features, paste your Gemini API key into the input box in the sidebar. Otherwise, toggle "Disable All AI Features" on.  
3. **Confirm File Roles:** Ensure the correct files are assigned as the Groundwater, Rainfall, and Time-Series datasets.  
4. **Map Columns:** Review the AI-generated column mappings or map them manually if AI is disabled.  
5. **Filter Data:** Use the sidebar controls to select the state, district, basin, station, and time period you wish to analyze.  
6. **Explore Tabs:** Navigate through the main tabs at the top of the page to access different dashboards and analytical tools.

## **🤝 Contributing**

Contributions are welcome\! If you have suggestions for improvements or want to add new features, please feel free to fork the repository, make your changes, and submit a pull request.

## **📄 License**

This project is licensed under the MIT License. See the LICENSE file for more details.

*This dashboard was created to demonstrate the power of combining modern data science tools and generative AI for tackling complex environmental challenges.*