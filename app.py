import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="Hoodvisor",
    page_icon="🔒",
    layout="centered"
)

# Load and prepare the real data
@st.cache_data
def load_data():
    data = pd.read_csv('Street_Data_with_coordinates.csv')
    # Rename columns to match our needs
    data = data.rename(columns={
        'lat': 'latitude',
        'lng': 'longitude',
        'SafetyScore': 'safety_score'
    })
    return data

# Train the model
@st.cache_resource
def train_model():
    # Load real data
    data = load_data()
    
    # Prepare features and target
    X = data[['latitude', 'longitude']]
    y = data['safety_score']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# Main application
def main():
    st.title("🔒 HoodVisor")
    st.write("Click on the map to select a location and predict its safety score.")
    
    # Load data and model
    data = load_data()
    model, scaler = train_model()
    
    # Manual coordinate input first
    st.markdown('<div class="coordinate-input" style="margin-bottom: 0;">', unsafe_allow_html=True)
    st.markdown('<h2>📍 Enter Coordinates Manually</h2>', unsafe_allow_html=True)
    
    # Add input validation ranges
    min_lat, max_lat = data['latitude'].min(), data['latitude'].max()
    min_lng, max_lng = data['longitude'].min(), data['longitude'].max()
    
    col1, col2 = st.columns(2)
    with col1:
        input_latitude = st.number_input(
            "Latitude",
            value=data['latitude'].mean(),
            min_value=min_lat,
            max_value=max_lat,
            format="%.6f",
            help=f"Enter latitude between {min_lat:.6f} and {max_lat:.6f}"
        )
    with col2:
        input_longitude = st.number_input(
            "Longitude",
            value=data['longitude'].mean(),
            min_value=min_lng,
            max_value=max_lng,
            format="%.6f",
            help=f"Enter longitude between {min_lng:.6f} and {max_lng:.6f}"
        )
    
    manual_predict = st.button("🎯 Predict Safety Score", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Map section with black styling moved after coordinate input
    st.markdown("""
        <div class="subheader" style="background-color: black; border-left: 5px solid black; margin-bottom: 0;">
            <h2 style="color: white; margin-bottom: 0;">🗺️ Select Location on Map</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Map container with styling
    st.markdown('<div class="map-container" style="margin: 0;">', unsafe_allow_html=True)
    # Create a Folium map centered on the data
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=13)
    
    # Add coordinates popup
    m.add_child(folium.LatLngPopup())
    
    # Create a custom click handler that removes previous markers
    js = """
    var markers = [];
    function handleClick(e) {
        // Clear all existing markers
        markers.forEach(marker => map.removeLayer(marker));
        markers = [];
        
        // Add new marker
        var marker = L.marker(e.latlng).addTo(map);
        markers.push(marker);
        marker.bindPopup('Selected Location').openPopup();
    }
    map.on('click', handleClick);
    """
    
    # Add the custom JavaScript to the map
    m.get_root().script.add_child(folium.Element(js))
    
    map_data = st_folium(m, height=400, width=700)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if map_data['last_clicked'] or manual_predict:
        # Get coordinates either from map click or manual input
        if map_data['last_clicked']:
            latitude = map_data['last_clicked']['lat']
            longitude = map_data['last_clicked']['lng']
        else:
            latitude = input_latitude
            longitude = input_longitude
        
        # Prepare input data
        input_data = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude]
        })
        
        # Scale input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display prediction
        st.subheader("Prediction Results")
        
        # Create a color gradient based on the safety score
        normalized_score = (prediction - data['safety_score'].min()) / (data['safety_score'].max() - data['safety_score'].min())
        
        # Calculate RGB values for a red to green gradient
        red = int(255 * (1 - normalized_score))
        green = int(255 * normalized_score)
        color = f"linear-gradient(135deg, rgba({red}, {green}, 0, 0.9), rgba({red}, {green}, 0, 0.7))"
        
        st.markdown(
            f"""
            <div class="prediction-card" style="
                background: {color};
                padding: 20px;
                border-radius: 15px;
                width: 80%;
                margin: 20px auto;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 5px;
            ">
                <p style="color: white; opacity: 0.9; margin: 0; font-size: 20px;">Safety Score:</p>
                <h1 style="color: white; font-size: 36px; margin: 0; font-weight: bold;">{prediction:.2f}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display interpretation with enhanced styling
        max_score = data['safety_score'].max()
        if prediction >= max_score * 0.8:
            safety_level = "Very Safe"
            level_color = "#28a745"  # Dark green
        elif prediction >= max_score * 0.6:
            safety_level = "Safe"
            level_color = "#5cb85c"  # Light green
        elif prediction >= max_score * 0.4:
            safety_level = "Moderate"
            level_color = "#ffc107"  # Yellow
        elif prediction >= max_score * 0.2:
            safety_level = "Unsafe"
            level_color = "#dc3545"  # Light red
        else:
            safety_level = "Very Unsafe"
            level_color = "#c41e3a"  # Dark red
            
        st.markdown(
            f"""
            <div class="safety-level" style="
                background-color: {level_color}; 
                color: white;
                text-align: center;
                width: 80%;
                margin: 10px auto;
                padding: 15px;
                border-radius: 10px;
                font-size: 20px;
            ">
                Safety Level: <strong>{safety_level}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Footer only
    footer_html = """    
        <div style="position: fixed; left: 50%; bottom: 20px; transform: translateX(-50%); 
                    background-color: transparent; text-align: center; padding: 10px;">
            Developed with ❤️ by Pavi, Vini and Reghu
        </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 