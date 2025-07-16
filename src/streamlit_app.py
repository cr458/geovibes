"""Streamlit-based user interface for GeoVibes."""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Point
import json
import os
import io
import shapely

from ui import GeoVibesLogic
from ui_config import UIConstants, BasemapConfig, LayerStyles

st.set_page_config(layout="wide")

# --- State Management ---

def initialize_state():
    """Initialize the GeoVibesLogic class and other session state variables."""
    if 'logic' not in st.session_state:
        # TODO: Make this configurable from the UI
        duckdb_directory = "data" 
        try:
            st.session_state.logic = GeoVibesLogic(
                duckdb_directory=duckdb_directory,
                verbose=True
            )
        except (FileNotFoundError, ValueError) as e:
            st.error(f"Failed to initialize GeoVibes. No database found in `{duckdb_directory}` directory.")
            st.stop()
            
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
        
    if 'current_basemap' not in st.session_state:
        st.session_state.current_basemap = 'MAPTILER'

# --- Map Rendering ---

def render_map(logic: GeoVibesLogic):
    """Renders the Folium map and handles interactions."""
    # Create a Folium map with the selected basemap
    basemap_url = logic.basemap_tiles[st.session_state.current_basemap]
    m = folium.Map(
        location=[logic.center_y, logic.center_x], 
        zoom_start=UIConstants.DEFAULT_ZOOM,
        tiles=basemap_url,
        attr=BasemapConfig.MAPTILER_ATTRIBUTION if 'maptiler' in basemap_url else ''
    )
    
    # Add boundary layer
    if logic.effective_boundary_path:
        style = LayerStyles.get_region_style()
        folium.GeoJson(
            gpd.read_file(logic.effective_boundary_path).to_json(),
            name="Boundary",
            style_function=lambda x: style
        ).add_to(m)

    # Add positive and negative labels
    pos_geojson = logic.get_positive_layer_geojson()
    neg_geojson = logic.get_negative_layer_geojson()

    if pos_geojson['features']:
        folium.GeoJson(
            pos_geojson, 
            name='Positive Labels',
            marker=folium.CircleMarker(radius=5, **LayerStyles.get_point_style(UIConstants.POS_COLOR))
        ).add_to(m)

    if neg_geojson['features']:
        folium.GeoJson(
            neg_geojson, 
            name='Negative Labels',
            marker=folium.CircleMarker(radius=5, **LayerStyles.get_point_style(UIConstants.NEG_COLOR))
        ).add_to(m)
        
    # Add search results
    if st.session_state.search_results is not None:
        gdf = st.session_state.search_results
        min_dist = gdf['distance'].min()
        max_dist = gdf['distance'].max()
        
        for _, row in gdf.iterrows():
            color = UIConstants.distance_to_color(row['distance'], min_dist, max_dist)
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),
                radius=UIConstants.SEARCH_POINT_RADIUS,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=UIConstants.POINT_FILL_OPACITY
            ).add_to(m)

    # Add custom vector layer
    if logic.vector_layer_data:
        style = LayerStyles.get_region_style(color='#FF6B6B', fillColor='#FF6B6B')
        folium.GeoJson(
            logic.vector_layer_data,
            name="Vector Layer",
            style_function=lambda x: style
        ).add_to(m)

    # Enable drawing
    draw_options = {
        'polygon': st.session_state.get('selection_mode') == 'Polygon',
        'polyline': False, 'rectangle': False, 'circle': False, 'marker': False, 'circlemarker': False
    }
    folium.plugins.Draw(export=False, draw_options=draw_options).add_to(m)

    # Render the map with streamlit-folium
    map_data = st_folium(m, width='100%', height=700, returned_objects=['last_clicked', 'all_drawings'])

    # --- Handle Map Interactions ---
    
    # Handle polygon drawing for labeling
    if map_data['all_drawings']:
        drawn_polygon = map_data['all_drawings'][0]['geometry']
        polygon = shapely.geometry.Polygon(drawn_polygon['coordinates'][0])
        
        # This part is simplified and may be slow for large datasets.
        # It queries all points and checks for intersection in memory.
        all_points_query = "SELECT id, ST_AsWKB(geometry) as geometry FROM geo_embeddings"
        all_points_gdf = logic.duckdb_connection.execute(all_points_query).fetch_geo_df()
        
        points_in_polygon = all_points_gdf[all_points_gdf.geometry.within(polygon)]
        point_ids = points_in_polygon['id'].tolist()
        
        if point_ids:
            with st.spinner(f"Labeling {len(point_ids)} points..."):
                for point_id in point_ids:
                    logic.label_point_logic(point_id)
            st.toast(f"Labeled {len(point_ids)} points as {logic.current_label}")
            st.rerun()

    # Handle map clicks for labeling
    elif map_data.get("last_clicked") and st.session_state.get('selection_mode') == 'Point':
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]
        
        with st.spinner("Finding nearest point..."):
            point_id = logic.find_nearest_point_id(lat, lon)
        
        if point_id:
            logic.label_point_logic(point_id)
            st.toast(f"Labeled point {point_id} as {logic.current_label}")
            st.rerun() # Force a rerun to update the map layers

# --- Sidebar Controls ---

def render_sidebar(logic: GeoVibesLogic):
    """Renders the sidebar controls."""
    st.sidebar.title("GeoVibes Controls")

    # --- Database Selection ---
    if logic.available_databases and len(logic.available_databases) > 1:
        db_options = {os.path.basename(db): db for db in logic.available_databases}
        
        # Find the index of the current database
        db_names = list(db_options.keys())
        current_db_name = os.path.basename(logic.current_database_path)
        current_index = db_names.index(current_db_name) if current_db_name in db_names else 0

        selected_db_name = st.sidebar.selectbox(
            "Select Database", 
            options=db_names,
            index=current_index
        )
        
        selected_db_path = db_options[selected_db_name]

        if selected_db_path != logic.current_database_path:
            with st.spinner(f"Switching to {selected_db_name}..."):
                del st.session_state.logic
                # A bit of a hack to force re-initialization with the new path
                # A better approach would be a dedicated method in GeoVibesLogic
                st.session_state.next_db_path = selected_db_path
                st.rerun()

    # --- Search ---
    st.sidebar.header("Similarity Search")
    num_neighbors = st.sidebar.slider("Number of neighbors", 
                                       min_value=UIConstants.MIN_NEIGHBORS, 
                                       max_value=UIConstants.MAX_NEIGHBORS, 
                                       value=UIConstants.DEFAULT_NEIGHBORS, 
                                       step=UIConstants.NEIGHBORS_STEP)
    if st.sidebar.button("Search", use_container_width=True, type="primary"):
        with st.spinner("Searching for similar points..."):
            results = logic.search(num_neighbors)
            if results is not None:
                st.session_state.search_results = results
                st.toast(f"Found {len(results)} similar points.")
            else:
                st.sidebar.warning("Add positive labels before searching.")
        st.rerun()

    if st.sidebar.button("🗑️ Reset All", use_container_width=True):
        logic.reset_all_logic()
        st.session_state.search_results = None
        st.toast("Reset all labels and search results.")
        st.rerun()

    # --- Labeling ---
    with st.sidebar.expander("Labeling", expanded=True):
        label_options = ["Positive", "Negative", "Erase"]
        current_label = st.radio("Label Type", options=label_options, horizontal=True)
        logic.current_label = current_label
        
        if current_label == 'Positive':
            logic.select_val = UIConstants.POSITIVE_LABEL
        elif current_label == 'Negative':
            logic.select_val = UIConstants.NEGATIVE_LABEL
        else:
            logic.select_val = UIConstants.ERASE_LABEL

        st.session_state.selection_mode = st.radio("Selection Mode", options=["Point", "Polygon"], horizontal=True)

    # --- Basemaps ---
    with st.sidebar.expander("Basemaps"):
        basemap_keys = list(logic.basemap_tiles.keys())
        st.session_state.current_basemap = st.radio("Select Basemap", basemap_keys, 
                                                     index=basemap_keys.index(st.session_state.current_basemap))

    # --- Export & Tools ---
    with st.sidebar.expander("Export & Tools"):
        if st.button("💾 Save Labeled Dataset", use_container_width=True):
            # This is a simplified save, we are not prompting for a filename
            logic.save_dataset(None) 
            st.success("Dataset saved to `labeled_dataset_...` files.")

        uploaded_file = st.file_uploader("📂 Load Labeled Dataset", type=['geojson', 'parquet'])
        if uploaded_file:
            with st.spinner("Loading dataset..."):
                content = uploaded_file.getvalue()
                logic.load_dataset_from_content(content, uploaded_file.name)
            st.toast("Dataset loaded successfully.")
            st.rerun()

        uploaded_vector = st.file_uploader("📄 Add Vector Layer", type=['geojson', 'parquet'])
        if uploaded_vector:
            with st.spinner("Loading vector layer..."):
                content = uploaded_vector.getvalue()
                logic.add_vector_layer_from_content(content, uploaded_vector.name)
            st.toast("Vector layer added.")
            st.rerun()

# --- Main App ---

def main():
    """Main function to run the Streamlit app."""
    st.title("🌍 GeoVibes Explorer")
    
    initialize_state()
    
    # Check for database change signal
    if 'next_db_path' in st.session_state:
        db_path = st.session_state.pop('next_db_path')
        st.session_state.logic = GeoVibesLogic(duckdb_path=db_path, verbose=True)
    
    logic = st.session_state.logic
    
    render_sidebar(logic)
    render_map(logic)

if __name__ == "__main__":
    main() 