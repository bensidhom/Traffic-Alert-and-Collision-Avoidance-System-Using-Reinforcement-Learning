import streamlit as st
import numpy as np
import plotly.graph_objects as go
import requests

# --- Streamlit Layout ---
st.set_page_config(page_title="ADS-B CPA Viewer", layout="wide")

# --- ADS-B API ---
def get_adsb_aircraft(lat_center, lon_center, radius_km=50, username=None, password=None):
    R = 6371  # Earth radius in km
    delta = radius_km / R
    lat_min = lat_center - np.rad2deg(delta)
    lat_max = lat_center + np.rad2deg(delta)
    lon_min = lon_center - np.rad2deg(delta)
    lon_max = lon_center + np.rad2deg(delta)

    url = f"https://opensky-network.org/api/states/all?lamin={lat_min}&lomin={lon_min}&lamax={lat_max}&lomax={lon_max}"
    auth = (username, password) if username and password else None
    response = requests.get(url, auth=auth)

    if response.status_code == 200:
        data = response.json()
        return data['states'] if data['states'] else []
    else:
        st.error(f"Failed to fetch ADS-B data: {response.status_code}")
        return []

# --- Coordinate Conversion ---
def geodetic_to_cartesian(lat, lon, alt, ref_lat):
    R = 6371000
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    x = R * lon_rad * np.cos(ref_lat_rad)
    y = R * lat_rad
    z = alt
    return np.array([x, y, z])

# --- CPA/TCPA Calculation ---
def compute_cpa_tcpa(pos1, vel1, pos2, vel2, epsilon=1e-6):
    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1
    rel_speed_sq = np.dot(rel_vel, rel_vel)
    if rel_speed_sq < epsilon:
        tcpa = 0.0
        cpa_pos1 = pos1
        cpa_pos2 = pos2
    else:
        tcpa = -np.dot(rel_pos, rel_vel) / rel_speed_sq
        tcpa = max(tcpa, 0)
        cpa_pos1 = pos1 + vel1 * tcpa
        cpa_pos2 = pos2 + vel2 * tcpa
    cpa_dist = np.linalg.norm(cpa_pos1 - cpa_pos2)
    return cpa_dist, tcpa, cpa_pos1, cpa_pos2

# --- Sidebar Settings ---
st.sidebar.header("📡 ADS-B & CPA Settings")
center_lat = st.sidebar.number_input("Center Latitude", value=40.0)
center_lon = st.sidebar.number_input("Center Longitude", value=-75.0)
threshold_radius = st.sidebar.number_input("CPA Safety Threshold (m)", value=500.0, min_value=100.0)
time_minutes = st.sidebar.slider("Time into the future (minutes)", 0, 60, 10)
time_sec = time_minutes * 60

username = st.sidebar.text_input("OpenSky Username")
password = st.sidebar.text_input("OpenSky Password", type="password")

# --- Fetch Live Aircraft ---
st.title("✈️ Real-Time ADS-B CPA & TCPA Viewer")
adsb_planes = get_adsb_aircraft(center_lat, center_lon, 50, username, password)

if not adsb_planes:
    st.warning("No aircraft data received.")
    st.stop()

# --- Filter and Prepare Aircraft Data ---
states = []
for s in adsb_planes:
    if all(s[i] is not None for i in [5, 6, 7, 9, 10]):
        states.append({
            "icao": s[0],
            "callsign": s[1].strip() if s[1] else s[0],
            "lon": s[5],
            "lat": s[6],
            "alt": s[7],
            "velocity": s[9],
            "heading": s[10],
            "v_rate": s[11] or 0.0
        })

if not states:
    st.warning("No valid aircraft with full position/speed data.")
    st.stop()

# --- Choose Reference Aircraft ---
reference_callsigns = [f"{i+1}: {s['callsign']} ({s['icao']})" for i, s in enumerate(states)]
ref_index = st.selectbox("Select Reference Aircraft", list(range(len(reference_callsigns))), format_func=lambda i: reference_callsigns[i])
ref_state = states[ref_index]
ref_lat = ref_state['lat']
ref_cartesian = geodetic_to_cartesian(ref_state['lat'], ref_state['lon'], ref_state['alt'], ref_lat)
heading_rad = np.deg2rad(ref_state['heading'])
ref_velocity = np.array([
    ref_state['velocity'] * np.sin(heading_rad),
    ref_state['velocity'] * np.cos(heading_rad),
    ref_state['v_rate']
])

# --- Compute for All Other Aircraft ---
positions = []
velocities = []
labels = []
intrusions = []

for i, s in enumerate(states):
    if i == ref_index:
        continue

    lat, lon, alt = s['lat'], s['lon'], s['alt']
    pos = geodetic_to_cartesian(lat, lon, alt, ref_lat)
    heading_rad = np.deg2rad(s['heading'])
    vel = np.array([
        s['velocity'] * np.sin(heading_rad),
        s['velocity'] * np.cos(heading_rad),
        s['v_rate']
    ])
    positions.append(pos)
    velocities.append(vel)
    labels.append(s['callsign'])

    cpa_dist, tcpa, _, _ = compute_cpa_tcpa(ref_cartesian, ref_velocity, pos, vel)
    if cpa_dist <= threshold_radius:
        intrusions.append((s['callsign'], cpa_dist, tcpa))

# --- Display Intrusion Warnings ---
st.markdown("## 🚨 CPA Intrusion Warnings")
if intrusions:
    for call, dist, time in intrusions:
        st.error(f"{call}: CPA = {dist:.2f} m, TCPA = {time:.2f} s")
else:
    st.success("✅ No aircraft currently within the safety zone.")

# --- Plotly Visualization ---
st.markdown("## 🛰 3D Aircraft Motion Viewer")
fig = go.Figure()

# Safety Sphere Around Reference Plane
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = threshold_radius * np.cos(u) * np.sin(v) + ref_cartesian[0]
y_sphere = threshold_radius * np.sin(u) * np.sin(v) + ref_cartesian[1]
z_sphere = threshold_radius * np.cos(v) + ref_cartesian[2]

fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    showscale=False, opacity=0.2,
    colorscale='Reds', name="Safety Zone",
    hoverinfo='skip'
))

# Reference Aircraft Path
ref_traj = ref_cartesian + np.outer(np.linspace(0, time_sec, 100), ref_velocity)
fig.add_trace(go.Scatter3d(x=ref_traj[:, 0], y=ref_traj[:, 1], z=ref_traj[:, 2],
    mode='lines', name=f"Reference: {ref_state['callsign']}"
))
fig.add_trace(go.Scatter3d(x=[ref_cartesian[0]], y=[ref_cartesian[1]], z=[ref_cartesian[2]],
    mode='markers', marker=dict(size=6, color='black'), name="Start Position"
))

# Other Aircraft Paths
for i, pos in enumerate(positions):
    traj = pos + np.outer(np.linspace(0, time_sec, 100), velocities[i])
    is_intruding = labels[i] in [x[0] for x in intrusions]
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode='lines', name=f"{labels[i]} Path"
    ))
    fig.add_trace(go.Scatter3d(
        x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
        mode='markers', marker=dict(size=5, color='red' if is_intruding else 'blue'),
        name=f"{labels[i]} @ T+{time_minutes}min"
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        zaxis_title="Altitude (m)"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    title="Live Aircraft Trajectories & Safety Monitoring",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
