import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- Streamlit Layout Config ---
st.set_page_config(page_title="3D CPA & TCPA Viewer", layout="wide")

# --- Geodetic to Cartesian ---
def geodetic_to_cartesian(lat, lon, alt, ref_lat):
    R = 6371000
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    ref_lat_rad = np.deg2rad(ref_lat)

    x = R * lon_rad * np.cos(ref_lat_rad)  # East
    y = R * lat_rad                        # North
    z = alt                                # Altitude
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

# --- Sidebar Config ---
st.sidebar.header("⚙️ Settings")
threshold_radius = st.sidebar.number_input("CPA Safety Threshold (meters)", value=500.0, min_value=100.0)
time_minutes = st.sidebar.slider("Time into the future (minutes)", 0, 60, 10)
time_sec = time_minutes * 60

# --- Input Aircraft Info ---
st.title("✈️ 3D CPA and TCPA Visualizer with Safety Threshold")
st.markdown("The first aircraft is used as the reference for CPA/TCPA and intrusion alerts.")

num_planes = st.number_input("Number of airplanes", min_value=2, value=3)
raw_positions = []
velocities = []

for i in range(num_planes):
    st.subheader(f"Aircraft {i+1}")
    col1, col2, col3 = st.columns(3)
    with col1:
        lat = st.number_input(f"Latitude {i+1}", value=30.0 + i)
    with col2:
        lon = st.number_input(f"Longitude {i+1}", value=-90.0 + i)
    with col3:
        alt = st.number_input(f"Altitude {i+1} (m)", value=10000.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        speed = st.number_input(f"Speed {i+1} (m/s)", value=250.0)
    with col5:
        heading = st.number_input(f"Heading {i+1} (°)", value=(i * 45.0) % 360)
    with col6:
        vertical_speed = st.number_input(f"Vertical Speed {i+1} (m/s)", value=0.0)

    heading_rad = np.deg2rad(heading)
    vx = speed * np.sin(heading_rad)
    vy = speed * np.cos(heading_rad)
    vz = vertical_speed

    raw_positions.append((lat, lon, alt))
    velocities.append(np.array([vx, vy, vz]))

# --- Convert Coordinates ---
ref_lat = raw_positions[0][0]
positions = [geodetic_to_cartesian(lat, lon, alt, ref_lat) for (lat, lon, alt) in raw_positions]

# --- CPA/TCPA Results ---
st.markdown("## 📊 CPA and TCPA Results")
intrusions = []

for i in range(1, num_planes):
    cpa_dist, tcpa, _, _ = compute_cpa_tcpa(positions[0], velocities[0], positions[i], velocities[i])
    if cpa_dist <= threshold_radius:
        intrusions.append(i)
        st.error(f"🚨 Plane {i+1} violates threshold! CPA = `{cpa_dist:.2f} m`, TCPA = `{tcpa:.2f} s`")
    else:
        st.success(f"✈️ Plane {i+1} safe. CPA = `{cpa_dist:.2f} m`, TCPA = `{tcpa:.2f} s`")

if intrusions:
    st.warning(f"⚠️ {len(intrusions)} intrusion(s) detected within {threshold_radius:.0f} meters.")
else:
    st.success("✅ All aircraft are at a safe distance.")

# --- Interactive 3D Plot (Plotly) ---
st.markdown("## 🛰 3D Trajectory Viewer with Safety Zone")

fig = go.Figure()

# Draw safety sphere around plane 1
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x_sphere = threshold_radius * np.cos(u) * np.sin(v) + positions[0][0]
y_sphere = threshold_radius * np.sin(u) * np.sin(v) + positions[0][1]
z_sphere = threshold_radius * np.cos(v) + positions[0][2]

fig.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere,
    showscale=False,
    opacity=0.2,
    colorscale='Reds',
    name="Safety Zone",
    hoverinfo='skip'
))

# Plot each aircraft path
for i in range(num_planes):
    traj = positions[i] + np.outer(np.linspace(0, time_sec, 100), velocities[i])
    curr_pos = positions[i] + velocities[i] * time_sec
    color = 'red' if i in intrusions else 'blue'

    # Path line
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
        mode='lines',
        name=f"Plane {i+1} Path"
    ))

    # Starting position
    fig.add_trace(go.Scatter3d(
        x=[positions[i][0]], y=[positions[i][1]], z=[positions[i][2]],
        mode='markers',
        marker=dict(size=6, color='black'),
        name=f"Plane {i+1} Start"
    ))

    # Position at selected time
    fig.add_trace(go.Scatter3d(
        x=[curr_pos[0]], y=[curr_pos[1]], z=[curr_pos[2]],
        mode='markers',
        marker=dict(size=5, color=color),
        name=f"Plane {i+1} @ T+{time_minutes}min"
    ))

fig.update_layout(
    scene=dict(
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        zaxis_title="Altitude (m)"
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    title="Aircraft Predicted Trajectories",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
