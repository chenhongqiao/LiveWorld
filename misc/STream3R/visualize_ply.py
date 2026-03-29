import gradio as gr
import trimesh
import numpy as np
import plotly.graph_objects as go


def visualize_ply(file_path):
    if file_path is None:
        return None

    mesh = trimesh.load(str(file_path))

    if isinstance(mesh, trimesh.PointCloud):
        points = mesh.vertices
        colors = mesh.colors[:, :3] if mesh.colors is not None else None
    elif isinstance(mesh, trimesh.Trimesh):
        points = mesh.vertices
        colors = mesh.visual.vertex_colors[:, :3] if mesh.visual.vertex_colors is not None else None
    elif isinstance(mesh, trimesh.Scene):
        all_points = []
        all_colors = []
        for name, geom in mesh.geometry.items():
            if hasattr(geom, "vertices"):
                all_points.append(geom.vertices)
                if hasattr(geom, "colors") and geom.colors is not None:
                    all_colors.append(geom.colors[:, :3])
                elif hasattr(geom, "visual") and geom.visual.vertex_colors is not None:
                    all_colors.append(geom.visual.vertex_colors[:, :3])
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0) if all_colors else None
    else:
        return None

    # Subsample if too many points
    max_points = 500_000
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    marker = dict(size=1)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8)
        marker["color"] = [f"rgb({r},{g},{b})" for r, g, b in colors]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
    )
    return fig


with gr.Blocks(title="PLY Point Cloud Viewer") as demo:
    gr.Markdown("## PLY Point Cloud Viewer")
    file_input = gr.File(label="Upload PLY file", file_types=[".ply"])
    plot = gr.Plot(label="Point Cloud")
    file_input.upload(fn=visualize_ply, inputs=file_input, outputs=plot, api_name=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
