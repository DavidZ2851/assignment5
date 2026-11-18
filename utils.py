import os
import torch
import numpy as np
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio

def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    else:
        path = os.path.join(args.checkpoint_dir, 'model_epoch_{}.pt'.format(epoch))
    torch.save(model.state_dict(), path)

def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def viz_seg (verts, labels, path, device):
    """
    visualize segmentation result
    output: a 360-degree gif
    """
    image_size=256
    background_color=(1, 1, 1)
    colors = [[1.0,1.0,1.0], [1.0,0.0,1.0], [0.0,1.0,1.0],[1.0,1.0,0.0],[0.0,0.0,1.0], [1.0,0.0,0.0]]

    # Construct various camera viewpoints
    dist = 3
    elev = 0
    azim = [180 - 12*i for i in range(30)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    sample_verts = verts.unsqueeze(0).repeat(30,1,1).to(torch.float)
    sample_labels = labels.unsqueeze(0)
    sample_colors = torch.zeros((1, verts.shape[0], 3))

    # Colorize points based on segmentation labels
    for i in range(6):
        sample_colors[sample_labels==i] = torch.tensor(colors[i])

    sample_colors = sample_colors.repeat(30,1,1).to(torch.float)

    point_cloud = pytorch3d.structures.Pointclouds(points=sample_verts, features=sample_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    rend = renderer(point_cloud, cameras=c).cpu().numpy() # (30, 256, 256, 3)
    rend = (rend * 255).astype(np.uint8)

    imageio.mimsave(path, rend, fps=15, loop=0)



def rotate_points(points, deg, axis='x'):
    """Rotate (B, N, 3) or (N, 3) points by `deg` degrees about `axis` (x, y or z)."""
    if deg == 0:
        return points
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    R = {'x': [[1, 0, 0], [0, c, -s], [0, s, c]],
         'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    R = torch.tensor(R, dtype=points.dtype, device=points.device)
    return points @ R.T


@torch.no_grad()
def batched_predict(model, data, batch_size, device):
    """Run the model over `data` (B, N, 3) in mini-batches; returns argmax labels on CPU."""
    model.eval()
    preds = []
    for i in range(0, data.shape[0], batch_size):
        out = model(data[i:i + batch_size].to(device).float())
        preds.append(out.argmax(dim=-1).cpu())
    return torch.cat(preds)


def viz_cloud(verts, path, device, color=(0.3, 0.5, 0.9), n_views=30):
    """Render a single-colour 360-degree gif of a point cloud."""
    dist, elev = 3, 0
    azim = [180 - 360 // n_views * i for i in range(n_views)]
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
    v = verts.unsqueeze(0).repeat(n_views, 1, 1).to(torch.float)
    rgb = torch.tensor(color, dtype=torch.float).view(1, 1, 3).expand(n_views, verts.shape[0], 3)
    pc = pytorch3d.structures.Pointclouds(points=v, features=rgb).to(device)
    renderer = get_points_renderer(image_size=256, device=device)
    rend = (renderer(pc, cameras=c).cpu().numpy() * 255).astype(np.uint8)
    imageio.mimsave(path, rend, fps=15, loop=0)
