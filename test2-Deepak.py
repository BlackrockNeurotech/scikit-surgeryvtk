import os
import platform

import cv2
import numpy as np
from sksurgeryimage.utilities.utilities import are_similar
import sksurgeryvtk.models.vtk_surface_model as sm

import vtk
from PySide6.QtWidgets import QApplication
from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow

def create_vtk_error_redirect(output_path="tests/output/vtk.err.txt"):
    """Redirect VTK errors to a file (like your fixture)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    err_out = vtk.vtkFileOutputWindow()
    err_out.SetFileName(output_path)
    vtk_std_err = vtk.vtkOutputWindow()
    vtk_std_err.SetInstance(err_out)
    return vtk_std_err

def create_overlay_window():
    """Create VTKOverlayWindow with same init_widget behavior as fixture."""
    if platform.system() == "Linux":
        init_widget_flag = False
    else:
        init_widget_flag = True

    vtk_overlay = VTKOverlayWindow(offscreen=False, init_widget=init_widget_flag)
    return vtk_overlay

def run_overlay_liver_points(widget_vtk_overlay, pyside_qt_app):
    output_name = "tests/output/"
    os.makedirs(output_name, exist_ok=True)

    
    model = sm.VTKSurfaceModel("brain_models/brain_simple.stl", (1.0, 1.0, 1.0))
    

    input_image_file = "tests/data/liver/fig06_case1b.png"
    image = cv2.imread(input_image_file)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_image_file}")
    print(f"image.shape of {input_image_file} = {image.shape}")
    

    width = image.shape[1]
    height = image.shape[0]


    screen = pyside_qt_app.primaryScreen()
    while width >= screen.geometry().width() or height >= screen.geometry().height():
        width //= 2
        height //= 2

    print(f"Width should be {width}, height should be {height}.")

    ## This part of the code can be used to test clipping the model to a plane at Z=25, which should give us a point cloud of the model's surface at that plane. 
    # Adjust the plane parameters as needed for your specific model and desired clipping.
    # Create a clipping plane at Z = 25
    plane = vtk.vtkPlane()
    plane.SetOrigin(0, 0, 25)   # point on the plane
    plane.SetNormal(0, 0, -1)   # normal pointing downward (clips Z < 25)

    # Apply the clip to the model's VTK polydata
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(model.get_vtk_source_data())  # adjust attribute name if needed
    clipper.SetClipFunction(plane)
    clipper.Update()

    # clipped_model = sm.VTKSurfaceModel.__new__(sm.VTKSurfaceModel)
    clipped_model_dat = clipper.GetOutput()
    clipped_model = sm.VTKSurfaceModel(None,  (1.0, 1.0, 1.0))  # create a new VTKSurfaceModel with the clipped data
    clipped_model.set_source(clipped_model_dat)
    print(type(model.source))
    print(type(clipped_model.source))
    point_cloud = clipped_model.get_points_as_numpy()
    print(f"Loaded model with {point_cloud.shape} points")
    print(point_cloud[:5])  # Print first 5 points to verify


    widget_vtk_overlay.add_vtk_models([clipped_model])

    # widget_vtk_overlay.add_vtk_models([model])
    # widget_vtk_overlay.set_video_image(image)
    # widget_vtk_overlay.set_camera_pose(np.linalg.inv(model_to_camera))
    widget_vtk_overlay.show()
    widget_vtk_overlay.resize(width, height)
    widget_vtk_overlay.AddObserver("ExitEvent", lambda o, e, a=pyside_qt_app: a.quit())

    print(f"Width is {widget_vtk_overlay.width()}, height is {widget_vtk_overlay.height()}.")

    # Run event loop so you can see the window
    pyside_qt_app.exec()

    widget_vtk_overlay.close()


def main():
    # Create QApplication (must be created once per process)
    app = QApplication([])

    # Redirect VTK errors like the fixture did
    create_vtk_error_redirect("tests/output/vtk.err.txt")

    # Create overlay window
    overlay = create_overlay_window()

    # model = sm.VTKSurfaceModel("tests/data/liver/liver_sub.ply", (1.0, 1.0, 1.0))

    # Run your test logic as a script
    run_overlay_liver_points(overlay, app)


if __name__ == "__main__":
    main()