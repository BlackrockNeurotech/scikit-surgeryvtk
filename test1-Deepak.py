# -*- coding: utf-8 -*-

import copy
import os
import platform

import cv2
import numpy as np
from sksurgeryimage.utilities.utilities import are_similar
import sksurgeryvtk.models.vtk_surface_model as sm

import vtk
from PySide6.QtWidgets import QApplication
from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow


def _reproject_and_save_image(image,
                              model_to_camera,
                              point_cloud,
                              camera_matrix,
                              output_file):
    output_image = copy.deepcopy(image)
    rmat = model_to_camera[:3, :3]
    rvec = cv2.Rodrigues(rmat)[0]
    tvec = model_to_camera[:3, 3]

    projected, _ = cv2.projectPoints(point_cloud, rvec, tvec, camera_matrix, None)
    for i in range(projected.shape[0]):
        x_c, y_c = projected[i][0]
        x_c = int(x_c)
        y_c = int(y_c)

        if x_c <= 0 or x_c >= output_image.shape[1]:
            continue
        if y_c <= 0 or y_c >= output_image.shape[0]:
            continue

        output_image[y_c, x_c, :] = [255, 0, 0]

    cv2.imwrite(output_file, output_image)


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

    # Reference image selection
    ref_image_path = "tests/data/liver/fig06_case1b_overlay.png"
    in_github_ci = os.environ.get("CI")
    if in_github_ci:
        ref_image_path = "tests/data/liver/fig06_case1b_overlay_for_ci.png"

    print(f"\nenviron = {in_github_ci}")
    print(f"platform.system = {platform.system()}")
    print(f"platform.machine = {platform.machine()}")
    print(f"platform.architecture = {platform.architecture()}")

    print(f"\nusing ref_image_path: {ref_image_path}")
    reference_image = cv2.imread(ref_image_path)
    if reference_image is None:
        raise FileNotFoundError(f"Could not read reference image: {ref_image_path}")
    print(f"reference_image.shape of {ref_image_path} = {reference_image.shape}")

    input_image_file = "tests/data/liver/fig06_case1b.png"
    image = cv2.imread(input_image_file)
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {input_image_file}")
    print(f"image.shape of {input_image_file} = {image.shape}")

    width = image.shape[1]
    height = image.shape[0]

    model_to_camera_file = "tests/data/liver/model_to_camera.txt"
    model_to_camera = np.loadtxt(model_to_camera_file)

    model = sm.VTKSurfaceModel("tests/data/liver/liver_sub.ply", (1.0, 1.0, 1.0))
    point_cloud = model.get_points_as_numpy()
    print(f"Loaded model with {point_cloud.shape} points")

    intrinsics_file = "tests/data/liver/calib.left.intrinsics.txt"
    intrinsics = np.loadtxt(intrinsics_file)

    output_image_file = "tests/output/liver_sub_projected.png"
    print(f"output_image_file= {output_image_file}")

    _reproject_and_save_image(image, model_to_camera, point_cloud, intrinsics, output_image_file)

    screen = pyside_qt_app.primaryScreen()
    while width >= screen.geometry().width() or height >= screen.geometry().height():
        width //= 2
        height //= 2

    print(f"Width should be {width}, height should be {height}.")

    widget_vtk_overlay.add_vtk_models([model])
    widget_vtk_overlay.set_video_image(image)
    widget_vtk_overlay.set_camera_pose(np.linalg.inv(model_to_camera))
    widget_vtk_overlay.show()
    widget_vtk_overlay.resize(width, height)
    widget_vtk_overlay.AddObserver("ExitEvent", lambda o, e, a=pyside_qt_app: a.quit())

    print(f"Width is {widget_vtk_overlay.width()}, height is {widget_vtk_overlay.height()}.")

    opengl_mat, vtk_mat = widget_vtk_overlay.set_camera_matrix(intrinsics)
    if opengl_mat and vtk_mat:
        print(f"OpenGL matrix= {opengl_mat}")
        print(f"VTK matrix= {vtk_mat}")

        for r in range(0, 4):
            for c in range(0, 4):
                assert np.isclose(opengl_mat.GetElement(r, c), vtk_mat.GetElement(r, c))

        ref_output_image_path = "tests/output/liver_sub_projected.png"
        widget_vtk_overlay.save_scene_to_file(ref_output_image_path)

        rendered_image = cv2.imread(ref_output_image_path)
        if rendered_image is None:
            raise FileNotFoundError(f"Could not read rendered image: {ref_output_image_path}")

        print(f"reference_image.shape of {ref_image_path} = {reference_image.shape}")
        print(f"rendered_image.shape of {ref_output_image_path} = {rendered_image.shape}")

        if "Linux" not in platform.system():
            assert are_similar(
                reference_image,
                rendered_image,
                threshold=0.995,
                metric=cv2.TM_CCOEFF_NORMED,
                mean_threshold=0.005,
            )

    # Run event loop so you can see the window
    # pyside_qt_app.exec()

    widget_vtk_overlay.close()


def main():
    # Create QApplication (must be created once per process)
    app = QApplication([])

    # Redirect VTK errors like the fixture did
    create_vtk_error_redirect("tests/output/vtk.err.txt")

    # Create overlay window
    overlay = create_overlay_window()

    # Run your test logic as a script
    run_overlay_liver_points(overlay, app)


if __name__ == "__main__":
    main()