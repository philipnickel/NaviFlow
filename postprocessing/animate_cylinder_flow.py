#!/usr/bin/env python3

import os
import glob
from paraview.simple import *

def create_animation(data_dir, output_file):
    # Disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # Get all VTK files sorted by time
    vtk_files = sorted(glob.glob(os.path.join(data_dir, '*.vtk')))
    if not vtk_files:
        print("No VTK files found in directory:", data_dir)
        return

    # Create a new 'Legacy VTK Reader'
    flow = LegacyVTKReader(FileNames=vtk_files)

    # Get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # Show data in view
    flowDisplay = Show(flow, renderView1)

    # Set scalar coloring
    ColorBy(flowDisplay, ('POINTS', 'velocity_magnitude'))

    # Get color transfer function/opacity function
    velocityLUT = GetColorTransferFunction('velocity_magnitude')
    velocityPWF = GetOpacityTransferFunction('velocity_magnitude')

    # Set scalar coloring range
    velocityLUT.RescaleTransferFunction(0.0, 2.0)
    velocityPWF.RescaleTransferFunction(0.0, 2.0)

    # Add streamlines
    streamTracer1 = StreamTracer(Input=flow,
        SeedType='Line',
        Vectors=['POINTS', 'velocity'])

    # Properties modified on streamTracer1.SeedType
    streamTracer1.SeedType.Point1 = [0.0, 0.0, 0.0]
    streamTracer1.SeedType.Point2 = [0.0, 0.4, 0.0]
    streamTracer1.SeedType.Resolution = 20

    # Show streamlines
    streamTracerDisplay = Show(streamTracer1, renderView1)
    ColorBy(streamTracerDisplay, ('POINTS', 'velocity_magnitude'))

    # Update the view to ensure all elements are shown
    renderView1.ResetCamera()
    renderView1.CameraPosition = [0.2, 0.2, 2.0]
    renderView1.CameraFocalPoint = [0.2, 0.2, 0.0]
    renderView1.CameraParallelScale = 0.3

    # Save animation
    SaveAnimation(output_file, renderView1, 
                 ImageResolution=[1920, 1080],
                 FrameRate=10)

if __name__ == "__main__":
    # Get the experiment directory from config
    experiment_dir = "experiments/Collocated/transient_cylinderFlow"
    data_dir = os.path.join(experiment_dir, "results", "vtk")
    output_file = os.path.join(experiment_dir, "results", "animation.mp4")
    
    create_animation(data_dir, output_file) 