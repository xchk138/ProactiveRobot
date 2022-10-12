# DockMode 
## Dock mode works on robots such as ships, drones and cars.
## Visual target is required to locate the home of robots.

### Instead of using carefully designed visual marks to guide robots heading home, this project uses random but type-fixed targets like a ship-house or colored container.
### We extends the dock mode to all casual scenaries and make it much more easier to realize the docking functionality.


### requirements

- GCC 9.4.0 or MSVC 2012 (C11 features used)
- OpenCV 4.5.5 (opencv 4 is basically enough)
- FFMPEG should be available to OpenCV to decode sort of videos.
- ONNX libs on python envs.
- pytorch 1.12.0 (+cu116 is optional if Nvidia GPU is unavailable)
- Netron to visualize .onnx models.