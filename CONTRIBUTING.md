# Contributing to 4DoF Vision-Guided Robotic Pen Sorting

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, hardware)
- Relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Possible implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create your branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed

3. **Test your changes**
   - Run existing tests: `python test_pixel_conversion.py` and `python test_coordinates.py`
   - Test in mock mode: `python camera_stream.py --mock-robot ResearchDataset`
   - If you have hardware, test the full pipeline

4. **Commit your changes**
   - Use clear, descriptive commit messages
   - Reference any related issues

5. **Push to your fork** and submit a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

6. **In your pull request description:**
   - Describe what changes you made and why
   - Reference any related issues
   - Include screenshots/videos if applicable
   - Note any breaking changes

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines for Python
- Use meaningful variable names
- Add docstrings for functions and classes
- Keep functions focused and modular

### Testing

- Test coordinate transformations thoroughly
- Verify calibration works with different cameras
- Test both STANDARD and COMPLEX motion modes
- Ensure mock mode works without hardware

### Documentation

- Update the README if you add features
- Add comments for non-obvious code
- Update type hints where applicable

## Project Structure

- `camera_stream.py` - Main detection and control loop
- `camera_calibrate.py` - Camera intrinsic calibration
- `aruco_pose.py` - Extrinsic calibration
- `RoArm/` - Robot control modules
- `Misc/` - Utility scripts

## Areas for Contribution

Here are some areas where contributions would be particularly valuable:

### Vision System
- Support for additional object types beyond pens
- Improved color classification accuracy
- Real-time performance optimization
- Support for multiple cameras

### Motion Planning
- New grasp strategies for different object orientations
- Obstacle avoidance
- Multi-object sequencing
- Adaptive gripping force

### Calibration
- Automatic checkerboard detection for easier calibration
- Support for different ArUco dictionary sizes
- Calibration validation tools
- GUI for calibration workflow

### Hardware Support
- Support for other robot arms
- Alternative communication protocols
- Support for different gripper types

### Documentation
- Tutorial videos
- Detailed calibration guide with photos
- Troubleshooting flowchart
- Performance benchmarks

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Tag @Anirudhpro in issues or PRs

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the project
- Be patient with newcomers

Thank you for contributing!
