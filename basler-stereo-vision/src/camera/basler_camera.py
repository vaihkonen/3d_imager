import numpy as np
from pypylon import pylon
import logging

class BaslerCamera:
    def __init__(self, camera_ip=None, camera_index=0):
        self.camera_ip = camera_ip
        self.camera_index = camera_index
        self.camera = None
        self.converter = None
        self.is_grabbing = False
        self.discovered_cameras = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

    def discover_cameras(self):
        """Discover all available Basler cameras on the network"""
        try:
            # Get the transport layer factory
            tlFactory = pylon.TlFactory.GetInstance()
            
            # Enumerate all available cameras
            cameras = tlFactory.EnumerateDevices()
            
            self.discovered_cameras = []
            for i, camera_info in enumerate(cameras):
                try:
                    camera_details = {
                        'index': i,
                        'device_class': camera_info.GetDeviceClass(),
                        'serial_number': camera_info.GetSerialNumber(),
                        'model_name': camera_info.GetModelName(),
                        'vendor_name': camera_info.GetVendorName(),
                        'device_version': camera_info.GetDeviceVersion(),
                        'ip_address': camera_info.GetIpAddress() if hasattr(camera_info, 'GetIpAddress') else 'N/A',
                        'mac_address': camera_info.GetMacAddress() if hasattr(camera_info, 'GetMacAddress') else 'N/A',
                        'friendly_name': camera_info.GetFriendlyName()
                    }
                    self.discovered_cameras.append(camera_details)
                    self.logger.info(f"Found camera {i}: {camera_details['model_name']} "
                                   f"(S/N: {camera_details['serial_number']}, "
                                   f"IP: {camera_details['ip_address']})")
                except Exception as e:
                    self.logger.warning(f"Could not get details for camera {i}: {str(e)}")
            
            self.logger.info(f"Discovered {len(self.discovered_cameras)} Basler camera(s)")
            return self.discovered_cameras
            
        except Exception as e:
            self.logger.error(f"Failed to discover cameras: {str(e)}")
            return []

    def initialize(self):
        """Initialize the Basler camera - auto-discover or use specified IP/index"""
        try:
            # Get the transport layer factory
            tlFactory = pylon.TlFactory.GetInstance()
            
            if self.camera_ip:
                # Use specific IP address if provided
                self.logger.info(f"Connecting to camera at IP: {self.camera_ip}")
                info = pylon.DeviceInfo()
                info.SetIpAddress(self.camera_ip)
                cameras = tlFactory.EnumerateDevices([info])
                
                if len(cameras) == 0:
                    raise RuntimeError(f"No Basler camera found at IP: {self.camera_ip}")
                    
                selected_camera = cameras[0]
                
            else:
                # Auto-discover cameras
                self.logger.info("Auto-discovering Basler cameras on network...")
                cameras = tlFactory.EnumerateDevices()
                
                if len(cameras) == 0:
                    raise RuntimeError("No Basler cameras found on the network")
                
                # Display discovered cameras
                self.discover_cameras()
                
                # Select camera by index
                if self.camera_index >= len(cameras):
                    raise RuntimeError(f"Camera index {self.camera_index} out of range. "
                                     f"Found {len(cameras)} camera(s)")
                
                selected_camera = cameras[self.camera_index]
                self.logger.info(f"Selected camera {self.camera_index}: "
                               f"{self.discovered_cameras[self.camera_index]['model_name']}")
            
            # Create camera instance
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_camera))
            
            # Open the camera
            self.camera.Open()
            
            # Get actual IP address after connection
            if not self.camera_ip and hasattr(selected_camera, 'GetIpAddress'):
                self.camera_ip = selected_camera.GetIpAddress()
            
            # Set camera parameters for optimal performance
            self._configure_camera()
            
            # Create image converter for color conversion
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self.logger.info(f"Basler camera initialized successfully at IP: {self.camera_ip}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            return False

    def _configure_camera(self):
        """Configure camera parameters for stereo vision"""
        try:
            # Enable free-running mode
            self.camera.AcquisitionMode.SetValue('Continuous')
            
            # Set exposure time (adjust as needed)
            if self.camera.ExposureTime.IsWritable():
                self.camera.ExposureTime.SetValue(10000)  # 10ms
            
            # Set gain (adjust as needed)
            if self.camera.Gain.IsWritable():
                self.camera.Gain.SetValue(1.0)
            
            # Set pixel format to RGB8 if available
            if self.camera.PixelFormat.IsWritable():
                available_formats = self.camera.PixelFormat.Symbolics
                if 'RGB8' in available_formats:
                    self.camera.PixelFormat.SetValue('RGB8')
                elif 'BayerRG8' in available_formats:
                    self.camera.PixelFormat.SetValue('BayerRG8')
                    
            self.logger.info("Camera parameters configured")
            
        except Exception as e:
            self.logger.warning(f"Some camera parameters could not be set: {str(e)}")

    def start_grabbing(self):
        """Start continuous image acquisition"""
        try:
            if self.camera and self.camera.IsOpen():
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                self.is_grabbing = True
                self.logger.info("Started grabbing images")
                return True
        except Exception as e:
            self.logger.error(f"Failed to start grabbing: {str(e)}")
            return False

    def stop_grabbing(self):
        """Stop continuous image acquisition"""
        try:
            if self.camera and self.is_grabbing:
                self.camera.StopGrabbing()
                self.is_grabbing = False
                self.logger.info("Stopped grabbing images")
        except Exception as e:
            self.logger.error(f"Failed to stop grabbing: {str(e)}")

    def capture_frame(self, timeout_ms=5000):
        """Capture a single frame from the camera"""
        try:
            if not self.camera or not self.camera.IsOpen():
                self.logger.error("Camera not initialized or opened")
                return None
            
            # Start grabbing if not already started
            if not self.is_grabbing:
                self.start_grabbing()
            
            # Wait for an image and then retrieve it
            grabResult = self.camera.RetrieveResult(timeout_ms, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                # Convert the image to BGR format
                image = self.converter.Convert(grabResult)
                img_array = image.GetArray()
                
                # Release the grab result
                grabResult.Release()
                
                return img_array
            else:
                self.logger.error("Image grab failed")
                grabResult.Release()
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {str(e)}")
            return None

    def get_camera_info(self):
        """Get camera information"""
        if not self.camera or not self.camera.IsOpen():
            return None
        
        try:
            info = {
                'DeviceVendorName': str(self.camera.DeviceVendorName.GetValue()),
                'DeviceModelName': str(self.camera.DeviceModelName.GetValue()),
                'DeviceSerialNumber': str(self.camera.DeviceSerialNumber.GetValue()),
                'DeviceVersion': str(self.camera.DeviceVersion.GetValue()),
                'IP_Address': self.camera_ip
            }
            return info
        except Exception as e:
            self.logger.error(f"Failed to get camera info: {str(e)}")
            return None

    def list_discovered_cameras(self):
        """Return list of discovered cameras"""
        return self.discovered_cameras

    def get_camera_count(self):
        """Get the number of discovered cameras"""
        return len(self.discovered_cameras)

    @staticmethod
    def list_available_cameras():
        """Static method to quickly list all available cameras without creating an instance"""
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            cameras = tlFactory.EnumerateDevices()
            
            camera_list = []
            for i, camera_info in enumerate(cameras):
                try:
                    camera_details = {
                        'index': i,
                        'serial_number': camera_info.GetSerialNumber(),
                        'model_name': camera_info.GetModelName(),
                        'vendor_name': camera_info.GetVendorName(),
                        'ip_address': camera_info.GetIpAddress() if hasattr(camera_info, 'GetIpAddress') else 'N/A',
                        'friendly_name': camera_info.GetFriendlyName()
                    }
                    camera_list.append(camera_details)
                except Exception:
                    pass
            
            return camera_list
            
        except Exception:
            return []

    def is_connected(self):
        """Check if camera is connected and ready"""
        try:
            return self.camera is not None and self.camera.IsOpen()
        except:
            return False

    def close(self):
        """Release camera resources and close connection"""
        try:
            if self.is_grabbing:
                self.stop_grabbing()
            
            if self.camera and self.camera.IsOpen():
                self.camera.Close()
                self.logger.info("Camera closed successfully")
                
        except Exception as e:
            self.logger.error(f"Error closing camera: {str(e)}")
        finally:
            self.camera = None
            self.converter = None
            self.is_grabbing = False