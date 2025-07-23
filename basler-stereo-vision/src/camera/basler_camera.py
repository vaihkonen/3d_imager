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

    def _safe_set_parameter(self, param_names, value, description=""):
        """
        Safely set a camera parameter, trying multiple parameter names if necessary.
        
        Args:
            param_names: String or list of parameter names to try
            value: Value to set
            description: Description for logging
            
        Returns:
            tuple: (success, parameter_name_used)
        """
        if isinstance(param_names, str):
            param_names = [param_names]
            
        for param_name in param_names:
            try:
                if hasattr(self.camera, param_name):
                    param = getattr(self.camera, param_name)
                    # Check if parameter is writable (use property, not method)
                    if hasattr(param, 'IsWritable') and param.IsWritable:
                        param.SetValue(value)
                        return True, param_name
                    else:
                        self.logger.debug(f"Parameter {param_name} exists but is not writable")
                else:
                    self.logger.debug(f"Parameter {param_name} does not exist on this camera model")
            except Exception as e:
                self.logger.debug(f"Failed to set {param_name}: {str(e)}")
                continue
                
        return False, None

    def _configure_optional_parameters(self):
        """Configure optional parameters that may not be available on all camera models"""
        optional_configs = []
        
        # Try to set frame rate if supported
        success, param_used = self._safe_set_parameter(['AcquisitionFrameRateEnable'], True)
        if success:
            success2, param_used2 = self._safe_set_parameter(['AcquisitionFrameRate', 'AcquisitionFrameRateAbs'], 30.0)
            if success2:
                optional_configs.append(f"FrameRate ({param_used2})")
        
        # Try to set width and height if supported
        success, param_used = self._safe_set_parameter(['Width'], 1920)
        if success:
            optional_configs.append(f"Width ({param_used})")
            
        success, param_used = self._safe_set_parameter(['Height'], 1080)
        if success:
            optional_configs.append(f"Height ({param_used})")
        
        # Try to enable auto white balance if available
        success, param_used = self._safe_set_parameter(['BalanceWhiteAuto'], 'Continuous')
        if success:
            optional_configs.append(f"BalanceWhiteAuto ({param_used})")
        
        if optional_configs:
            self.logger.info(f"Optional parameters configured: {', '.join(optional_configs)}")
        
        return optional_configs

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
        configured_params = []
        failed_params = []
        
        try:
            # Enable free-running mode
            success, param_used = self._safe_set_parameter('AcquisitionMode', 'Continuous')
            if success:
                configured_params.append(f"AcquisitionMode ({param_used})")
            else:
                failed_params.append("AcquisitionMode (not available or not writable)")
            
            # Set exposure time (try different parameter names for different camera models)
            success, param_used = self._safe_set_parameter(['ExposureTime', 'ExposureTimeAbs'], 10000)
            if success:
                configured_params.append(f"ExposureTime ({param_used})")
            else:
                failed_params.append("ExposureTime (not available or not writable)")
            
            # Set gain (try different parameter names for different camera models)
            success, param_used = self._safe_set_parameter(['Gain'], 1.0)
            if not success:
                success, param_used = self._safe_set_parameter(['GainRaw'], 100)
            
            if success:
                configured_params.append(f"Gain ({param_used})")
            else:
                failed_params.append("Gain (not available or not writable)")
            
            # Set pixel format to the best available option
            pixel_format_set = False
            try:
                if hasattr(self.camera, 'PixelFormat') and hasattr(self.camera.PixelFormat, 'IsWritable') and self.camera.PixelFormat.IsWritable:
                    available_formats = list(self.camera.PixelFormat.Symbolics)
                    format_priority = ['RGB8', 'BayerRG8', 'BayerGB8', 'BayerGR8', 'BayerBG8', 'Mono8']
                    
                    for fmt in format_priority:
                        if fmt in available_formats:
                            try:
                                self.camera.PixelFormat.SetValue(fmt)
                                configured_params.append(f"PixelFormat ({fmt})")
                                pixel_format_set = True
                                break
                            except Exception as e:
                                self.logger.debug(f"Failed to set PixelFormat to {fmt}: {str(e)}")
                                continue
                    
                    if not pixel_format_set:
                        failed_params.append("PixelFormat (no suitable format could be set)")
                else:
                    failed_params.append("PixelFormat (not available or not writable)")
            except Exception as e:
                self.logger.debug(f"Error accessing PixelFormat: {str(e)}")
                failed_params.append("PixelFormat (error accessing parameter)")
            
            # Log results
            if configured_params:
                self.logger.info(f"Essential camera parameters configured: {', '.join(configured_params)}")
            
            if failed_params:
                self.logger.debug(f"Camera parameters not set: {', '.join(failed_params)}")
            
            # Configure optional parameters
            self._configure_optional_parameters()
                
        except Exception as e:
            self.logger.warning(f"Error during camera configuration: {str(e)}")

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

    def capture_frame(self, timeout_ms=5000, max_retries=2):
        """Capture a single frame from the camera with retry logic for reliability"""
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if not self.camera or not self.camera.IsOpen():
                    self.logger.error("Camera not initialized or opened")
                    return None
                
                # For stereo vision, use single frame acquisition instead of continuous
                # This prevents conflicts when multiple cameras are capturing simultaneously
                if self.is_grabbing:
                    self.stop_grabbing()
                
                # Add small delay if this is a retry to let camera settle
                if retry_count > 0:
                    import time
                    time.sleep(0.1)
                
                # Use single frame grab strategy
                grabResult = self.camera.GrabOne(timeout_ms)
                
                if grabResult.GrabSucceeded():
                    # Convert the image to BGR format
                    image = self.converter.Convert(grabResult)
                    img_array = image.GetArray()
                    
                    # Release the grab result
                    grabResult.Release()
                    
                    if retry_count > 0:
                        self.logger.debug(f"Frame capture succeeded on retry {retry_count}")
                    
                    return img_array
                else:
                    # Don't log error on first attempt, only on retries
                    if retry_count == max_retries:
                        self.logger.error("Single frame grab failed after all retries")
                    else:
                        self.logger.debug(f"Frame grab attempt {retry_count + 1} failed, retrying...")
                    
                    grabResult.Release()
                    retry_count += 1
                    continue
                    
            except Exception as e:
                if retry_count == max_retries:
                    self.logger.error(f"Failed to capture frame after {max_retries + 1} attempts: {str(e)}")
                else:
                    self.logger.debug(f"Capture attempt {retry_count + 1} failed: {str(e)}, retrying...")
                
                retry_count += 1
                continue
        
        return None

    def capture_frame_continuous(self, timeout_ms=5000):
        """Capture a frame using continuous grabbing mode (use with caution in multi-camera setups)"""
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
            self.logger.error(f"Failed to capture frame in continuous mode: {str(e)}")
            return None

    @staticmethod
    def capture_stereo_frames(camera_left, camera_right, timeout_ms=5000):
        """
        Capture frames from both cameras in sequence for stereo vision.
        Sequential capture is more reliable than simultaneous capture for most setups.
        Returns tuple (left_frame, right_frame) where either can be None if capture failed.
        """
        import time
        
        # Capture left frame first
        start_time = time.time()
        left_frame = camera_left.capture_frame(timeout_ms)
        
        # Capture right frame immediately after
        right_frame = camera_right.capture_frame(timeout_ms)
        
        capture_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        if left_frame is not None and right_frame is not None:
            camera_left.logger.debug(f"Stereo capture completed in {capture_time:.1f}ms")
        
        return left_frame, right_frame

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