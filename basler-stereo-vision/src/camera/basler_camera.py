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

    def _safe_set_parameter(self, param_names, value, description="", retries=3):
        """
        Safely set a camera parameter, trying multiple parameter names if necessary.
        Uses NodeMap API for reliable parameter access.

        Args:
            param_names: String or list of parameter names to try
            value: Value to set
            description: Description for logging
            retries: Number of times to retry setting the parameter

        Returns:
            tuple: (success, parameter_name_used)
        """
        from pypylon import genicam
        import time

        if isinstance(param_names, str):
            param_names = [param_names]
            
        for param_name in param_names:
            for attempt in range(retries):
                try:
                    # Use NodeMap API for reliable parameter access
                    node_map = self.camera.GetNodeMap()
                    param = node_map.GetNode(param_name)

                    if param is not None:
                        # Check if parameter is writable
                        access_mode = param.GetAccessMode()
                        is_writable = access_mode == genicam.RW or access_mode == genicam.WO

                        if is_writable:
                            # Set value based on parameter type
                            param.SetValue(value)

                            # Verify it was set (for critical parameters)
                            if hasattr(param, 'GetValue'):
                                try:
                                    actual_value = param.GetValue()
                                    if actual_value == value or abs(actual_value - value) < 1:
                                        self.logger.debug(f"Successfully set {param_name} = {value}")
                                        return True, param_name
                                except:
                                    # If we can't verify, assume success
                                    pass

                            self.logger.debug(f"Set {param_name} = {value}")
                            return True, param_name
                        else:
                            if attempt == 0:  # Only log once
                                self.logger.debug(f"Parameter {param_name} exists but is not writable (access_mode={access_mode})")
                            break  # Don't retry if not writable
                    else:
                        if attempt == 0:  # Only log once
                            self.logger.debug(f"Parameter {param_name} does not exist")
                        break  # Don't retry if doesn't exist

                except Exception as e:
                    if attempt < retries - 1:
                        self.logger.debug(f"Failed to set {param_name} (attempt {attempt+1}/{retries}): {str(e)}")
                        time.sleep(0.1)  # Small delay before retry
                        continue
                    else:
                        self.logger.debug(f"Failed to set {param_name} after {retries} attempts: {str(e)}")
                        break

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
            
            # CRITICAL: Set buffer count BEFORE opening camera
            # High-resolution GigE cameras need more buffers to prevent underruns
            # Default is 5, we use 15 for 12MP cameras
            self.camera.MaxNumBuffer = 15

            # Open the camera
            self.camera.Open()
            
            # IMPORTANT: Stop any ongoing grabbing first (camera might be in invalid state)
            if self.camera.IsGrabbing():
                self.logger.warning("Camera was already grabbing, stopping first...")
                self.camera.StopGrabbing()
                self.is_grabbing = False

            # Get actual IP address after connection
            if not self.camera_ip and hasattr(selected_camera, 'GetIpAddress'):
                self.camera_ip = selected_camera.GetIpAddress()
            
            # CRITICAL: Many camera parameters require acquisition to be stopped
            # Ensure camera is in correct state before configuration
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            # Set camera parameters for optimal performance
            self._configure_camera()
            
            # Create image converter for color conversion
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self.logger.info(f"Basler camera initialized successfully at IP: {self.camera_ip}")
            self.logger.info(f"Camera buffer count set to: {self.camera.MaxNumBuffer.Value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {str(e)}")
            return False

    def _configure_camera(self):
        """Configure camera parameters for stereo vision"""
        configured_params = []
        failed_params = []
        
        try:
            # ===== CRITICAL GigE NETWORK PARAMETERS (Must configure FIRST) =====
            # These parameters prevent "buffer incompletely grabbed" errors

            # For GigE cameras, we need to access the StreamGrabber parameters
            # These control the network transport layer
            try:
                # Get the stream grabber parameters (for GigE cameras)
                if hasattr(self.camera, 'StreamGrabber'):
                    stream_grabber = self.camera.StreamGrabber

                    # MaxNumBuffer - increase buffer count
                    if hasattr(stream_grabber, 'MaxNumBuffer'):
                        try:
                            stream_grabber.MaxNumBuffer.SetValue(10)
                            configured_params.append("StreamGrabber.MaxNumBuffer=10")
                        except Exception as e:
                            self.logger.debug(f"Could not set MaxNumBuffer: {e}")

                    # MaxBufferSize - increase buffer size for high-res cameras
                    if hasattr(stream_grabber, 'MaxBufferSize'):
                        try:
                            # For 12MP camera: 4024x3036x3 = ~36MB per frame
                            stream_grabber.MaxBufferSize.SetValue(40000000)  # 40MB
                            configured_params.append("StreamGrabber.MaxBufferSize=40MB")
                        except Exception as e:
                            self.logger.debug(f"Could not set MaxBufferSize: {e}")
            except Exception as e:
                self.logger.debug(f"Could not access StreamGrabber parameters: {e}")

            # 1. Packet Size - CRITICAL: Reduce from default 9000 to standard MTU
            # Jumbo frames (9000) can cause packet loss if network isn't configured for it
            success, param_used = self._safe_set_parameter(['GevSCPSPacketSize'], 1500)
            if success:
                configured_params.append(f"PacketSize=1500 (reduced from jumbo)")
            else:
                self.logger.warning("Failed to set GevSCPSPacketSize - this is CRITICAL for reliability!")

            # 2. Inter-Packet Delay - CRITICAL for preventing packet loss with high-res cameras
            # Default is 0, we need significant delay for 12MP cameras
            # Higher delay = more time between packets = less congestion
            # For acA4024-8gc (12MP), we need substantial delay due to large frame size
            success, param_used = self._safe_set_parameter(['GevSCPD'], 20000)
            if success:
                configured_params.append(f"InterPacketDelay=20000us (CRITICAL)")
            else:
                self.logger.warning("Failed to set GevSCPD (Inter-Packet Delay) - this is CRITICAL!")

            # 3. Frame Transmission Delay
            success, param_used = self._safe_set_parameter(['GevSCFTD'], 0)
            if success:
                configured_params.append(f"FrameTransmissionDelay=0")

            # 4. Heartbeat Timeout
            success, param_used = self._safe_set_parameter(['GevHeartbeatTimeout'], 30000)
            if success:
                configured_params.append(f"HeartbeatTimeout=30000ms ({param_used})")

            # ===== ACQUISITION PARAMETERS =====

            # Enable free-running mode (no trigger)
            success, param_used = self._safe_set_parameter('AcquisitionMode', 'Continuous')
            if success:
                configured_params.append(f"AcquisitionMode=Continuous ({param_used})")

            # Set exposure time
            success, param_used = self._safe_set_parameter(['ExposureTime', 'ExposureTimeAbs'], 10000)
            if success:
                configured_params.append(f"ExposureTime=10000us ({param_used})")

            # Set gain
            success, param_used = self._safe_set_parameter(['Gain'], 1.0)
            if not success:
                success, param_used = self._safe_set_parameter(['GainRaw'], 100)
            
            if success:
                configured_params.append(f"Gain ({param_used})")

            # Set pixel format to the best available option
            pixel_format_set = False
            try:
                if hasattr(self.camera, 'PixelFormat'):
                    param = self.camera.PixelFormat
                    if hasattr(param, 'IsWritable') and param.IsWritable:
                        available_formats = list(param.Symbolics)
                        format_priority = ['RGB8', 'BayerRG8', 'BayerGB8', 'BayerGR8', 'BayerBG8', 'Mono8']

                        for fmt in format_priority:
                            if fmt in available_formats:
                                try:
                                    param.SetValue(fmt)
                                    configured_params.append(f"PixelFormat={fmt}")
                                    pixel_format_set = True
                                    break
                                except Exception as e:
                                    self.logger.debug(f"Failed to set PixelFormat to {fmt}: {str(e)}")
                                    continue
            except Exception as e:
                self.logger.debug(f"Error accessing PixelFormat: {str(e)}")

            # Log results
            if configured_params:
                self.logger.info(f"Camera parameters configured: {', '.join(configured_params)}")
            else:
                self.logger.warning("No camera parameters were successfully configured!")

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

    def capture_frame(self, timeout_ms=15000, max_retries=2):
        """Capture a single frame from the camera with retry logic for reliability

        Args:
            timeout_ms: Timeout in milliseconds (default 15000 for high-res GigE cameras - 12MP takes time!)
            max_retries: Maximum number of retry attempts
        """
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                if not self.camera or not self.camera.IsOpen():
                    self.logger.error("Camera not initialized or opened")
                    return None
                
                # IMPORTANT: Stop grabbing if it's running - GrabOne() handles its own acquisition
                # The "device physically removed" error often happens when grabbing state is inconsistent
                if self.is_grabbing or self.camera.IsGrabbing():
                    try:
                        self.camera.StopGrabbing()
                        self.is_grabbing = False
                        self.logger.debug("Stopped continuous grabbing before GrabOne")
                    except Exception as e:
                        self.logger.debug(f"Error stopping grab (continuing): {e}")

                # Add small delay if this is a retry to let camera settle
                if retry_count > 0:
                    import time
                    time.sleep(0.2)
                    self.logger.debug(f"Retry attempt {retry_count}")

                # GrabOne internally starts and stops grabbing for a single frame
                # This is the recommended way for multi-camera setups
                grabResult = self.camera.GrabOne(timeout_ms)
                
                if grabResult.GrabSucceeded():
                    # Convert the image to BGR format
                    image = self.converter.Convert(grabResult)
                    img_array = image.GetArray()
                    
                    # Release the grab result
                    grabResult.Release()
                    
                    if retry_count > 0:
                        self.logger.info(f"Frame capture succeeded on retry {retry_count}")

                    return img_array
                else:
                    # Log the actual error from grab result
                    error_code = grabResult.GetErrorCode() if hasattr(grabResult, 'GetErrorCode') else 'Unknown'
                    error_desc = grabResult.GetErrorDescription() if hasattr(grabResult, 'GetErrorDescription') else 'No description'

                    grabResult.Release()

                    if retry_count == max_retries:
                        self.logger.error(f"Frame grab failed after all retries. Error: {error_code} - {error_desc}")
                    else:
                        self.logger.debug(f"Frame grab attempt {retry_count + 1} failed, retrying...")

                    retry_count += 1
                    continue
                    
            except Exception as e:
                error_msg = str(e)

                if retry_count == max_retries:
                    self.logger.error(f"Failed to capture frame after {max_retries + 1} attempts: {error_msg}")
                    self.logger.error("Possible causes:")
                    self.logger.error("  - Camera in invalid state")
                    self.logger.error("  - Trigger mode enabled (should be off)")
                    self.logger.error("  - Camera hardware issue")
                else:
                    self.logger.debug(f"Capture attempt {retry_count + 1} failed: {error_msg}")

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