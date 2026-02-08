#!/usr/bin/env python3
"""
Diagnostic script to check available GigE parameters on Basler cameras
"""

from pypylon import pylon
import sys

def main():
    print("Basler GigE Camera Parameter Diagnostic")
    print("=" * 60)

    try:
        tlFactory = pylon.TlFactory.GetInstance()
        cameras = tlFactory.EnumerateDevices()

        if len(cameras) == 0:
            print("No cameras found!")
            return

        print(f"Found {len(cameras)} camera(s). Testing first camera...")

        cam = pylon.InstantCamera(tlFactory.CreateDevice(cameras[0]))
        cam.MaxNumBuffer = 15
        cam.Open()

        print(f"\nCamera: {cam.GetDeviceInfo().GetModelName()}")
        print(f"IP: {cam.GetDeviceInfo().GetIpAddress()}")
        print("\n" + "=" * 60)
        print("CRITICAL GigE PARAMETERS:")
        print("=" * 60)

        # Check GevSCPD (Inter-Packet Delay)
        params_to_check = [
            ('GevSCPD', 'Inter-Packet Delay'),
            ('GevSCPSPacketSize', 'Packet Size'),
            ('GevSCFTD', 'Frame Transmission Delay'),
            ('GevHeartbeatTimeout', 'Heartbeat Timeout'),
            ('AcquisitionMode', 'Acquisition Mode'),
            ('ExposureTime', 'Exposure Time'),
            ('ExposureTimeAbs', 'Exposure Time (Abs)'),
            ('TriggerMode', 'Trigger Mode'),
        ]

        for param_name, description in params_to_check:
            print(f"\n{param_name} ({description}):")
            try:
                param = cam.GetNodeMap().GetNode(param_name)
                if param is not None:
                    from pypylon import genicam
                    print(f"  EXISTS: Yes")
                    access_mode = param.GetAccessMode()
                    is_writable = access_mode == genicam.RW or access_mode == genicam.WO
                    print(f"  Writable: {is_writable} (access_mode={access_mode})")
                    if is_writable:
                        try:
                            if hasattr(param, 'GetValue'):
                                current_val = param.GetValue()
                                print(f"  Current Value: {current_val}")
                            if hasattr(param, 'GetMin') and hasattr(param, 'GetMax'):
                                print(f"  Range: {param.GetMin()} to {param.GetMax()}")
                        except Exception as e:
                            print(f"  Error getting value: {e}")
                    else:
                        print(f"  Status: LOCKED (not writable, access_mode={access_mode})")
                else:
                    print(f"  EXISTS: No")
            except Exception as e:
                print(f"  ERROR: {e}")

        cam.Close()
        print("\n" + "=" * 60)
        print("Diagnostic complete!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


