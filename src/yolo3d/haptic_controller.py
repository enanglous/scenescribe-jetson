import os
import threading
import time

class SysfsPWM:
    """Direct hardware PWM controller to bypass Jetson.GPIO bugs."""
    def __init__(self, platform_address, pwm_freq=100):
        self.address = platform_address
        self.period_ns = int(1e9 / pwm_freq)
        self.pwm_base = self._find_pwm_base()
        self.pwm_path = os.path.join(self.pwm_base, 'pwm0') if self.pwm_base else None

        if not self.pwm_base:
            raise RuntimeError(f"PWM controller {self.address} not found in sysfs.")

        # Export the PWM channel to userspace if not already exported
        if not os.path.exists(self.pwm_path):
            try:
                with open(os.path.join(self.pwm_base, 'export'), 'w') as f:
                    f.write('0')
                time.sleep(0.1) # Wait a moment for the OS to create the pwm0 directory
            except Exception as e:
                print(f"Failed to export {self.address}: {e}")

        # Initialize the period and ensure the duty cycle starts at 0 (OFF)
        if os.path.exists(self.pwm_path):
            self._write('period', self.period_ns)
            self.ChangeDutyCycle(0)
            self._write('enable', 1)

    def _find_pwm_base(self):
        """Scans the system to find which pwmchip number Linux assigned to the address."""
        sysfs_dir = '/sys/class/pwm'
        if not os.path.exists(sysfs_dir): return None
        
        for d in os.listdir(sysfs_dir):
            if d.startswith('pwmchip'):
                device_path = os.path.realpath(os.path.join(sysfs_dir, d, 'device'))
                if self.address in device_path:
                    return os.path.join(sysfs_dir, d)
        return None

    def _write(self, filename, value):
        """Helper to write values to the sysfs hardware files."""
        try:
            with open(os.path.join(self.pwm_path, filename), 'w') as f:
                f.write(str(value))
        except Exception:
            pass

    def ChangeDutyCycle(self, percentage):
        """Mirrors the Jetson.GPIO syntax for easy swapping."""
        duty_ns = int(self.period_ns * (percentage / 100.0))
        self._write('duty_cycle', duty_ns)

    def stop(self):
        """Turns off the signal."""
        self.ChangeDutyCycle(0)
        self._write('enable', 0)


class HapticController:
    """
    Handles non-blocking PWM haptic feedback for Jetson Orin Nano with safe shutdown.
    """
    def __init__(self, pwm_freq=100, vibration_time=1.0, duty_cycle=20):
        self.vibration_time = vibration_time
        self.duty_cycle = duty_cycle 

        # Map to the direct kernel platform addresses instead of buggy pin numbers
        # Pin 32 is 32e0000.pwm, Pin 33 is 32c0000.pwm
        self.right_pwm = SysfsPWM('32e0000.pwm', pwm_freq)
        self.left_pwm = SysfsPWM('32e0000.pwm', pwm_freq)

        self.is_left_active = False
        self.is_right_active = False
        self.stop_event = threading.Event()

    def _vibrate_worker(self, motor_pwm, side):
        try:
            motor_pwm.ChangeDutyCycle(self.duty_cycle)
            
            interrupted = self.stop_event.wait(self.vibration_time)

            if not interrupted:
                motor_pwm.ChangeDutyCycle(0)
                
        except Exception:
            pass 
        finally:
            if side == 'left':
                self.is_left_active = False
            else:
                self.is_right_active = False

    def trigger_left(self):
        if not self.is_left_active and not self.stop_event.is_set():
            self.is_left_active = True
            threading.Thread(target=self._vibrate_worker, args=(self.left_pwm, 'left'), daemon=True).start()

    def trigger_right(self):
        if not self.is_right_active and not self.stop_event.is_set():
            self.is_right_active = True
            threading.Thread(target=self._vibrate_worker, args=(self.right_pwm, 'right'), daemon=True).start()

    def process_command(self, action_string):
        if "TURN LEFT" in action_string:
            self.trigger_left()
        elif "TURN RIGHT" in action_string:
            self.trigger_right()

    def cleanup(self):
        self.stop_event.set()
        self.stop_event.wait(0.1)

        try:
            self.left_pwm.stop()
            self.right_pwm.stop()
        except Exception:
            pass

def main():
    haptic=HapticController()
    while True:
        # Test Left
        print("Testing Left (Pin 32)")
        haptic.process_command("TURN LEFT")
        time.sleep(3) # Wait for vibration to finish
        
        # Test Right
        print("Testing Right (Pin 33)")
        haptic.process_command("TURN RIGHT")
        time.sleep(3)

    haptic.cleanup()

if __name__ == "__main__":
    main()