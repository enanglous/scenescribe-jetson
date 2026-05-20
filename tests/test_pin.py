import Jetson.GPIO as GPIO
import time

# Pin Definitions
led_pin = 7 # Board pin number

# Pin Setup
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.LOW) # Set pin as an output

try:
    while True:
        GPIO.output(led_pin, GPIO.HIGH) # Turn LED on
        time.sleep(0.5)
        # GPIO.output(led_pin, GPIO.LOW) # Turn LED off
        # time.sleep(0.5)
except KeyboardInterrupt:
    print("Exiting")
    GPIO.cleanup() # Clean up all GPIOs
