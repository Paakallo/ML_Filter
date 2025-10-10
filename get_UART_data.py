import serial
import matplotlib.pyplot as plt


port_name = "/dev/ttyUSB0"
baudrate = 115200

conn = serial.Serial(port=port_name,baudrate=baudrate, timeout=1)

# conn.open()
input_data = []
inf_time = []
ideal_data = []
rec_data = []

try:
    while True:
        if conn.in_waiting > 0:
            line = conn.readline().decode("utf-8").strip()
            print(line)
            name = line.split(":")[0].strip()
            number = float(line.split(":")[1].strip())
            if name == "Noise":        
                input_data.append(number)
            if name == "Ideal":
                ideal_data.append(number)
            if name == "Output":
                rec_data.append(number)
            if name == "Inf_time":
                inf_time.append(number)

except KeyboardInterrupt:
    print("\nStopping.")
finally:
    conn.close()

with open("rec_data.txt", 'w') as file:
    for val in rec_data:
        file.write(f"{val}\n")

with open("inf_time.txt", 'w') as file:
    for val in inf_time:
        file.write(f"{val}\n")

with open("input_data.txt", 'w') as file:
    for val in input_data:
        file.write(f"{val}\n")

with open("ideal_data.txt", 'w') as file:
    for val in ideal_data:
        file.write(f"{val}\n")
