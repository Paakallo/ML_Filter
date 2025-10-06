import serial
import matplotlib.pyplot as plt


port_name = "/dev/ttyUSB0"
baudrate = 115200

conn = serial.Serial(port=port_name,baudrate=baudrate)

# conn.open()
input_data = []
ideal_data = []
rec_data = []

try:
    i = 0
    while True:
        if conn.in_waiting > 0:
            line = conn.readline().decode("utf-8").strip()
            i+=1
            print(line)
            if i == 1:
                input_data.append(line)
            elif i == 2:
                ideal_data.append(line)
            elif i == 3:
                rec_data.append(line)
                i = 0
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    conn.close()

with open("rec_data.txt", 'w') as file:
    for val in rec_data:
        file.write(f"{val}\n")

with open("ideal_data.txt", 'w') as file:
    for val in ideal_data:
        file.write(f"{val}\n")

with open("input_data.txt", 'w') as file:
    for val in input_data:
        file.write(f"{val}\n")

