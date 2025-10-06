import serial
import matplotlib.pyplot as plt


port_name = "/dev/ttyUSB0"
baudrate = 115200

conn = serial.Serial(port=port_name,baudrate=baudrate)

# conn.open()
data = []
try:
    while True:
        if conn.in_waiting > 0:
            line = conn.readline().decode("utf-8").strip()
            # try:
            #     value = int(line)
            #     print(f"Received int: {value}")
            # except ValueError:
            #     print(f"Non-integer received: {line}")
            print(line)
            data.append(line)
except KeyboardInterrupt:
    print("\nStopping.")
finally:
    conn.close()

with open("STM_data.txt", 'w') as file:
    for val in data:
        file.write(f"{val}\n")


# plt.plot(data)
# plt.xlabel("Sample index")
# plt.ylabel("Value")
# plt.title("Signal from STM_data.txt")
# plt.legend()
# plt.grid(True)
# plt.show()
