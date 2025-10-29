import numpy as np
from qtm.pseudo import UPFv2Data
import sys

def main(filename):
    try:
        # Assuming UPFv2Data is used to process the file
        data = UPFv2Data(filename)
        print("File processed successfully.")
        # Add further processing logic here if needed
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: atwfc_gen.py <filename.upf>")
    else:
        main(sys.argv[1])