import streamlit as st
import numpy as np
import torch
from model import CpGCounter


# Load the trained PyTorch model
model = CpGCounter(input_size=128, hidden_size=16, num_layers=1)
model.load_state_dict(torch.load('weights.pth'))
model.eval()


def pad_sequence(input_list, fixed_length=128):
    padding_length = fixed_length - len(input_list)
    padded_list = np.pad(input_list, (0, padding_length), mode='constant', constant_values=0)
    return padded_list


# Define DNA alphabet mapping
alphabet = 'NACGT'
dna2int = {a: i for i, a in enumerate(alphabet)}


# Function to preprocess input sequence
def preprocess_sequence(input_sequence):
    # Ensure the sequence contains only valid DNA characters ('NACGT')
    sequence = input_sequence.upper()
    valid_chars = set('NACGT')
    if all(char in valid_chars for char in sequence):
        # Convert the sequence to a list of integers
        seq_int = [dna2int[c] for c in sequence]

        # Pad the sequence
        seq_int = pad_sequence(seq_int)
        seq_tensor = torch.tensor(seq_int, dtype=torch.float32).unsqueeze(0)
        return seq_tensor
    else:
        return None  # Invalid sequence
 
 
# Streamlit app
st.title("CpG Prediction App")

# Input DNA sequence
input_sequence = st.text_input("Enter DNA Sequence (e.g., NCACANNTNCGGAGGCGNA):")

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    if input_sequence:
        with st.spinner("Predicting..."):
            # Preprocess the input sequence
            input_tensor = preprocess_sequence(input_sequence)

            if input_tensor is not None:
                # Make a prediction
                with torch.no_grad():
                    output = model(input_tensor.unsqueeze(0))
                prediction = output.item()
                # Display the prediction
                st.success(f"Predicted CpG Count: {prediction:.2f}")
            else:
                st.warning("Invalid DNA sequence. Please enter a valid DNA sequence.")
    else:
        st.warning("Please enter a DNA sequence.")

# Footer
st.sidebar.markdown("Autonomize.ai")

