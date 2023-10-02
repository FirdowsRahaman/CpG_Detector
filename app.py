import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# Define the LSTM-based model
class CpGCounter(nn.Module):
    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1):
        super(CpGCounter, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        logits = self.classifier(out[:, -1, :])  
        return logits


# Load the pre-trained PyTorch model
model = CpGCounter(input_size=128, hidden_size=16, num_layers=1, bidirectional=True)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Function to preprocess input sequence
def preprocess_sequence(sequence):
    # Convert the sequence to a list of integers
    seq_int = [dna2int[c] for c in sequence]
    # Pad the sequence
    seq_tensor = torch.tensor(seq_int)
    return seq_tensor

# Streamlit app
st.title("CpG Prediction App")

# Input DNA sequence
input_sequence = st.text_input("Enter DNA Sequence (e.g., NCACANNTNCGGAGGCGNA):")

# Define DNA alphabet mapping
alphabet = 'NACGT'
dna2int = {a: i for i, a in enumerate(alphabet)}

# Make a prediction when the user clicks the "Predict" button
if st.button("Predict"):
    if input_sequence:
        with st.spinner("Predicting..."):
            # Preprocess the input sequence
            input_tensor = preprocess_sequence(input_sequence)
            # Make a prediction
            with torch.no_grad():
                output = model(input_tensor.unsqueeze(0))
            prediction = output.item()
            # Display the prediction
            st.success(f"Predicted CpG Count: {prediction:.2f}")
    else:
        st.warning("Please enter a DNA sequence.")

# Footer
st.sidebar.markdown("By Your Name")

