import pygame
import streamlit as st
import pretty_midi
import tensorflow as tf
import tempfile
import shutil
import altair as alt
import pandas as pd

# Define available instruments
INSTRUMENTS = {
    "Piano": 0,
    "Guitar": 24,
    "Flute": 73,
    "Violin": 40,
}

# Define the custom loss function that encourages non-negative outputs


def mse_with_positive_pressure(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    positive_pressure_loss = tf.reduce_mean(tf.maximum(0.0, -y_pred))
    return mse_loss + positive_pressure_loss

# Define the model architecture


def create_model(seq_length):
    input_shape = (seq_length, 3)  # seq_length, pitch/step/duration
    learning_rate = 0.005

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        # 128 possible pitches
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),  # Step size for time
        # Duration of the note
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, optimizer=optimizer)

    return model

# Function to handle the uploaded model


def load_uploaded_model(uploaded_file):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load the model weights
        model = create_model(seq_length=10)
        model.load_weights(tmp_file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to play the MIDI file using Pygame


def play_music(midi_filename):
    pygame.init()
    pygame.display.set_mode((1, 1))
    pygame.mixer.init(frequency=44100, size=-16, channels=2,
                      buffer=2048)  # Increased buffer size
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.set_volume(0.8)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)  # Delay to prevent high CPU usage
        for event in pygame.event.get():  # Handle events to avoid blocking
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    pygame.quit()  # Quit pygame after the music has finished playing

# Function to create a MIDI file from notes


def create_midi_file(output_file, notes, instrument):
    # Create a PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI()

    # Create an instrument based on the selected instrument and add it to the PrettyMIDI object
    instrument_program = INSTRUMENTS.get(instrument, 0)
    midi_instrument = pretty_midi.Instrument(program=instrument_program)

    # Add notes to the instrument
    midi_instrument.notes.extend(notes)

    # Add the instrument to the PrettyMIDI object
    midi_data.instruments.append(midi_instrument)

    # Write the MIDI data to a file
    midi_data.write(output_file)
    st.success(f"MIDI file '{output_file}' created successfully.")


# Streamlit Interface
st.title("Music Generation from LSTM Network")

# Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    # Upload Model Weights
    model_upload = st.file_uploader(
        "Upload Model Weights", type=["keras", "h5"])

    # Upload an Input MIDI File
    midi_upload = st.file_uploader("Upload a MIDI File", type=["mid", "midi"])

    # Instrument selection
    selected_instrument = st.selectbox(
        "Choose an Instrument", options=list(INSTRUMENTS.keys()))

# Load model if uploaded
model = None
if model_upload:
    model = load_uploaded_model(model_upload)
    if model:
        st.success("Model weights successfully loaded!")

# Show the button to play music after the model is loaded and an input MIDI file is uploaded
if model and midi_upload:
    play_button = st.button("Play MIDI File")

    # Play the uploaded MIDI file when the button is clicked
    if play_button:
        midi_filename = 'uploaded_input.mid'
        with open(midi_filename, 'wb') as f:
            f.write(midi_upload.read())
        play_music(midi_filename)
        st.write("Playing the uploaded MIDI file...")
