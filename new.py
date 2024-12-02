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

# Function to generate notes using the model


def generate_notes_from_model(model, seq_length=10):
    # Start with a random initial sequence or zeros (or you could input any sequence)
    # [seq_length, 3] -> pitch/step/duration
    initial_sequence = tf.zeros((1, seq_length, 3), dtype=tf.float32)

    generated_notes = []
    current_sequence = initial_sequence

    for _ in range(seq_length):
        # Predict the next sequence using the model
        predictions = model(current_sequence)

        # Extract pitch, step, and duration predictions
        pitch_pred = predictions['pitch']
        step_pred = predictions['step']
        duration_pred = predictions['duration']

        # Convert the predictions into actual note values
        # Convert pitch to integer
        pitch = tf.argmax(pitch_pred, axis=-1).numpy()[0]
        step = step_pred.numpy()[0]  # Direct prediction for step
        duration = duration_pred.numpy()[0]  # Direct prediction for duration

        # Ensure pitch, step, and duration are all floats (if necessary)
        pitch = float(pitch)
        step = float(step)
        duration = float(duration)

        # Create a pretty_midi note using predicted pitch, step (time), and duration
        start_time = step  # Use 'step' as the start time of the note
        end_time = start_time + duration
        note = pretty_midi.Note(velocity=100, pitch=int(
            pitch), start=start_time, end=end_time)

        generated_notes.append(note)

        # Update the sequence for the next prediction (Shift the sequence)
        current_sequence = tf.concat([current_sequence[:, 1:, :], tf.reshape(
            tf.convert_to_tensor([[pitch, step, duration]], dtype=tf.float32), (1, 1, 3))], axis=1)

    return generated_notes


# Streamlit Interface
st.title("Music Generation from LSTM Network")

# Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    # Upload Model Weights
    model_upload = st.file_uploader(
        "Upload Model Weights", type=["keras", "h5"])

    # Instrument selection
    selected_instrument = st.selectbox(
        "Choose an Instrument", options=list(INSTRUMENTS.keys()))

    # Select notes length for visualization
    num_notes = st.slider("Number of Notes", min_value=5,
                          max_value=20, value=5)

# Load model if uploaded
model = None
if model_upload:
    model = load_uploaded_model(model_upload)
    if model:
        st.success("Model weights successfully loaded!")

# Show buttons to generate and play music after model is loaded
if model:
    col1, col2 = st.columns(2)
    with col1:
        generate_button = st.button("Generate MIDI File")
    with col2:
        play_button = st.button("Play MIDI File")

    # Generate MIDI file when button is clicked
    if generate_button:
        notes = generate_notes_from_model(model)
        midi_filename = 'generated_output.mid'
        create_midi_file(midi_filename, notes, selected_instrument)
        st.audio(midi_filename)

    # Play the generated MIDI file when button is clicked
    if play_button:
        midi_filename = 'generated_output.mid'
        play_music(midi_filename)
        st.write("Music is playing...")
