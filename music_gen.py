import pygame
import streamlit as st
import pandas as pd
import numpy as np
import pretty_midi
import collections
import glob
import pathlib
import os
import altair as alt
import tensorflow as tf

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
    input_shape = (seq_length, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
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


# Load the model
model_path = "W:/front/model_500_100_epocs.keras"  # Update with your model path
model = create_model(seq_length=10)

try:
    model.load_weights(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")

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


def load_midi_files(data_dir_path):
    data_dir = pathlib.Path(data_dir_path)
    if not data_dir.exists():
        st.error(f"Directory '{data_dir}' does not exist.")
        return []

    midi_files = glob.glob(str(data_dir / '*.midi'), recursive=True)
    return midi_files

# Function to convert MIDI file to notes


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)

    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

# Function to preprocess input notes


def preprocess_input(notes):
    notes = np.array(notes)
    notes = (notes - notes.min()) / (notes.max() - notes.min())
    return notes


def generate_music(seed_notes, generation_length, model):
    input_data = preprocess_input(seed_notes)

    if input_data.shape[0] < 10:
        padding = np.zeros((10 - input_data.shape[0], input_data.shape[1]))
        input_data = np.vstack((padding, input_data))
    elif input_data.shape[0] > 10:
        input_data = input_data[-10:]

    generated_notes = []

    for _ in range(generation_length):
        input_data_expanded = np.expand_dims(input_data, axis=0)
        predicted_note = model.predict(input_data_expanded, verbose=0)

        pitch = np.argmax(predicted_note['pitch'][0])
        step = predicted_note['step'][0][0]
        duration = predicted_note['duration'][0][0]

        generated_notes.append([pitch, step, duration])

        input_data = np.delete(input_data, 0, axis=0)
        input_data = np.vstack((input_data, [[pitch, step, duration]]))

    return np.array(generated_notes)

# Convert generated notes to MIDI and play


def notes_to_midi(generated_notes, instrument_program=0, midi_path="generated_music.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=instrument_program)

    start_time = 0
    for note in generated_notes:
        pitch, step, duration = int(note[0]), note[1], note[2]
        start_time += step
        end_time = start_time + duration
        midi_note = pretty_midi.Note(
            velocity=100, pitch=pitch, start=start_time, end=end_time
        )
        instrument.notes.append(midi_note)

    pm.instruments.append(instrument)

    try:
        pm.write(midi_path)
        st.success(f"MIDI file written successfully to {midi_path}")
    except Exception as e:
        st.error(f"Error writing MIDI file: {e}")

    return midi_path

# Function for data analysis and visualization


def visualize_notes(notes: pd.DataFrame):
    st.subheader("Pitch Distribution")
    pitch_chart = alt.Chart(notes).mark_bar().encode(
        x='pitch:Q',
        y='count()',
    ).properties(width=600, height=400)
    st.altair_chart(pitch_chart)

    st.subheader("Note Step Distribution")
    step_chart = alt.Chart(notes).mark_bar().encode(
        x='step:Q',
        y='count()',
    ).properties(width=600, height=400)
    st.altair_chart(step_chart)

    st.subheader("Note Duration Distribution")
    duration_chart = alt.Chart(notes).mark_bar().encode(
        x='duration:Q',
        y='count()',
    ).properties(width=600, height=400)
    st.altair_chart(duration_chart)


# Example usage in Streamlit app
st.title("AI-Powered Music Generation App")

uploaded_file = st.sidebar.file_uploader(
    "Upload a MIDI file", type=["mid", "midi"])
uploaded_midi_path = None

if uploaded_file is not None:
    st.write(f"Processing file: {uploaded_file.name}")
    if not os.path.exists("temp"):
        os.makedirs("temp")

    uploaded_midi_path = os.path.join("temp", uploaded_file.name)
    with open(uploaded_midi_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    notes_df = midi_to_notes(uploaded_midi_path)
    st.subheader("Extracted Notes")
    st.dataframe(notes_df)

    # Display data analysis charts
    visualize_notes(notes_df)

    seed_notes_input = st.sidebar.text_input(
        "Enter seed note sequence (pitch,duration pairs, comma-separated)", value="60,0.5,62,0.5,64,0.5,65,0.5,67,0.5")
    generation_length = st.sidebar.slider(
        "Length of generated sequence (in seconds)", min_value=1, max_value=30, value=10)  # Allow up to 30 seconds

    # Calculate number of notes based on desired duration
    notes_per_second = 10  # Adjust based on how many notes you want to play per second
    total_notes = generation_length * notes_per_second

    selected_instrument_name = st.sidebar.selectbox(
        "Select Instrument", options=list(INSTRUMENTS.keys()))
    selected_instrument = INSTRUMENTS[selected_instrument_name]

    if st.sidebar.button("Generate Music"):
        # Create seed_notes from input
        seed_notes = []
        seed_notes_list = seed_notes_input.split(',')
        for i in range(0, len(seed_notes_list), 2):
            pitch = int(seed_notes_list[i])
            duration = float(seed_notes_list[i + 1])
            # Using 0.1 for the step value as a placeholder
            seed_notes.append([pitch, 0.1, duration])

        generated_notes = generate_music(
            np.array(seed_notes), total_notes, model)

        midi_path = notes_to_midi(
            generated_notes, instrument_program=selected_instrument, midi_path="temp/generated_music.mid")
        st.write("Playing Generated MIDI...")

        try:
            play_music(midi_path)  # Play the MIDI file
        except Exception as e:
            st.error(f"Error during playback: {e}")
