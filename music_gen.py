import pygame
import streamlit as st
import pretty_midi
import tensorflow as tf
import tempfile

# Define available instruments
INSTRUMENTS = {
    "Piano": 0,
    "Flute": 73,
    "Violin": 40,
    "Clarinet": 71,  # Added Clarinet
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
        # Step size for time
        'step': tf.keras.layers.Dense(1, name='step')(x),
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
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

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
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(midi_filename)
    pygame.mixer.music.set_volume(0.8)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

    pygame.quit()

# Function to create a MIDI file from notes


def create_midi_file(output_file, notes, instrument):
    midi_data = pretty_midi.PrettyMIDI()
    instrument_program = INSTRUMENTS.get(instrument, 0)  # Default to Piano (0)
    midi_instrument = pretty_midi.Instrument(program=instrument_program)
    midi_instrument.notes.extend(notes)
    midi_data.instruments.append(midi_instrument)
    midi_data.write(output_file)
    st.success(f"MIDI file '{output_file}' created successfully.")


# Streamlit Interface
st.title("NoteSynth: Generating Musical Notes with Deep Learning")
st.write(
    """
    NoteSynth is an innovative application that combines deep learning and music synthesis. 
    The project is designed to generate and play musical notes based on trained models and user input. 
    Users can upload custom-trained model weights and MIDI files, select from a variety of instruments 
    such as Piano, Flute, Violin, and Clarinet, and generate dynamic musical compositions.
    
    """
)

# Sidebar for user inputs
with st.sidebar:
    st.header("Settings")
    model_upload = st.file_uploader(
        "Upload Model Weights", type=["keras", "h5"])
    midi_upload = st.file_uploader("Upload a MIDI File", type=["mid", "midi"])
    selected_instrument = st.selectbox(
        "Choose an Instrument", options=list(INSTRUMENTS.keys()))

# Load model if uploaded
model = None
if model_upload:
    model = load_uploaded_model(model_upload)
    if model:
        st.success("Model weights successfully loaded!")

# Show button to create and play MIDI
if model and midi_upload:
    create_button = st.button("Create MIDI File and Play")
    if create_button:
        midi_filename = 'generated_output.mid'

        # Generate notes for at least 10 seconds
        test_notes = [
            pretty_midi.Note(velocity=100, pitch=60 + (i % 12),
                             start=i * 0.5, end=(i + 1) * 0.5)
            for i in range(20)  # 20 notes, each lasting 0.5 seconds
        ]

        create_midi_file(midi_filename, test_notes, selected_instrument)
        play_music(midi_filename)
        st.write(
            f"Playing the generated MIDI file with {selected_instrument}.")
