import dataclasses
import io
import random
import uuid

import streamlit as st
from streamlit_extras.let_it_rain import rain
from streamlit_extras.streaming_write import write
import plotly.express as px

import requests
import time
from PIL import Image

import pandas as pd

st.set_page_config(
    page_title="Bruno Sudré",
    page_icon="🐘",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Bruno Sudré's Animal Hangman"
    }
)

st.markdown(
    """
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ANIMAL_CNN_API_URL = "https://animals-prediction-api.onrender.com"
GET_ANIMAL_PREDICTION_URL = ANIMAL_CNN_API_URL + "/predict"
POST_TRAIN_REQUEST_URL = ANIMAL_CNN_API_URL + "/train-request"
GET_TRAIN_REQUESTS_URL = ANIMAL_CNN_API_URL + "/train-requests"

MAX_LIVES = 6
CLASS_NAMES = ['dog', 'horse', 'elephant', 'butterfly', 'chicken',
               'cat', 'cow', 'sheep', 'spider', 'squirrel']


@dataclasses.dataclass
class GameState:
    def __init__(self, prediction=None):
        self.id = str(uuid.uuid4())
        self.selected_image = None
        self.prediction = prediction
        self.pred_probabilities = None
        self.is_game_active = False
        self.correct_guesses = set()
        self.incorrect_guesses = set()
        self.is_training_requested = False


@st.cache_resource
def create_object() -> GameState:
    return GameState()


game_state = create_object()

HANGMAN_STEPS: list[str] = ['''
  +---+
  |   |
      |
      |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
      |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
  |   |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|   |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\\  |
      |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\\  |
 /    |
      |
=========''', '''
  +---+
  |   |
  O   |
 /|\\  |
 / \\  |
      |
=========''']


def reset_game():
    """Reset the game state for a new round."""
    with st.spinner("**Resetting game...**"):
        guess_inputs = [key for key in st.session_state.keys() if key.startswith("guess_input")]
        for guess_input in guess_inputs:
            del st.session_state[guess_input]

        st.session_state.clear()
        st.cache_resource.clear()
        time.sleep(0.8)
        st.rerun()


# if "session_id" not in st.session_state:
#     st.session_state["session_id"] = str(uuid.uuid4())
#     st.session_state["is_page_reload"] = False
# else:
#     reset_game()


def draw_hangman():
    used_lives = len(game_state.incorrect_guesses)
    hangman = HANGMAN_STEPS[used_lives]
    st.markdown(f"```\n{hangman}\n```")


def find_indexes(s, ch):
    return [i for i, letter in enumerate(s) if letter.lower() == ch.lower()]


def has_won_game():
    return len(game_state.correct_guesses) == len(set(game_state.prediction))


def has_lost_game():
    return len(game_state.incorrect_guesses) == MAX_LIVES


def has_finished_game():
    return (len(game_state.incorrect_guesses) == MAX_LIVES - 1
            or len(game_state.correct_guesses) + 1 == len(set(game_state.prediction)))


def verify_letter_matches(guess):
    guess = guess.lower()
    prediction = game_state.prediction.lower()
    if guess:
        if guess.strip():
            if guess in prediction:
                st.toast(f"Correct guess: {guess} ✅")
                game_state.correct_guesses.add(guess)
                indexes = find_indexes(prediction, guess)
                for i in indexes:
                    letter_id = f"letter_{i}"
                    st.session_state[letter_id] = guess.upper()
            else:
                if len(game_state.incorrect_guesses) < len(HANGMAN_STEPS):
                    game_state.incorrect_guesses.add(guess)
                    st.toast(f"Incorrect guess: {guess} ❌")
        else:
            st.toast("You should enter a letter, remove the space! ⚠️")


def generate_fields():
    left_column, right_column = st.columns(2)

    prediction = game_state.prediction
    with left_column:
        guess = st.text_input(label="Try a letter", max_chars=1, key=f"guess_input",
                              disabled=has_finished_game())

    verify_letter_matches(guess)

    with right_column:
        draw_hangman()

    columns = st.columns(len(prediction))

    for i, col in enumerate(columns):
        col.text_input(label=f"{i + 1}",
                       max_chars=1,
                       key=f"letter_{i}",
                       disabled=True)


def request_training(image_file, correct_class):
    try:
        with st.spinner("Request image training..."):
            time.sleep(2)
            bytes_data = game_state.selected_image.getvalue()

            files = {
                "file": (image_file.name, io.BytesIO(bytes_data), image_file.type)
            }

            data = {
                "class_name": correct_class
            }

            response = requests.post(
                url=POST_TRAIN_REQUEST_URL,
                data=data,
                files=files
            )

            if response.status_code == 201:
                game_state.is_training_requested = True
                st.success("The training on the image was requested! Thanks for your feedback. 😎")
            else:
                st.error(f"Error on request training, status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error on requesting training... {e}")


@st.dialog("Have you found a wrong prediction?")
def feedback_dialog():
    st.write(f"We apologize for the wrong prediction. Would you like to help us improve the model?")
    st.write(
        "Below you can enter what is the correct animal name for the image, however remember that our model"
        f" predicts only for these animals: :green[**{', '.join(CLASS_NAMES)}**].")
    correct_class = st.text_input("Enter the correct class name").strip()
    if st.button("Submit"):
        if not game_state.is_training_requested:
            if correct_class.lower() == game_state.prediction.lower():
                st.info("This is already the current prediction! Keep enjoying the game. 😄")
            elif correct_class.lower() in list(map(lambda x: x.lower(), CLASS_NAMES)):
                request_training(game_state.selected_image, correct_class.lower())
            else:
                st.error("You can only insert an available animal name")
        else:
            st.info("The training for the image was already requested. Keep enjoying the game. 😄")


def get_training_requests_status():
    try:
        response = requests.get(url=GET_TRAIN_REQUESTS_URL)
        return response.json()
    except Exception as e:
        st.error(f"Error on requesting training requests status... {e}")


def show_game():
    if not game_state.is_game_active:
        with st.spinner("**Creating game...**"):
            game_state.is_game_active = True
            time.sleep(0.8)

    generate_fields()

    if len(game_state.incorrect_guesses) > 0:
        st.subheader(f"Incorrect Guesses: :red[{', '.join(game_state.incorrect_guesses)}]")

    if has_won_game():
        st.success("You won!")
        st.balloons()
        if st.button("New Game", key="new_game_button"):
            reset_game()

    if has_lost_game():
        st.error(f"You lost! Correct answer: **{game_state.prediction.lower()}**")
        rain(
            emoji="❌",
            font_size=54,
            falling_speed=3,
            animation_length="looser",
        )
        if st.button("Try again", key="try_again_button"):
            reset_game()

    if has_won_game() or has_lost_game():
        with st.expander("Explanation"):
            left_column, right_column = st.columns(2)
            left_column.write("This was the random selected image:")

            bytes_data = game_state.selected_image.getvalue()
            pil_image = Image.open(io.BytesIO(bytes_data))
            left_column.image(pil_image, caption=game_state.prediction.upper(), use_container_width=True)

            right_column.write("This were the prediction probabilities for this image:")
            for key, value in game_state.pred_probabilities.items():
                right_column.write(f"- **{key.upper()}**: :green[{value * 100:.4f}%]")

            if st.button("Have you found a wrong prediction?", key="feedback_btn"):
                feedback_dialog()

            st.write("Currently, we have this requests for improving the model")
            training_requests_results = dict(get_training_requests_status())
            df = pd.DataFrame(list(training_requests_results.items()), columns=['Animal', 'Count'])
            df.set_index('Animal', inplace=True)
            fig = px.bar(df)
            st.plotly_chart(fig)


def predict_image(image_file):
    bytes_data = image_file.getvalue()
    game_state.selected_image = image_file
    files = {
        "file": (image_file.name, io.BytesIO(bytes_data), image_file.type)
    }
    response = requests.post(url=GET_ANIMAL_PREDICTION_URL, files=files)
    return response.json()


def main():
    def stream_data(x):
        for word in x.split(" "):
            yield word + " "
            time.sleep(0.15)

    st.title(":green[**ANIMALS HANGMAN**] 🐘 🐶 🐓 🐈")

    if not game_state.is_game_active:
        text = (
            "This is a little game that based on uploaded images, randomly choose one of them and make a prediction "
            "based on it to generate the game's phrase.")
        # write(stream_data(x=text))
        # time.sleep(0.3)
        st.write(text)
        st.subheader("Upload the image you want to generate the game")
        uploaded_files = st.file_uploader(" ", type=["JPG", "PNG"], accept_multiple_files=True)

        if uploaded_files:
            if not game_state.prediction:
                with st.spinner("Processing images..."):
                    selected_image = random.sample(uploaded_files, k=1)[0]
                    predict_result = predict_image(selected_image)
                    game_state.prediction = predict_result["prediction"]
                    game_state.pred_probabilities = predict_result["probabilities"]

            if st.button("Create game"):
                with st.spinner("**Creating game...**"):
                    game_state.is_game_active = True
                    time.sleep(0.8)
                    st.rerun()
    else:
        show_game()


if __name__ == "__main__":
    main()
