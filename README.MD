# Animals Hangman 🐘

This is a little game that based on uploaded images,
randomly choose one of them and make a prediction
based on it to generate the game's phrase.

The steps are as follows:

1. Upload the images you would like to consider in the game.
   [IMAGE]
2. Each uploaded images will be interpreted as bytes and sent
   as request to the endpoint /predict at the (animals prediction api)[link to swagger docs].
3. The (animals prediction api)[link to swagger docs] uses a Convolution Neural Network model
   that was trained over 60,000 of animal images with classes: `dog`, `horse`, `elephant`, `butterfly`, `chicken`,
   `cat`, `cow`, `sheep`, `spider`, and `squirrel`. **It means the model has a temporary limitation to classify
   only images of these kind of animals.**
4. The response from the api is composed by the final prediction to the image and its probabilities
   of being any of the 10 animals (being the prediction the higher probability).
5. After getting all the predictions for the images, the system will randomly select
   one of them to build the hangman game.
6. With the hangman generated, the user can make the guesses and receive the
   feedbacks whether he made a right guess or a wrong one.
7. The maximum of try guesses is 6 in order to construct the complete body of the hangman.
8. Once the user found out the animal name, the system is going to feedback with commemorative balloons. 🎈
   Otherwise, if the user exceed the maximum of try guesses a rain of ❌ emojis is showed. In both cases, a
   button for creating a new game becomes available.
