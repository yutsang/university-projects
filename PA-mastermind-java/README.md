# Mastermind Java Game

This is a Java implementation of the Mastermind game, designed for use with the Greenfoot educational environment. The game features a graphical interface, multiple character types, and a scoring system.

## Features
- Play the classic Mastermind game with animal and balloon characters
- Save and load game state
- High score tracking
- Modular code structure for easy extension

## Project Structure
- `MastermindWorld.java`: Main game logic and world setup
- `GWorld.java`: Utility and world management functions
- `Judge.java`, `NiceJudge.java`, `FriendJudge.java`, `NaughtyGuy.java`: Judge and character logic
- `Balloon.java`, `Peg.java`, `Block.java`, `Wombat.java`, `Dolphin.java`, `Elephant.java`, `Pig.java`, `OrangeBalloon.java`, `YellowBalloon.java`: Game pieces and actors
- `Message.java`, `SpecialCharacter.java`: Messaging and special actor logic
- `images/`: All image assets for the game
- `doc/`: Javadoc-generated documentation

## Requirements
- [Greenfoot](https://www.greenfoot.org/) (Java educational IDE)
- Java 8 or later

## How to Run
1. Open Greenfoot.
2. Import this project folder.
3. Open `MastermindWorld.java` and run the scenario.

## How to Play
- The goal is to guess the secret combination of animals.
- Use the interface to place your guesses and receive feedback from the judges.
- Try to achieve the highest score possible!

## Contributing
Feel free to fork and submit pull requests for improvements or bug fixes.

## License
This project is for educational use. See individual file headers for authorship. 