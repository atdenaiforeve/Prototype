from learner import LanguageLearner


def main() -> None:
    learner = LanguageLearner()

    print("Prototype language learner")
    print("Type training text one line at a time.")
    print("Press Enter on an empty line to finish. Type 'quit' to stop.")

    while True:
        text = input("> ").strip()

        if text.lower() == "quit":
            break

        if not text:
            break

        learner.train(text)
        print("Learned.")

    learner.save()

    print("\nSaved.")
    print("Stats:", learner.stats())

    while True:
        word = input("\nEnter a word to predict from (or 'quit'): ").strip()

        if word.lower() == "quit":
            break

        predictions = learner.predict_next(word)

        if not predictions:
            print("I have not learned what usually comes after that yet.")
            continue

        print("Predictions:")
        for item in predictions:
            print(
                f"  {item['word']} "
                f"(confidence: {item['confidence']:.2f}, "
                f"seen: {item['seen']})"
            )


if __name__ == "__main__":
    main()
