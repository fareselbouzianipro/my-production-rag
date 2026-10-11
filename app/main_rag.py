from app.rag import answer_question


def main() -> None:
    question = input("Ask Acme documentation: ")

    result = answer_question(question)

    print(result)


if __name__ == "__main__":
    main()
