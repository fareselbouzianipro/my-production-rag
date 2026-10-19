from production_rag.rag import answer_question


def main() -> None:
    question = input("Ask about football regulations: ")

    result = answer_question(question)

    print(result)


if __name__ == "__main__":
    main()
