from app.chatbot import chatbot

while True:
    request = input("Ты: ")
    if request.lower() == "выход":
        break
    answer = chatbot(request)
    print("BOT:", answer)