function sendMessage() {
    var userMessage = document.getElementById("userInput").value;
    if (userMessage.trim() === "") return;  // Ignore empty input

    var messages = document.getElementById("messages");

    // Display user message
    var userMessageDiv = document.createElement("div");
    userMessageDiv.innerHTML = "<strong>You:</strong> " + userMessage;
    messages.appendChild(userMessageDiv);

    // Clear input field
    document.getElementById("userInput").value = "";

    // Simulate chatbot response
    setTimeout(function() {
        var botMessageDiv = document.createElement("div");
        botMessageDiv.innerHTML = "<strong>Chatbot:</strong> " + getBotResponse(userMessage);
        messages.appendChild(botMessageDiv);
    }, 1000);
}

// Basic chatbot responses
function getBotResponse(message) {
    var responses = {
        "hello": "Hi! How can I help you today?",
        "goodbye": "Goodbye! Have a great day!",
        "what is your name": "I'm your friendly chatbot, here to assist you."
    };
    return responses[message.toLowerCase()] || "Sorry, I didn’t understand that. Could you rephrase?";
}

function quickReply(message) {
    document.getElementById("userInput").value = message;
    sendMessage();
}