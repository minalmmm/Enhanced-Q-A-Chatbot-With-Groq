# Enhanced Q&A Chatbot with Groq

## 1. Project Setup and Configuration

### **API Keys**
   - Obtain the necessary API keys:
     - `LANGCHAIN_API_KEY`: Required for authentication with LangChain.
     - `GROQ_API_KEY`: Required for authentication with Groq’s models.
   - Store these keys securely in the `.env` file.

### **Environment Configuration**
   - Create a `.env` file in the root directory of the project and add the following:
     ```ini
     LANGCHAIN_API_KEY=your_langchain_api_key
     GROQ_API_KEY=your_groq_api_key
     ```
   - This will securely store and provide the API keys to the application.

### **Install Dependencies**
   - Ensure the required Python libraries are installed. Run the following command to install the necessary dependencies:
     ```bash
     pip install streamlit langchain langchain_groq python-dotenv
     ```

## 2. User Interface with Streamlit

### **Streamlit App**
   - The application is built with Streamlit, providing a user-friendly interface for the Q&A chatbot. 
   - It runs on the web, enabling real-time interaction with the chatbot.

### **Sidebar Settings**
   - Users are prompted to enter their **Groq API Key** for authentication.
   - **Model Selection**: The user can choose a model from the available Groq models such as:
     - `llama-3.3-70b`
     - `llama-3.1-8b-instant`
     - `mixtral-8x7b-32768`
     - `gemma2-9b-it`
     - `whisper-large-v3`
   - **Response Parameters**: Users can adjust settings like:
     - **Temperature**: A slider (0.0 to 1.0) to control the creativity of the model's responses.
     - **Max Tokens**: A slider (50 to 300) to control the length of the generated responses.

### **Main Interface**
   - Users can enter a question into a text input box and receive answers generated by the selected Groq model.
   - The response will appear below the input box after the user clicks **Submit**.

## 3. Query Processing and Response Generation

### **Generate Response**
   - When a user submits a query, the app sends the question to the Groq model using the provided API key.
   - The response is generated based on the selected model and parameters like temperature and max tokens.
   - The LangChain API is used to process the model’s output and provide a well-structured response.

## 4. Error Handling

   - The app includes error handling to display a message if there is any issue with the API key, model, or response generation process.
   - For example, if a user selects a model they do not have access to or there is an issue with the API, an error message will be displayed to inform the user.

## 5. Application Output

   - **Chatbot Response**: Once a valid response is generated, it is displayed below the user input.
   - The response can vary based on the selected model and the custom settings adjusted by the user.

## 6. Model Selection and Customization

   - The app supports multiple Groq models, allowing users to choose based on the task requirements. Available models include:
     - **Llama-3.3-70B**: High-performance model for large and complex tasks.
     - **Llama-3.1-8B-Instant**: Optimized for quick responses with a smaller context window.
     - **Mixtral-8x7B-32768**: Suited for handling longer text sequences and complex tasks.
     - **Gemma2-9B-IT**: A versatile model for general-purpose tasks.
     - **Whisper-Large-V3**: Best for tasks related to automatic speech recognition and processing long sequences.

## 7. Running the Application

   - After setting up the environment and dependencies, users can run the application using Streamlit:
     ```bash
     streamlit run app.py
     ```
   - The app will open in the default web browser where users can interact with the chatbot.

## 8. Sample Interaction

   ![Output](https://github.com/minalmmm/Enhanced-Q-A-Chatbot-With-Groq/blob/main/img1.png)
   ![Output](https://github.com/minalmmm/Enhanced-Q-A-Chatbot-With-Groq/blob/main/img2.png)
   ![Output](https://github.com/minalmmm/Enhanced-Q-A-Chatbot-With-Groq/blob/main/img3.png)
   ![Output](https://github.com/minalmmm/Enhanced-Q-A-Chatbot-With-Groq/blob/main/img4.png)
   ![Output](https://github.com/minalmmm/Enhanced-Q-A-Chatbot-With-Groq/blob/main/img5.png)
   
