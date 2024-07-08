import os
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.llms import OpenAI
import openai
import re
from docx import Document
import ast
from streamlit_image_select import image_select

# Load environment variables
load_dotenv()

# Access secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
INDEX_NAME = st.secrets["INDEX_NAME"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment="us-west1-gcp")
index = pc.Index(INDEX_NAME)

# Set up the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    embedding = response['data'][0]['embedding']
    return embedding

def get_docx_text(docx_path):
    text = ""
    document = Document(docx_path)
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

def split_text_by_sections(text):
    chunks = []
    current_chunk = []
    in_section = False

    for line in text.split("\n"):
        if "====" in line:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            in_section = True
        elif in_section and line.strip() == "":
            continue  # Skip empty lines within a section
        else:
            current_chunk.append(line)
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def get_nutrition_facts(text):
    in_nutrition_section = False
    nutrition_lines = []

    for line in text.split("\n"):
        if "Nutrition Facts" in line:
            in_nutrition_section = True
        if in_nutrition_section:
            if line.startswith("# D"):  # End of nutrition facts section
                break

            nutrition_lines.append(line)
            
    return "\n".join(nutrition_lines)


def extract_metadata(section):
    metadata = {}
    lines = section.split('\n')
    in_description_section = False
    in_ingredient_quantities_section = False
    in_ingredient_parts_section = False
    in_nutrition_section = False
    in_instruction_section = False
    instruction_lines = []
    description_lines = []
    ingredient_quantities_lines = []
    ingredient_parts_lines = []
    in_image_section = False
    image_urls_raw = []

    for line in lines:
        if "Food Name:" in line:
            metadata['Food Name'] = line.split("Food Name:")[1].strip()
        elif "Description:" in line:
            in_description_section = True
            description_lines = [line.split("Description:")[1].strip()]
        elif "Recipe Category:" in line:
            metadata['Recipe Category'] = line.split("Recipe Category:")[1].strip()
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Keywords:" in line:
            metadata['Keywords'] = line.split("Keywords:")[1].strip()
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Recipe Ingredient Quantities:" in line:
            in_ingredient_quantities_section = True
            ingredient_quantities_lines = [line.split("Recipe Ingredient Quantities:")[1].strip()]
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Recipe Ingredient Parts:" in line:
            in_ingredient_parts_section = True
            ingredient_parts_lines = [line.split("Recipe Ingredient Parts:")[1].strip()]
            in_ingredient_quantities_section = False
            if ingredient_quantities_lines:
                metadata['Recipe Ingredient Quantities'] = " ".join(ingredient_quantities_lines)
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Recipe Servings:" in line:
            metadata['Recipe Servings'] = line.split("Recipe Servings:")[1].strip()
            in_ingredient_parts_section = False
            if ingredient_parts_lines:
                metadata['Recipe Ingredient Parts'] = " ".join(ingredient_parts_lines)
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Recipe Instructions:" in line:
            in_instruction_section = True
            instruction_lines = [line.split("Recipe Instructions:")[1].strip()]
            in_ingredient_parts_section = False
            if ingredient_parts_lines:
                metadata['Recipe Ingredient Parts'] = " ".join(ingredient_parts_lines)
            in_ingredient_quantities_section = False
            if ingredient_quantities_lines:
                metadata['Recipe Ingredient Quantities'] = " ".join(ingredient_quantities_lines)
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Nutrition Facts" in line:
            in_nutrition_section = True
            in_instruction_section = False
            if instruction_lines:
                metadata['Recipe Instructions'] = " ".join(instruction_lines)
            in_ingredient_parts_section = False
            if ingredient_parts_lines:
                metadata['Recipe Ingredient Parts'] = " ".join(ingredient_parts_lines)
            in_ingredient_quantities_section = False
            if ingredient_quantities_lines:
                metadata['Recipe Ingredient Quantities'] = " ".join(ingredient_quantities_lines)
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif "Food images:" in line:
            in_image_section = True
            in_instruction_section = False
            in_nutrition_section = False
            if instruction_lines:
                metadata['Recipe Instructions'] = " ".join(instruction_lines)
            in_ingredient_parts_section = False
            if ingredient_parts_lines:
                metadata['Recipe Ingredient Parts'] = " ".join(ingredient_parts_lines)
            in_ingredient_quantities_section = False
            if ingredient_quantities_lines:
                metadata['Recipe Ingredient Quantities'] = " ".join(ingredient_quantities_lines)
            in_description_section = False
            if description_lines:
                metadata['Description'] = " ".join(description_lines)
        elif in_instruction_section:
            if line.strip() and not any(key in line for key in ["Food Name:", "Description:", "Recipe Category:", "Keywords:", "Recipe Ingredient Quantities:", "Recipe Ingredient Parts:", "Recipe Servings:", "Nutrition Facts", "Food images:"]):
                instruction_lines.append(line.strip())
        elif in_description_section:
            if line.strip() and not any(key in line for key in ["Food Name:", "Recipe Category:", "Keywords:", "Recipe Ingredient Quantities:", "Recipe Ingredient Parts:", "Recipe Servings:", "Recipe Instructions:", "Nutrition Facts", "Food images:"]):
                description_lines.append(line.strip())
        elif in_ingredient_quantities_section:
            if line.strip() and not any(key in line for key in ["Food Name:", "Description:", "Recipe Category:", "Keywords:", "Recipe Ingredient Parts:", "Recipe Servings:", "Recipe Instructions:", "Nutrition Facts", "Food images:"]):
                ingredient_quantities_lines.append(line.strip())
        elif in_ingredient_parts_section:
            if line.strip() and not any(key in line for key in ["Food Name:", "Description:", "Recipe Category:", "Keywords:", "Recipe Ingredient Quantities:", "Recipe Servings:", "Recipe Instructions:", "Nutrition Facts", "Food images:"]):
                ingredient_parts_lines.append(line.strip())
        elif in_nutrition_section:
            if "Calories:" in line:
                metadata['Calories'] = int(float(line.split("Calories:")[1].strip().split()[0]))
            elif "Fat Content:" in line:
                metadata['Fat'] = int(float(line.split("Fat Content:")[1].strip().split()[0]))
            elif "Carbohydrate Content:" in line:
                metadata['Carbohydrates'] = int(float(line.split("Carbohydrate Content:")[1].strip().split()[0]))
            elif "Protein Content:" in line:
                metadata['Protein'] = int(float(line.split("Protein Content:")[1].strip().split()[0]))
            elif line.strip() == "]":  # End of nutrition facts section
                in_nutrition_section = False
        elif in_image_section:
            image_urls_raw.append(line.strip())
            if line.strip().endswith(']'):
                image_urls_str = ''.join(image_urls_raw)
                image_urls = re.findall(r'https?://\S+\.(?:jpg|jpeg|png)', image_urls_str)
                if image_urls:
                    metadata['Food Images'] = str([[url] for url in image_urls])
                in_image_section = False

    if in_instruction_section and instruction_lines:
        metadata['Recipe Instructions'] = " ".join(instruction_lines)
    
    if in_description_section and description_lines:
        metadata['Description'] = " ".join(description_lines)

    if in_ingredient_quantities_section and ingredient_quantities_lines:
        metadata['Recipe Ingredient Quantities'] = " ".join(ingredient_quantities_lines)
    
    if in_ingredient_parts_section and ingredient_parts_lines:
        metadata['Recipe Ingredient Parts'] = " ".join(ingredient_parts_lines)

    return metadata


def process_and_store_embeddings(docx_path, namespace):
    raw_text = get_docx_text(docx_path)
    sections = split_text_by_sections(raw_text)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace=namespace,
        index_name=INDEX_NAME
    )

    for section in sections:
        metadata = extract_metadata(section)
        nutrition_facts = get_nutrition_facts(section)

        # Check for Food Name first
        if 'Food Name' in metadata:
            id = metadata['Food Name']
        else:
            id = str(uuid.uuid4())  # Generate a unique ID if Food Name is not available

        if nutrition_facts:
            nutrition_embedding = get_embedding(nutrition_facts, model="text-embedding-ada-002")
            metadata['Nutrition Facts'] = nutrition_facts
            vectorstore.add_texts(texts=[nutrition_facts], embeddings=[nutrition_embedding], metadatas=[metadata], ids=[id])
        else:
            text_embedding = get_embedding(section, model="text-embedding-ada-002")
            metadata['text'] = section
            vectorstore.add_texts(texts=[section], metadatas=[metadata], embeddings=[text_embedding], ids=[id])


# Preprocess and store embeddings if not already done
namespace = "foodrecipe5"
if index.describe_index_stats()["namespaces"].get(namespace, {}).get("vector_count", 0) == 0:
    pdf_path = "Recipess.docx"
    process_and_store_embeddings(pdf_path, namespace)

def load_embeddings(namespace):
    return PineconeVectorStore(
        index=index,
        embedding=embedding_model,
        namespace=namespace,
        index_name=INDEX_NAME
    )

# Preprocess and store embeddings if not already done
namespace = "foodrecipe5"
if index.describe_index_stats()["namespaces"].get(namespace, {}).get("vector_count", 0) == 0:
    pdf_path = "Recipess.docx"
    process_and_store_embeddings(pdf_path, namespace)
st.title("Personalized Diet Recommender ☕")
st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
st.subheader('Your Best Food Advisor 🥄')
st.markdown('<style>h3{color: pink; text-align: center;}</style>', unsafe_allow_html=True)
st.markdown('<style>div[class="stButton"] button { background-color: #4CAF50; }</style>', unsafe_allow_html=True)

def query_recipes(vector):
    response = index.query(
        namespace="foodrecipe5",
        vector=vector,
        top_k=2,
        include_metadata=True
    )
    return response

def get_conversation_chain(vectorstore):
    llm = ChatGroq(temperature=0.8, groq_api_key=st.secrets["GROQ_API_KEY"], model_name="llama3-70b-8192")
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
    )
    return conversation_chain

def handle_userinput(user_question, conversation):
    response = conversation({'question': user_question, 'chat_history': []})
    return response['answer']

def display_recipe_info(food_info):
    st.write("### Recipe Information:")
    for info in food_info:
        st.write(f"##### **Food Name:** {info['Food Name']} ")
        st.write(f"**Description:** {info['Description']}")
        st.write(f"**Ingredients:** {info['Recipe Ingredient Quantities']}")
        st.write(f"**Parts:** {info['Recipe Ingredient Parts']}")
        st.write(f"**Servings:** {info['Recipe Servings']}")
        st.markdown(f"<p><b>Keywords:</b> {info['Keywords']}</p>", unsafe_allow_html=True)

        st.markdown(f"<p><b>Instructions:</b> {info['Recipe Instructions']}</p>", unsafe_allow_html=True)

        st.markdown(f'''{info['Nutrition Facts']}''', unsafe_allow_html=True)


        st.markdown("#### **Nutritional Information**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", info['Calories'])
        col2.metric("Fat", info['Fat'])
        col3.metric("Carbohydrates", info['Carbohydrates'])
        col4.metric("Protein", info['Protein'])

        if 'Food Images' in info:
            try:
                image_urls_str = info['Food Images']
                image_urls = ast.literal_eval(image_urls_str)
                flat_image_urls = [url for sublist in image_urls for url in sublist] if isinstance(image_urls, list) else [image_urls]

                if flat_image_urls:
                    selected_image = image_select(
                        label=f"Select an image for {info['Food Name']}",
                        images=flat_image_urls,
                        captions=[info['Food Name'] for _ in flat_image_urls],
                        use_container_width=True
                    )
                    if selected_image:
                        st.image(selected_image, caption=info['Food Name'], use_column_width=True, width=800)
                else:
                    st.write("No images available.")
            except Exception as e:
                st.write(f"Error loading images: {e}")

def main():
    if 'conversation' not in st.session_state:
        vectorstore = load_embeddings("foodrecipe5")
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.chat_history = []
        st.session_state.food_info = []
        st.session_state.llm_response = ""  # Initialize LLM response

    with st.form("recipe_form"):
        meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner"])
        calories = st.number_input("Calories (kcal)", min_value=0)
        carbs = st.number_input("Carbs (g)", min_value=0)
        protein = st.number_input("Protein (g)", min_value=0)
        fat = st.number_input("Fat (g)", min_value=0)
        diet_type = st.selectbox("Diet Type", ["Standard", "Vegan", "Ketogenic", "Paleo", "Vegetarian"])
        allergies = st.text_input("Food Allergies")
        health_problems = st.text_input("Health Problems")
        
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        query_text = f"Meal Type: {meal_type}, Calories: {calories} kcal, Carbs: {carbs} g, Protein: {protein} g, Fat: {fat} g, Diet Type: {diet_type}, Allergies: {allergies}, Health Problems: {health_problems}"
        query_embedding = get_embedding(query_text, model="text-embedding-ada-002")
        recipes = query_recipes(query_embedding)
        
        foodInfo=""
        food_info = []
        for match in recipes['matches']:
            for key, value in match['metadata'].items():
                foodInfo += f"**{key}**: {value}\n"
            metadata = match['metadata']
            food_info.append(metadata)
        
        st.session_state.food_info = food_info
        user_question = f"""
            Based on that food: {foodInfo}
            You are a professional chef with expertise in creating customized meal plans and you are also a dietitian. Return the matching recipes from the provided document, according to the user's criteria. Do not create new recipes. The user will enter the desired nutritional values for one serving and you will return the results based on the nutritional values for one serving only. Here are the criteria:
            
            ### Criteria:
            - **Calories**: {calories} kcal
            - **Carbs**: {carbs}g
            - **Protein**: {protein}g
            - **Fat**: {fat}g
            - **Diet Type**: {diet_type}
            - **Food Allergies**: {allergies}
            - **Health Problems**: {health_problems}
            - **Meal Type**: {meal_type}

            ### Instructions:
            1. Provide the name, detailed description, ingredients, instructions, directions, and nutrition facts for each meal.
            2. Ensure recipes account for any health problems or allergies mentioned.
            3. Format the response in a user-friendly way, with clear sections and bullet points for easy reading.
            4. Explain in great detail how the dish is made, including all direction steps.
            5. Provide the name, detailed description, instructions, ingredients, directions, and nutrition facts for that meal.
        """
        conversation = st.session_state.conversation
        st.session_state.llm_response = handle_userinput(user_question, conversation)
        st.session_state.chat_history.append(st.session_state.llm_response)
    
    if st.session_state.llm_response:
        st.write("### Dietitian's Answer: ")
        st.write(st.session_state.llm_response)

    if st.session_state.food_info:
        display_recipe_info(st.session_state.food_info)

    if st.button("Modify Recipe"):
        if not st.session_state.chat_history:
            st.write("Please get a recipe recommendation first.")
        else:
            original_recipe = st.session_state.chat_history[-1]
            modification_request = f"{st.session_state.food_info} choose the recipe that best fits your user criteria ,  Modify this recipe to be {diet_type} friendly and avoid these allergens: {allergies}. Recipe: {original_recipe} and Give all the information about the food. Provide the name, detailed description, ingredients, directions, and nutrition facts for each meal."
            modified_recipe = handle_userinput(modification_request, st.session_state.conversation)
            st.session_state.chat_history.append(modified_recipe)
            st.write("### Modified Recipe:")
            st.write(modified_recipe)

    with st.sidebar:
        st.header("Recipe Chatbot")
        user_input = st.text_input("Ask me anything about the suggested dish")
        if st.button("Send"):
            if not st.session_state.chat_history:
                st.write("Please get a recipe recommendation first.")
            else:
                conversation = st.session_state.conversation
                question_request = f"Answer all questions according to these dishes: {st.session_state.food_info}. The question is: {user_input}"
                response = handle_userinput(question_request, conversation)
                st.session_state.chat_history.append(response)
                st.write(response)

if __name__ == "__main__":
    main()