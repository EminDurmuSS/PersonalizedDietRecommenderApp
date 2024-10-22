import uuid
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
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
GROQ_API_KEY=st.secrets["GROQ_API_KEY"]
openai.api_key = OPENAI_API_KEY


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

def query_recipes(vector, offset=0):
    response = index.query(
        namespace="foodrecipe5",
        vector=vector,
        top_k=10,  # Retrieve more recipes to handle user requests for more options
        include_metadata=True
    )
    return response['matches'][offset:offset+1]  # Return only one recipe at a time


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

def format_ingredients(ingredients):
    # Convert the ingredients string to a list
    ingredients_list = ast.literal_eval(ingredients)
    formatted_ingredients = "\n".join([f"- {ingredient}" for ingredient in ingredients_list])
    return formatted_ingredients

def format_instructions(instructions):
    # Use regular expressions to split instructions based on the pattern `n-)`
    instructions_list = re.split(r'(\d+-\))', instructions)
    formatted_instructions = ""
    step = ""
    for item in instructions_list:
        if re.match(r'\d+-\)', item):
            if step:
                formatted_instructions += step.strip() + "\n"
            step = item
        else:
            step += " " + item.strip()
    if step:
        formatted_instructions += step.strip()
    return formatted_instructions

def format_health_type(health_type):
    # Convert the health type string to a list and then format it
    health_type_list = ast.literal_eval(health_type)
    formatted_health_type = "\n".join([f"* {ht}" for ht in health_type_list])
    return formatted_health_type

def format_nutrition_facts(nutrition_facts):
    # Convert the nutrition facts string to a list and then format it
    nutrition_facts_list = nutrition_facts.split(" - ")
    formatted_nutrition_facts = "\n".join([f"* {nf}" for nf in nutrition_facts_list])
    return formatted_nutrition_facts

def display_recipe_info(food_info):
    st.write("### Recipe Information:")
    for info in food_info:
        st.write(f"##### **Food Name:** {info['Food Name']} ")
        st.write(f"**Description:** {info.get('Description', 'N/A')}")
        st.write(f"**Meal Type:** {info.get('Meal Type', 'N/A')}")
        st.write(f"**Diet Type:** {info.get('Diet Type', 'N/A')}")
        
        health_type = info.get('Health Type', 'N/A')
        formatted_health_type = format_health_type(health_type)
        st.write("**Health Type:**")
        st.text(formatted_health_type)
        
        st.write(f"**Country/Region:** {info.get('Country/Region', 'N/A')}")
        st.write(f"**Occasion:** {info.get('Occasion', 'N/A')}")
        st.write(f"**Cook Time:** {info.get('Cook Time', 'N/A')}")

        st.write("**Ingredients:**")
        ingredients = info.get('Recipe Ingredients', 'N/A')
        formatted_ingredients = format_ingredients(ingredients)
        st.text(formatted_ingredients)

        st.write("**Instructions:**")
        instructions = info.get('Recipe Instructions', 'N/A')
        formatted_instructions = format_instructions(instructions)
        st.text(formatted_instructions)

        st.write(f"**Servings:** {info.get('Recipe Servings', 'N/A')}")
        st.markdown(f"<p><b>Keywords:</b> {info.get('Keywords', 'N/A')}</p>", unsafe_allow_html=True)
        
        nutrition_facts = info.get('Nutrition Facts', 'N/A')
        formatted_nutrition_facts = format_nutrition_facts(nutrition_facts)
        st.write("**Nutrition Facts:**")
        st.text(formatted_nutrition_facts)

        if 'Calories' in info and 'Fat' in info and 'Carbohydrates' in info and 'Protein' in info:
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
        st.session_state.selected_region = ""  # Default region empty
        st.session_state.selected_country = ""  # Default country empty
        st.session_state.recipe_offset = 0  # Initialize recipe offset for pagination

    region_cuisine_map = {
        "Global": ["General"],
        "Europe": ["General", "Italian", "French", "Dutch", "German", "Spanish", "Greek", "Portuguese", "Swedish"],
        "Asia": ["General", "Chinese", "Japanese", "Indian", "Korean", "Thai", "Vietnamese", "Filipino", "Indonesian"],
        "Americas": ["General", "American", "Mexican", "Brazilian", "Argentinian", "Canadian", "Peruvian", "Chilean"],
        "Africa": ["General", "North African", "West African", "East African", "South African", "Ethiopian", "Moroccan", "Egyptian"],
        "Middle East": ["General", "Turkish", "Lebanese", "Persian", "Israeli", "Syrian", "Saudi Arabian", "Emirati"],
        "Oceania": ["General", "Australian", "New Zealand"]
    }

    regions = [""] + list(region_cuisine_map.keys())
    region = st.selectbox("Select Region", regions)

    if region:
        countries = [""] + region_cuisine_map[region]
    else:
        countries = [""]

    if st.session_state.selected_region != region:
        st.session_state.selected_region = region
        st.session_state.selected_country = ""

    with st.form("recipe_form"):
        meal_type = st.selectbox("Meal Type", ["","Breakfast", "Brunch", "Lunch","Starter", "Dinner", "Snack", "Dessert","Drink"])
        calories = st.selectbox("Calories Level", ["Low", "Medium", "High"])
        carbs = st.selectbox("Carbs Level", ["Low", "Medium", "High"])
        protein = st.selectbox("Protein Level", ["Low", "Medium", "High"])
        fat = st.selectbox("Fat Level", ["Low", "Medium", "High"])
        diet_type = st.selectbox("Diet Type", ["Standard", "Vegan", "Ketogenic", "Paleo", "Vegetarian", "Gluten-Free"])

        country_cuisine = st.selectbox("Select Country/Cuisine", countries)

        if st.session_state.selected_country != country_cuisine:
            st.session_state.selected_country = country_cuisine

        occasion = st.text_input("Occasion")
        allergies = st.text_input("Food Allergies")
        health_problems = st.text_input("Health Problems")
        
        # Adding more cook time options based on dataset
        cook_time = st.selectbox("Cook Time", ["","less than 15 Mins", "less than 30 Mins", "less than 60 Mins", 
                                               "less than 4 Hours", "less than 6 Hours", "less than 7 Hours", 
                                               "less than 8 Hours", "less than 10 Hours", "less than 12 Hours", 
                                               "more than 12 Hours"])

        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        query_text = f"Meal Type: {meal_type}, Calories: {calories}, Carbs: {carbs}, Protein: {protein}, Fat: {fat}, Diet Type: {diet_type}, Country/Region: {country_cuisine},Occasion:{occasion}, Allergies: {allergies}, Health Problems: {health_problems}, Cook Time: {cook_time}"
        query_embedding = get_embedding(query_text, model="text-embedding-ada-002")
        recipes = query_recipes(query_embedding)

        foodInfo = ""
        food_info = []
        for match in recipes:
            for key, value in match['metadata'].items():
                foodInfo += f"**{key}**: {value}\n"
            metadata = match['metadata']
            food_info.append(metadata)

        st.session_state.food_info = food_info
        user_question = f"""
            Based on that food: {foodInfo}
            You are a professional chef with expertise in creating customized meal plans and you are also a dietitian. Return the matching recipes from the provided document, according to the user's criteria. Do not create new recipes. The user will enter the desired nutritional values for one serving and you will return the results based on the nutritional values for one serving only. Here are the criteria:

            ### Criteria:
            - **Calories**: {calories}
            - **Carbs**: {carbs}
            - **Protein**: {protein}
            - **Fat**: {fat}
            - **Diet Type**: {diet_type}
            - **Country/Region**: {country_cuisine}
            - **Food Allergies**: {allergies}
            - **Health Problems**: {health_problems}
            - **Meal Type**: {meal_type}
            - **Cook Time**: {cook_time}
            - **occasion**::{occasion}    
            
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

    if st.button("Get Another Recipe"):
        st.session_state.recipe_offset += 1  # Increment the offset to get the next recipe
        query_text = f"Meal Type: {meal_type}, Calories: {calories}, Carbs: {carbs}, Protein: {protein}, Fat: {fat}, Diet Type: {diet_type}, Country/Region: {country_cuisine},Occasion:{occasion}, Allergies: {allergies}, Health Problems: {health_problems}, Cook Time: {cook_time}"
        query_embedding = get_embedding(query_text, model="text-embedding-ada-002")
        recipes = query_recipes(query_embedding, offset=st.session_state.recipe_offset)
        
        food_info = []
        for match in recipes:
            metadata = match['metadata']
            food_info.append(metadata)

        st.session_state.food_info = food_info
        user_question = f"""
            Based on that food: {food_info}
            You are a professional chef with expertise in creating customized meal plans and you are also a dietitian. Return the matching recipes from the provided document, according to the user's criteria. Do not create new recipes. The user will enter the desired nutritional values for one serving and you will return the results based on the nutritional values for one serving only. Here are the criteria:

            ### Criteria:
            - **Calories**: {calories}
            - **Carbs**: {carbs}
            - **Protein**: {protein}
            - **Fat**: {fat}
            - **Diet Type**: {diet_type}
            - **Country/Region**: {country_cuisine}
            - **Food Allergies**: {allergies}
            - **Health Problems**: {health_problems}
            - **Meal Type**: {meal_type}
            - **Cook Time**: {cook_time}
            - **occasion**::{occasion}    
            
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

        st.write("### Dietitian's Answer: ")
        st.write(st.session_state.llm_response)

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