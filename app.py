import streamlit as st
import pandas as pd
from apyori import apriori

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

#process apriori results
def inspect(results):
    product1 = [tuple(result[2][0][0])[0] for result in results]
    product2 = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    return list(zip(product1, product2, supports, confidences, lifts))

#preprocess data into a list of transactions
def preprocess_data(df):
    transactions = []
    for i in range(len(df)):
        transactions.append([str(df.values[i, j]) for j in range(df.shape[1])])
    return transactions

#generate rules 
def generate_rules(transactions):
    rules = apriori(
        transactions=transactions,
        min_support=0.005,
        min_confidence=0.1,
        min_lift=3,
        min_length=2,
        max_length=2
    )
    return list(rules)

#recommendations based on user input
def get_recommendations(user_item, rules_df, confidence_threshold=0.2, lift_threshold=1.0):
    related_items = []
    for _, row in rules_df.iterrows():
        if user_item == row['product1'] and row['Confidence'] >= confidence_threshold and row['Lift'] >= lift_threshold:
            related_items.append((row['product2'], row['Confidence'], row['Lift']))
        elif user_item == row['product2'] and row['Confidence'] >= confidence_threshold and row['Lift'] >= lift_threshold:
            related_items.append((row['product1'], row['Confidence'], row['Lift']))
    
    if related_items:
        related_items.sort(key=lambda x: (x[1], x[2]), reverse=True)
        top_recommendations = related_items[:3]
        return top_recommendations
    else:
        return []

#recommendations displaying
def display_recommendations(user_item):
    recommendations = get_recommendations(user_item, st.session_state['DataFrame_intelligence'])

    if recommendations:
        st.write("\nYou may also need:")
        for item, confidence, lift in recommendations:
            st.write(f"{item} (Confidence: {confidence:.2f}, Lift: {lift:.2f}) - because customers who bought {user_item} also bought {item}.")
    else:
        st.write("No strong recommendations found for the item you entered.")
        new_items = st.text_input(f"What would you like to buy along with {user_item}? (Enter items separated by commas): ", key="new_items")
        if new_items:
            new_items = [item.strip() for item in new_items.split(',')]
            add_new_association(user_item, new_items)
            st.experimental_rerun()

#add a new association
def add_new_association(item1, item2):
    for item in item2:
        new_row = pd.DataFrame({
            'product1': [item1],
            'product2': [item],
            'Support': [0.001],  
            'Confidence': [0.5],  
            'Lift': [2.0]  
        })
        st.session_state['DataFrame_intelligence'] = pd.concat([st.session_state['DataFrame_intelligence'], new_row], ignore_index=True)
    st.write(f"New association added: {item1} -> {item2}")

df = load_data()
transactions = preprocess_data(df)
rules = generate_rules(transactions)

if 'DataFrame_intelligence' not in st.session_state:
    st.session_state['DataFrame_intelligence'] = pd.DataFrame(inspect(rules), columns=['product1', 'product2', 'Support', 'Confidence', 'Lift'])

# Data visualization
def visualize_data(df):
    # Flatten the DataFrame into a single list of items
    items = [item for row in df.values for item in row]

    # Drop NaN, coun frequency of each item
    item_counts = pd.Series(items).dropna().value_counts()
    st.subheader("Top 20 Items by Frequency")
    plt.figure(figsize=(10, 6))
    item_counts[:20].plot(kind='bar')
    plt.xlabel('Items')
    plt.ylabel('Frequency')
    plt.title('Top 20 Items by Frequency')
    st.pyplot(plt)

    #top associations by Lift
    st.subheader("Top Associations by Lift")
    top_associations = st.session_state['DataFrame_intelligence'].nlargest(n=20, columns='Lift')
    st.write(top_associations)
    plt.figure(figsize=(10, 6))
    plt.barh(top_associations['product1'] + ' & ' + top_associations['product2'], top_associations['Lift'])
    plt.xlabel('Lift')
    plt.ylabel('Product Association')
    plt.title('Top Associations by Lift')
    st.pyplot(plt)

st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Dataset", "Visualization", "Recommendation"])

if option == "Dataset":
    st.title("Dataset")
    st.dataframe(df, width=1500)
elif option == "Visualization":
    st.title("Data Visualization")
    visualize_data(df)
elif option == "Recommendation":
    st.title("Product Recommendation System")
    user_item = st.text_input("Enter an item (or type 'exit' to quit): ", key="user_item").strip()

    if user_item.lower() == 'exit':
        st.stop()
    elif user_item:
        display_recommendations(user_item)
