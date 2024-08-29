import google.generativeai as genai
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure the Gemini API (replace 'YOUR_API_KEY_HERE' with your actual API key)
api=' '
if (api==' '):
    api = input("Please enter your gemini api key: ")
genai.configure(api_key=api)

def get_embedding(text):
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(response['embedding'])
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def get_related_ideas(idea):
    prompt = f" USE VECTOR EMBEDDINGS AND Generate 5 distinct short (at max 3-4 word prefferably 1 word, push limits if necessary like for longer idea give longer output) ideas related to: {idea}. Provide each idea on a new line and provide no description just ideas. Provide closest ideas possible . NOTE: If the idea is a existing famous company, startup, organisation, site etc then give related ideas related to that thing's idea not word like for facebook output closest words like Instagram, whatsapp, twitter, Got it Yep do it."
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        related_ideas = [idea.strip() for idea in response.text.splitlines() if idea.strip()]
        return related_ideas[:5]  # Ensure we only return 5 ideas
    
    except Exception as e:
        print(f"Error generating related ideas: {e}")
        return []

def visualize_ideas(original_idea, related_ideas):
    ideas = [original_idea] + related_ideas
    embeddings = [get_embedding(idea) for idea in ideas]
    
    if any(embedding is None for embedding in embeddings) or len(embeddings) < 2:
        print("Error: Failed to get embeddings for all ideas.")
        return
    
    embeddings_array = np.array(embeddings)
    
    # Reduce to 3 dimensions using t-SNE
    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(ideas) - 1))
    embeddings_3d = tsne.fit_transform(embeddings_array)
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    for i, (idea, point) in enumerate(zip(ideas, embeddings_3d)):
        ax.scatter(*point, c=colors[i], s=300, label=idea)
        ax.text(*point, idea, fontsize=12, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Dimension 1', fontsize=14, labelpad=10)
    ax.set_ylabel('Dimension 2', fontsize=14, labelpad=10)
    ax.set_zlabel('Dimension 3', fontsize=14, labelpad=10)
    ax.set_title('3D Representation of Ideas', fontsize=20, fontweight='bold', pad=20)
    
    ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Improve aesthetics
    ax.set_facecolor('#F0F0F0')
    fig.patch.set_facecolor('#F0F0F0')
    
    plt.tight_layout()
    plt.show()

def main():
    original_idea = input("Enter your idea: ")
    related_ideas = get_related_ideas(original_idea)
    
    print("\nRelated Ideas:")
    for i, idea in enumerate(related_ideas, 1):
        print(f"{i}. {idea}")
    
    print("\nGenerating visualization...")
    visualize_ideas(original_idea, related_ideas)

if __name__ == "__main__":
    main()
