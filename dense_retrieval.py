import cohere
import numpy as np
import pandas as pd
from tqdm import tqdm
#salon_veggie.9t

api_key = "8Uu6PwpQBDZizoZAPjH8kCbDNTlzGc8GUlJB3XpP"

co = cohere.ClientV2(api_key)

text = """
Interstellar is a 2014 epic[b] science fiction drama film[1][9] directed by Christopher Nolan, 
who co-wrote the screenplay with his brother, Jonathan Nolan. 
It features an ensemble cast led by Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, 
Ellen Burstyn, and Michael Caine. 
Set in a dystopian future where Earth is suffering from catastrophic blight and famine, 
the film follows a group of astronauts who travel through space in search of a new home for humanity.

The film began development in 2006 from a treatment by producer Lynda Obst and theoretical physicist Kip Thorne,
 and was originally set to be directed by Steven Spielberg. 
 Thorne served as an executive producer and scientific consultant on the film, 
 and wrote the tie-in book The Science of Interstellar. It was Obst's final film as producer before her death. Cinematographer Hoyte van Hoytema shot it on 35 mm film in the Panavision anamorphic format and IMAX 70 mm. Filming began in late 2013 and took place in Alberta, Klaustur, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company DNEG created additional visual effects.

Interstellar premiered at the TCL Chinese Theatre on October 26, 2014, 
and was released in theaters in the United States on November 5, and in the United Kingdom on November 7, 
with Paramount Pictures distributing in the United States and Warner Bros. 
Pictures distributing in international markets. In the United States, it was first released on film stock, 
expanding to venues using digital projectors. It was a commercial success, grossing $681 million worldwide 
during its initial theatrical run, and $774.7 million worldwide with subsequent releases, making it the 
10th-highest-grossing film of 2014. The film received generally positive reviews from critics. 
Among its various accolades, Interstellar was nominated for five awards at the 87th Academy Awards, winning 
for Best Visual Effects.
"""

# split the text into a list of sentences
texts = text.split('.')

# remove empty spaces and new lines
texts = [t.strip('\n') for t in texts]

# creating the embeddings for the text by sending them to Cohere API, it will return vector for each text
response = co.embed(texts=texts, model="embed-v4.0", input_type="search_document",).embeddings

embeds = np.array(response.float)
print(f"shape of the embeddings: {embeds.shape}")

# Now we need to build the search index before we can search. An index stores the embeddings 
# and is optimized to quickly retrieve the nearest neighbors even in the large number of points.
import faiss

dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
print(index.is_trained)
index.add(np.float32(embeds))

# Now we can search the index using any query. We have to first embed the query and present its embedding
# to the index, which will retrieve the most similar sentence from the text.

def search(query, number_of_results=3):

    # get the query's embedding
    query_embed = co.embed(texts=[query], model="embed-v4.0", input_type="search_query",).embeddings.float[0]

    # retrieve the nearest neighbors 
    distances, similiar_item_ids = index.search(np.float32([query_embed]), number_of_results)

    # format the result
    text_np = np.array(texts)
    results = pd.DataFrame(data={'texts':text_np[similiar_item_ids[0]],'distance': distances[0]})

    print(f"Query: '{query}'\nNearest neighbors:")
    return results

query = "how precise was the science"
results = search(query)
print(results)