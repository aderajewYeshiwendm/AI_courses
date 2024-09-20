import os

def load_bbc_dataset(path):
    """Load BBC dataset from `path`"""
    categories = os.listdir(path)
    texts = []
    labels = []
    label_map = {category: idx for idx, category in enumerate(categories)}
    
    for category in categories:
        category_path = os.path.join(path, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            with open(file_path, 'r', encoding='latin1') as file:
                texts.append(file.read())
                labels.append(label_map[category])
    
    return texts, labels

bbc_texts, bbc_labels = load_bbc_dataset('path_to_bbc')
