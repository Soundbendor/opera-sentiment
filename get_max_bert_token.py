import os
from transformers import BertTokenizer, BertModel
from utilities.yamlhelp import safe_read_yaml
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

mother_path = "trimmed_30_Padding-S/ch"
token_sizes = []
output_sizes = []
for root, dirs, files in os.walk(mother_path):
    # print(dir)
    for dir in dirs:
        if "wav" in dir: # trimmed_30/ch/9/wav00/
            # check if "in" folder under it is empty
            if os.path.exists(os.path.join(root,dir,"in")): # for the case that the whole recording is dropped (due to being shorter than trimming size)
                data_dir = os.path.join(root,dir)
                yaml_file = os.path.join(root, "metadata.yaml")
                metadata = safe_read_yaml(yaml_file)
                english_lyrics = metadata['lyric']['english']
                # output = tokenizer(english_lyrics, return_tensors="pt", max_length=369, padding="max_length", truncation=True)
                output = tokenizer(english_lyrics, return_tensors="pt")
                token_sizes.append(output.input_ids.size()[1])
                model_output = model(**output).pooler_output
                output_sizes.append(model_output.size())
                break

print(token_sizes)
print(max(token_sizes))
print(output_sizes)
print(max(output_sizes))