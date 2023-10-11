from datasets import load_dataset
import pandas as pd
# import evaluate 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


# removing non english words

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

source = "es"
target = "en"

# test
text = ['RT <user> : When u walk straight into the kitchen to eat & ur mom hits u with the " ya saludaste " ',
'RT <user> : Estas son mis fotos favoritas de toda mi vida y no importa que hayan sido en enero del 2013 i can always relate',
'hoy empieza mi birthweek .. .',
'La vida es perra ya que aquí no existe gloria sin victoria ni victoria sin la guerra . NUEVO RAP PLAY PROXIMAMENTE !',
'<user> Nena queee . El best way to start a day es con un cacoteo intenso . #lamarasssh',
'<user> esque me fui de viaje a miami pero ya regresee lmao jaja jk y tambien esque somos mujeres ocupadas lmao',
'Comiendo mierda con cojones para hacer este paper',
'<user> ponte un gorrito de esos que son como una sombrilla hahaha',
'Simple Things//Miguel * mi nueva obsecion *',
'quiero Panda Express 👅']

tokenizer.src_lang = source
encoded_hi = tokenizer(text, return_tensors="pt", padding=True)
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.get_lang_id(target))
print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
